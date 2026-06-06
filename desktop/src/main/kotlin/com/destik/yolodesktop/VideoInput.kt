package com.destik.yolodesktop

import org.bytedeco.javacv.FFmpegFrameGrabber
import org.bytedeco.javacv.Frame
import org.bytedeco.javacv.Java2DFrameConverter
import org.bytedeco.javacv.OpenCVFrameGrabber
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

typealias FrameCallback = (BufferedImage) -> Unit

/** Grabs frames from webcam or HTTP MJPEG stream, calls [onFrame] on a background thread. */
class VideoInput(
    private val source: String,           // "0", "1", ... for webcam index; "http://..." for stream
    private val onFrame: FrameCallback,
    private val onError: (String) -> Unit
) {
    private val running = AtomicBoolean(false)
    private var thread: Thread? = null

    fun start() {
        if (running.getAndSet(true)) return
        thread = Thread(null, ::loop, "video-input", 0).also {
            it.isDaemon = true
            it.start()
        }
    }

    fun stop() {
        running.set(false)
        thread?.interrupt()
        thread = null
    }

    private fun loop() {
        val isUrl = source.startsWith("rtsp", true) || source.startsWith("http", true)
        when {
            // Prefer the *system* ffmpeg for network streams: it's typically newer
            // and (on a Pi) a Raspberry-Pi build that decodes far more reliably than
            // JavaCV's bundled one, and can use the board's hardware decoder. Auto-on
            // when ffmpeg is on PATH; YOLO_HWDEC=on forces it, =off keeps JavaCV.
            isUrl && useSystemFfmpeg() -> {
                if (hasSystemFfmpeg()) runSystemFfmpegLoop()
                else {
                    onError("YOLO_HWDEC=on but 'ffmpeg' not found on PATH — falling back to JavaCV. Install with: sudo apt install -y ffmpeg")
                    if (source.startsWith("rtsp", true)) runRtspLoop() else runMjpegLoop()
                }
            }
            source.startsWith("rtsp", ignoreCase = true) -> runRtspLoop()
            source.startsWith("http", ignoreCase = true) -> runMjpegLoop()
            // CSI ribbon camera on a Raspberry Pi (libcamera stack): not a V4L2
            // /dev/videoN device, so OpenCV can't open it by index. Stream it via
            // rpicam-vid (MJPEG) and parse the JPEG frames from its stdout.
            source.equals("rpicam", true) || source.equals("libcamera", true) ||
                source.equals("csi", true) -> runRpicamLoop()
            else -> runWebcamLoop()
        }
    }

    /** Whether to route network streams through the system ffmpeg. YOLO_HWDEC:
     *  on/off explicit; otherwise auto = use it whenever ffmpeg is installed. */
    private fun useSystemFfmpeg(): Boolean =
        when (System.getenv("YOLO_HWDEC")?.trim()?.lowercase()) {
            "off", "no", "0"  -> false
            "on", "yes", "1"  -> true
            else              -> hasSystemFfmpeg()   // auto
        }

    /** RTSP transport: "tcp" (default) or "udp". On a lossy link UDP drops packets
     *  (corruption); TCP retransmits. Set YOLO_RTSP_TRANSPORT=udp only if TCP stalls. */
    private fun rtspTransport() =
        System.getenv("YOLO_RTSP_TRANSPORT")?.trim()?.lowercase()?.takeIf { it == "udp" || it == "tcp" } ?: "tcp"

    private fun hasSystemFfmpeg(): Boolean = runCatching {
        ProcessBuilder("ffmpeg", "-version").redirectErrorStream(true).start()
            .also { it.inputStream.readBytes(); it.waitFor() }.exitValue() == 0
    }.getOrDefault(false)

    /**
     * RTSP/HTTP decode via the system `ffmpeg`, which is far more robust than
     * JavaCV's bundled one and can use the board's hardware decoder. ffmpeg decodes
     * and re-encodes to MJPEG on stdout; we parse the JPEG frames (self-delimiting,
     * so no fragile fixed-size framing) and decode them to [BufferedImage].
     *
     * MJPEG output (not raw bgr24) is deliberate: it sidesteps the unaccelerated
     * yuv420p→bgr24 swscale path that re-inits per frame on this ffmpeg build and
     * stalls. It's exactly what the known-good `ffmpeg … out.jpg` test used.
     *
     * Decode mode:
     *   • Software (default) — most robust; the Pi 5 A76 handles 720p easily.
     *   • Hardware via V4L2 request API: YOLO_FFMPEG_HWACCEL=drm (+ optional
     *     YOLO_FFMPEG_HWDEV). hwdownload brings frames back for the MJPEG encoder.
     *   • Explicit decoder: YOLO_FFMPEG_DECODER.
     * Optional scaling only if YOLO_CAM_W/H are set (default: keep native size).
     */
    private fun runSystemFfmpegLoop() {
        // Effective decode config — may fall back to software if a HW attempt yields
        // no frames, so a bad YOLO_FFMPEG_* combo can't trap us in a retry loop.
        var decoder = System.getenv("YOLO_FFMPEG_DECODER")?.trim()?.takeIf { it.isNotEmpty() }
        var hwaccel = System.getenv("YOLO_FFMPEG_HWACCEL")?.trim()?.takeIf { it.isNotEmpty() }
        val hwdev   = System.getenv("YOLO_FFMPEG_HWDEV")?.trim()?.takeIf { it.isNotEmpty() }
        val scale   = if (System.getenv("YOLO_CAM_W") != null || System.getenv("YOLO_CAM_H") != null)
            "scale=${camW()}:${camH()}" else null
        while (running.get() && !Thread.currentThread().isInterrupted) {
            // hwaccel frames may live in GPU memory; hwdownload pulls them back so the
            // MJPEG encoder (a CPU filter) can consume them.
            val vf = listOfNotNull(hwaccel?.let { "hwdownload" }, scale).joinToString(",")
            val cmd = buildList {
                add("ffmpeg"); add("-hide_banner"); add("-loglevel"); add("warning")
                if (hwaccel != null) { add("-hwaccel"); add(hwaccel) }
                if (hwdev != null && hwaccel != null) { add("-hwaccel_device"); add(hwdev) }
                if (source.startsWith("rtsp", true)) { add("-rtsp_transport"); add(rtspTransport()) }
                // NB: no -fflags nobuffer / -flags low_delay by default — they make
                // ffmpeg drop packets under jitter ("max delay reached" → corruption).
                // Opt into low latency (at the cost of robustness) with YOLO_LOWLATENCY=on.
                if (System.getenv("YOLO_LOWLATENCY")?.trim()?.lowercase() == "on") {
                    add("-fflags"); add("nobuffer"); add("-flags"); add("low_delay")
                }
                if (decoder != null) { add("-c:v"); add(decoder) }   // force decoder if asked
                add("-i"); add(source)
                add("-an")                                            // no audio
                if (vf.isNotEmpty()) { add("-vf"); add(vf) }
                add("-c:v"); add("mjpeg"); add("-q:v"); add("5")      // intermediate JPEG (re-encoded for the browser anyway)
                // stdout is a non-seekable pipe: ffmpeg buffers output and only ever
                // flushed the first frame. flush_packets forces it to push every frame
                // to the pipe immediately (now safe — pipe draining is decoupled/fast).
                add("-flush_packets"); add("1")
                add("-f"); add("image2pipe"); add("-")
            }
            val mode = hwaccel?.let { "hwaccel=$it" } ?: decoder?.let { "decoder=$it" } ?: "software"
            println("  RTSP: system ffmpeg decode ($mode), MJPEG pipe")
            var proc: Process? = null
            var frames = 0L
            val latestJpeg = AtomicReference<ByteArray?>(null)
            val lastFrameAt = java.util.concurrent.atomic.AtomicLong(System.currentTimeMillis())
            val sessionRunning = AtomicBoolean(true)
            var decodeThread: Thread? = null
            var watchdog: Thread? = null
            try {
                proc = ProcessBuilder(cmd).redirectErrorStream(false).start()
                val p = proc
                // Surface ffmpeg's stderr so the user can see whether HW decode engaged.
                val err = proc.errorStream
                Thread { runCatching { err.bufferedReader().forEachLine { System.err.println("ffmpeg: $it") } } }
                    .apply { isDaemon = true; start() }
                // Decoder: decodes only the LATEST jpeg (drops any it can't keep up
                // with) so pipe-draining never waits on the slow ImageIO decode.
                decodeThread = Thread({
                    var dec = 0L
                    while (sessionRunning.get() && running.get() && !Thread.currentThread().isInterrupted) {
                        val jpg = latestJpeg.getAndSet(null)
                        if (jpg == null) { try { Thread.sleep(2) } catch (e: InterruptedException) { break }; continue }
                        val img = runCatching { javax.imageio.ImageIO.read(jpg.inputStream()) }.getOrNull() ?: continue
                        onFrame(ensureRgb(img))
                        if (++dec == 1L) println("  RTSP: first frame decoded ${img.width}x${img.height} — OK")
                    }
                }, "mjpeg-decode").apply { isDaemon = true; start() }
                // Watchdog: if the pipe stalls (frames stop arriving) kill ffmpeg so we
                // reconnect instead of hanging forever on a blocked read.
                watchdog = Thread({
                    while (sessionRunning.get() && running.get()) {
                        try { Thread.sleep(1000) } catch (e: InterruptedException) { break }
                        if (System.currentTimeMillis() - lastFrameAt.get() > 6000) {
                            System.err.println("RTSP: pipe stalled >6s — restarting ffmpeg")
                            runCatching { p.destroyForcibly() }
                            break
                        }
                    }
                }, "mjpeg-watchdog").apply { isDaemon = true; start() }
                // Drain the pipe fast (no decode here) → ffmpeg never blocks on write,
                // so the RTSP session stays healthy (no more CSeq errors).
                frames = drainJpegPipe(proc.inputStream, latestJpeg, lastFrameAt)
            } catch (e: InterruptedException) {
                break
            } catch (e: Exception) {
                if (!running.get()) break
                onError("ffmpeg decode error: ${e.message} — reconnecting…")
            } finally {
                sessionRunning.set(false)
                runCatching { decodeThread?.interrupt() }
                runCatching { watchdog?.interrupt() }
                runCatching { proc?.destroy() }
            }
            if (!running.get()) break
            // If a HW attempt never produced a frame, that hwaccel/decoder isn't
            // usable on this build — drop it and continue in pure software.
            if (frames == 0L && (hwaccel != null || decoder != null)) {
                onError("ffmpeg ($mode) produced no frames — falling back to software decode")
                hwaccel = null; decoder = null
            }
            try { Thread.sleep(1500) } catch (_: InterruptedException) { break }  // reconnect backoff
        }
    }

    /**
     * Drains an MJPEG pipe, extracting complete JPEG frames (SOI 0xFFD8 … EOI
     * 0xFFD9) and publishing each into [latest] (latest-wins, older frames dropped).
     * Does NOT decode — that's the decode thread's job — so this loop is fast enough
     * to keep the pipe empty and stop ffmpeg ever blocking on write. Reads in big
     * chunks. Returns frames seen; stops on EOF / stop / interrupt.
     */
    private fun drainJpegPipe(
        stream: InputStream,
        latest: AtomicReference<ByteArray?>,
        lastFrameAt: java.util.concurrent.atomic.AtomicLong
    ): Long {
        val buf = ByteArray(64 * 1024)
        val frame = ByteArrayOutputStream(1 shl 19)
        var prev = -1; var inFrame = false; var count = 0L; var bytes = 0L
        while (running.get() && !Thread.currentThread().isInterrupted) {
            val n = stream.read(buf)
            if (n < 0) break
            bytes += n
            for (i in 0 until n) {
                val b = buf[i].toInt() and 0xFF
                if (!inFrame) {
                    if (prev == 0xFF && b == 0xD8) {        // start of image
                        inFrame = true; frame.reset(); frame.write(0xFF); frame.write(0xD8)
                    }
                } else {
                    frame.write(b)
                    if (prev == 0xFF && b == 0xD9) {        // end of image
                        latest.set(frame.toByteArray()); count++; lastFrameAt.set(System.currentTimeMillis())
                        if (count == 1L) println("  RTSP: first frame from ffmpeg — pipe OK")
                        else if (count % 100L == 0L) println("  RTSP: $count frames from ffmpeg (pipe bytes=$bytes)")
                        inFrame = false
                    }
                }
                prev = b
            }
        }
        return count
    }


    /** RTSP stream (e.g. SIYI ZR10: rtsp://192.168.144.25:8554/main.264) via FFmpeg. */
    private fun runRtspLoop() {
        // An H.265 stream can only be decoded starting from a keyframe (IDR). When
        // we connect we almost always join mid-GOP, so the decoder has no reference
        // and every frame until the next keyframe is garbage ("Could not find ref
        // with POC …" — green/black/smeared). The fix: drop everything until we've
        // seen the first keyframe, then the picture decodes cleanly. If no keyframe
        // (or no frame at all) arrives within staleMs, reconnect — a fresh RTSP
        // session usually makes the camera emit a new IDR right away.
        val staleMs = 5000L
        while (running.get() && !Thread.currentThread().isInterrupted) {
            val grabber = FFmpegFrameGrabber(source).apply {
                format = "rtsp"
                setOption("rtsp_transport", rtspTransport())   // tcp default; udp if it loses packets
                setOption("stimeout", "5000000")      // 5s socket timeout (microseconds)
                setOption("fflags", "nobuffer")       // don't buffer — keep latency low
                setOption("flags", "low_delay")
                // A small reorder queue lets the decoder rebuild after a lost
                // reference instead of freezing, while keeping latency low.
                setOption("reorder_queue_size", "16")
                setOption("probesize", "100000")      // start fast, don't pre-buffer
                setOption("analyzeduration", "0")
            }
            try {
                grabber.start()
                val converter = Java2DFrameConverter()
                val connectAt = System.currentTimeMillis()
                var lastFrameAt = connectAt
                var streaming = false
                // Give the decoder up to this long to flag a keyframe; if none is
                // reported (long GOP, or grabImage doesn't set the flag on this
                // build) we start streaming anyway rather than hang on a black
                // screen — the decoder cleans itself up at the next real IDR.
                val keyframeWaitMs = 3000L
                while (running.get() && !Thread.currentThread().isInterrupted) {
                    val frame: Frame? = grabber.grabImage()
                    if (frame == null) {
                        if (System.currentTimeMillis() - lastFrameAt > staleMs) {
                            onError("RTSP: no ${if (streaming) "frames" else "keyframe"} ${staleMs}ms — reconnecting…")
                            break
                        }
                        continue
                    }
                    // Hold off display until the first keyframe so we never show the
                    // garbled mid-GOP frames the decoder can't reconstruct — but never
                    // wait forever: release the gate after keyframeWaitMs regardless.
                    if (!streaming) {
                        when {
                            frame.keyFrame -> println("  RTSP: locked onto keyframe — streaming")
                            System.currentTimeMillis() - connectAt < keyframeWaitMs -> continue
                            else -> println("  RTSP: no keyframe flag after ${keyframeWaitMs}ms — streaming anyway")
                        }
                        streaming = true
                    }
                    val img = converter.convert(frame) ?: continue
                    lastFrameAt = System.currentTimeMillis()
                    onFrame(ensureRgb(img))
                }
            } catch (e: InterruptedException) {
                break
            } catch (e: Exception) {
                if (!running.get()) break
                onError("RTSP error: ${e.message} — reconnecting…")
                try { Thread.sleep(2000) } catch (_: InterruptedException) { break }
            } finally {
                runCatching { grabber.stop() }
                runCatching { grabber.release() }
            }
        }
    }

    /** Raspberry Pi CSI camera via rpicam-vid / libcamera-vid → MJPEG on stdout. */
    private fun runRpicamLoop() {
        val w = camW(); val h = camH(); val fps = camFps()
        val args = listOf(
            "-t", "0", "--codec", "mjpeg", "--nopreview",
            "--width", "$w", "--height", "$h", "--framerate", "$fps", "-o", "-"
        )
        var proc: Process? = null
        try {
            proc = startFirstAvailable(listOf("rpicam-vid", "libcamera-vid"), args) ?: run {
                onError("rpicam-vid / libcamera-vid not found — install with: sudo apt install -y rpicam-apps")
                return
            }
            // Drain stderr so the process never blocks on a full error pipe.
            val errStream = proc.errorStream
            Thread { runCatching { errStream.bufferedReader().forEachLine { } } }
                .apply { isDaemon = true; start() }

            val inp = proc.inputStream.buffered(1 shl 16)
            val frame = ByteArrayOutputStream(1 shl 18)
            var prev = -1
            var inFrame = false
            while (running.get() && !Thread.currentThread().isInterrupted) {
                val b = inp.read()
                if (b == -1) break
                if (!inFrame) {
                    if (prev == 0xFF && b == 0xD8) {        // JPEG start-of-image
                        inFrame = true; frame.reset(); frame.write(0xFF); frame.write(0xD8)
                    }
                } else {
                    frame.write(b)
                    if (prev == 0xFF && b == 0xD9) {        // JPEG end-of-image
                        val img = javax.imageio.ImageIO.read(frame.toByteArray().inputStream())
                        if (img != null) onFrame(ensureRgb(img))
                        inFrame = false
                    }
                }
                prev = b
            }
        } catch (e: InterruptedException) {
            // normal stop
        } catch (e: Exception) {
            if (running.get()) onError("CSI camera error: ${e.message}")
        } finally {
            runCatching { proc?.destroy() }
        }
    }

    /** Tries each binary name in turn; returns the first that starts, or null. */
    private fun startFirstAvailable(bins: List<String>, args: List<String>): Process? {
        for (bin in bins) {
            val p = runCatching {
                ProcessBuilder(listOf(bin) + args).redirectErrorStream(false).start()
            }.getOrNull()
            if (p != null) return p
        }
        return null
    }

    // Requested capture geometry — applies to every source type. Decoupled
    // inference means a higher resolution costs only JPEG encode/bandwidth, not
    // detection FPS, so a decent default is affordable.
    private fun camW()   = System.getenv("YOLO_CAM_W")?.toIntOrNull()   ?: 1280
    private fun camH()   = System.getenv("YOLO_CAM_H")?.toIntOrNull()   ?: 720
    private fun camFps() = System.getenv("YOLO_CAM_FPS")?.toIntOrNull() ?: 30

    private fun runWebcamLoop() {
        val idx = source.toIntOrNull() ?: 0
        val grabber = OpenCVFrameGrabber(idx)
        grabber.imageWidth  = camW()
        grabber.imageHeight = camH()
        grabber.frameRate   = camFps().toDouble()
        try {
            grabber.start()
            val converter = Java2DFrameConverter()
            while (running.get() && !Thread.currentThread().isInterrupted) {
                val frame: Frame = grabber.grab() ?: continue
                val img = converter.convert(frame) ?: continue
                // ensure TYPE_INT_RGB for consistent processing
                val rgb = if (img.type == BufferedImage.TYPE_INT_RGB) img else {
                    BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_RGB).also { dst ->
                        dst.createGraphics().apply { drawImage(img, 0, 0, null); dispose() }
                    }
                }
                onFrame(rgb)
            }
        } catch (e: InterruptedException) {
            // normal stop
        } catch (e: Exception) {
            if (running.get()) onError("Webcam error: ${e.message}")
        } finally {
            runCatching { grabber.stop() }
        }
    }

    private fun runMjpegLoop() {
        while (running.get() && !Thread.currentThread().isInterrupted) {
            try {
                connectAndRead()
            } catch (_: InterruptedException) {
                break
            } catch (e: Exception) {
                if (!running.get()) break
                onError("Stream error: ${e.message} — reconnecting…")
                try { Thread.sleep(2000) } catch (_: InterruptedException) { break }
            }
        }
    }

    private fun connectAndRead() {
        val conn = (URL(source).openConnection() as HttpURLConnection).apply {
            connectTimeout = 5_000
            readTimeout = 10_000
            connect()
        }
        try {
            val ct = conn.contentType ?: ""
            if (ct.contains("jpeg") || ct.contains("jpg")) {
                val img = javax.imageio.ImageIO.read(conn.inputStream) ?: return
                onFrame(ensureRgb(img))
                return
            }
            val boundary = Regex("boundary=([^;\\s]+)", RegexOption.IGNORE_CASE)
                .find(ct)?.groupValues?.get(1)?.trimStart('-') ?: "mjpeg"
            val inp = conn.inputStream.buffered()
            while (running.get() && !Thread.currentThread().isInterrupted) {
                val img = readNextFrame(inp, boundary) ?: break
                onFrame(ensureRgb(img))
            }
        } finally {
            runCatching { conn.disconnect() }
        }
    }

    private fun readNextFrame(inp: InputStream, boundary: String): BufferedImage? {
        var contentLength = -1
        // read headers until blank line
        while (true) {
            val line = readLine(inp) ?: return null
            if (line.isEmpty()) break
            if (line.startsWith("Content-Length:", ignoreCase = true))
                contentLength = line.substringAfter(":").trim().toIntOrNull() ?: -1
        }
        return if (contentLength > 0) {
            val bytes = inp.readNBytes(contentLength)
            javax.imageio.ImageIO.read(bytes.inputStream())
        } else {
            // boundary-scan fallback
            val baos = ByteArrayOutputStream(64 * 1024)
            val marker = "--$boundary".toByteArray()
            while (true) {
                val b = inp.read()
                if (b == -1) break
                baos.write(b)
                val data = baos.toByteArray()
                if (data.size > marker.size && data.containsSequence(marker, data.size - marker.size - 10))
                    break
            }
            val data = baos.toByteArray()
            val end = data.indexOfSequence(marker)
            val jpegBytes = if (end > 0) data.copyOf(end) else data
            javax.imageio.ImageIO.read(jpegBytes.inputStream())
        }
    }

    private fun readLine(inp: InputStream): String? {
        val sb = StringBuilder()
        var prev = -1
        while (true) {
            val b = inp.read()
            if (b == -1) return null
            if (prev == '\r'.code && b == '\n'.code) return sb.dropLast(1).toString()
            sb.append(b.toChar())
            prev = b
        }
    }

    private fun ByteArray.indexOfSequence(seq: ByteArray): Int {
        outer@ for (i in 0..size - seq.size) {
            for (j in seq.indices) if (this[i + j] != seq[j]) continue@outer
            return i
        }
        return -1
    }

    private fun ByteArray.containsSequence(seq: ByteArray, from: Int): Boolean {
        val start = maxOf(0, from)
        outer@ for (i in start..size - seq.size) {
            for (j in seq.indices) if (this[i + j] != seq[j]) continue@outer
            return true
        }
        return false
    }

    private fun ensureRgb(img: BufferedImage) =
        if (img.type == BufferedImage.TYPE_INT_RGB) img else
            BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_RGB).also { dst ->
                dst.createGraphics().apply { drawImage(img, 0, 0, null); dispose() }
            }
}
