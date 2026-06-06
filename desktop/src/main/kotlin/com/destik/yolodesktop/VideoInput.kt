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
        when {
            // Hardware-decode path (YOLO_HWDEC=on): pipe through the *system* ffmpeg
            // so it can use the board's hardware video decoder (e.g. the Pi 5's HEVC
            // block) instead of JavaCV's CPU-only bundled ffmpeg. Works for rtsp://
            // and http(s):// URLs. Falls back below if ffmpeg isn't on PATH.
            hwDecRequested() && (source.startsWith("rtsp", true) || source.startsWith("http", true)) -> {
                if (hasSystemFfmpeg()) runSystemFfmpegLoop()
                else {
                    onError("YOLO_HWDEC=on but 'ffmpeg' not found on PATH — falling back to software decode. Install with: sudo apt install -y ffmpeg")
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

    private fun hwDecRequested() = System.getenv("YOLO_HWDEC")?.trim()?.lowercase() == "on"

    /** RTSP transport: "tcp" (default) or "udp". Set YOLO_RTSP_TRANSPORT=udp if a
     *  TCP-interleaved feed (e.g. SIYI/LIVE555) loses packets after the keyframe. */
    private fun rtspTransport() =
        System.getenv("YOLO_RTSP_TRANSPORT")?.trim()?.lowercase()?.takeIf { it == "udp" || it == "tcp" } ?: "tcp"

    private fun hasSystemFfmpeg(): Boolean = runCatching {
        ProcessBuilder("ffmpeg", "-version").redirectErrorStream(true).start()
            .also { it.inputStream.readBytes(); it.waitFor() }.exitValue() == 0
    }.getOrDefault(false)

    /**
     * RTSP/HTTP decode via the system `ffmpeg`, which is far more robust than
     * JavaCV's bundled one and can use the board's hardware decoder. ffmpeg decodes
     * and emits raw BGR frames to stdout; we read them at a fixed size and wrap them
     * as [BufferedImage].
     *
     * Decode mode (in order of preference for a Pi 5 HEVC stream):
     *   • Software (default): just YOLO_HWDEC=on — most robust, the A76 handles 720p.
     *   • Hardware via the V4L2 request API: YOLO_FFMPEG_HWACCEL=drm (the Pi 5's HEVC
     *     block is a *stateless* decoder, reached through -hwaccel, NOT the old
     *     hevc_v4l2m2m m2m device which only exists on the Pi 4). Optional
     *     YOLO_FFMPEG_HWDEV (e.g. /dev/dri/card0). We add hwdownload so the frames
     *     come back to CPU memory for the BGR pipe.
     *   • Explicit decoder: YOLO_FFMPEG_DECODER (e.g. hevc on a build that maps it to HW).
     * Output geometry is YOLO_CAM_W/H (default 1280x720, the ZR10 main stream size).
     */
    private fun runSystemFfmpegLoop() {
        val w = camW(); val h = camH()
        val frameBytes = w * h * 3
        // Effective decode config — may fall back to software if a HW attempt yields
        // no frames, so a bad YOLO_FFMPEG_* combo can't trap us in a retry loop.
        var decoder = System.getenv("YOLO_FFMPEG_DECODER")?.trim()?.takeIf { it.isNotEmpty() }
        var hwaccel = System.getenv("YOLO_FFMPEG_HWACCEL")?.trim()?.takeIf { it.isNotEmpty() }
        val hwdev   = System.getenv("YOLO_FFMPEG_HWDEV")?.trim()?.takeIf { it.isNotEmpty() }
        while (running.get() && !Thread.currentThread().isInterrupted) {
            // With a hwaccel the decoded frame may live in GPU memory; hwdownload
            // pulls it back so scale→bgr24 can run on the CPU side for our raw pipe.
            val vf = if (hwaccel != null) "hwdownload,scale=$w:$h" else "scale=$w:$h"
            val cmd = buildList {
                add("ffmpeg"); add("-hide_banner"); add("-loglevel"); add("warning")
                if (hwaccel != null) { add("-hwaccel"); add(hwaccel) }
                if (hwdev != null && hwaccel != null) { add("-hwaccel_device"); add(hwdev) }
                if (source.startsWith("rtsp", true)) { add("-rtsp_transport"); add(rtspTransport()) }
                add("-fflags"); add("nobuffer"); add("-flags"); add("low_delay")
                if (decoder != null) { add("-c:v"); add(decoder) }   // force decoder if asked
                add("-i"); add(source)
                add("-an")                                            // no audio
                add("-vf"); add(vf)                                   // fixed size → fixed frame bytes
                add("-pix_fmt"); add("bgr24"); add("-f"); add("rawvideo"); add("-")
            }
            val mode = hwaccel?.let { "hwaccel=$it" } ?: decoder?.let { "decoder=$it" } ?: "software"
            println("  RTSP: system ffmpeg decode ($mode) → ${w}x$h")
            var proc: Process? = null
            var frames = 0L
            try {
                proc = ProcessBuilder(cmd).redirectErrorStream(false).start()
                // Surface ffmpeg's stderr so the user can see whether HW decode engaged.
                val err = proc.errorStream
                Thread { runCatching { err.bufferedReader().forEachLine { System.err.println("ffmpeg: $it") } } }
                    .apply { isDaemon = true; start() }
                val inp = proc.inputStream
                val buf = ByteArray(frameBytes)
                while (running.get() && !Thread.currentThread().isInterrupted) {
                    if (!readFully(inp, buf)) break              // pipe closed / ffmpeg exited
                    frames++
                    onFrame(bgrToImage(buf, w, h))
                }
            } catch (e: InterruptedException) {
                break
            } catch (e: Exception) {
                if (!running.get()) break
                onError("ffmpeg decode error: ${e.message} — reconnecting…")
            } finally {
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

    /** Reads exactly buf.size bytes; returns false on EOF before the buffer fills. */
    private fun readFully(inp: InputStream, buf: ByteArray): Boolean {
        var off = 0
        while (off < buf.size) {
            val n = inp.read(buf, off, buf.size - off)
            if (n < 0) return false
            off += n
        }
        return true
    }

    /** Wraps a packed bgr24 buffer as a BufferedImage (no per-pixel copy). */
    private fun bgrToImage(buf: ByteArray, w: Int, h: Int): BufferedImage {
        val img = BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR)
        val dst = (img.raster.dataBuffer as java.awt.image.DataBufferByte).data
        System.arraycopy(buf, 0, dst, 0, minOf(buf.size, dst.size))
        return img
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
