package com.destik.yolodesktop

import java.awt.image.BufferedImage
import java.io.OutputStream
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Records the annotated video to an H.264 file using the board's *hardware*
 * encoder when one is present (Pi `h264_v4l2m2m`, `h264_nvenc`, VA-API…), falling
 * back to software `libx264 ultrafast`. Frames are fed in as raw RGB over a pipe
 * to the system `ffmpeg`, so the heavy encode runs in silicon instead of stealing
 * CPU from inference — the explicit goal on weak single-board computers.
 *
 * Backpressure-safe: if the encoder can't keep up, the newest frames win and old
 * ones are dropped, so recording never stalls the live stream/inference threads.
 *
 * Enable from [Headless] with YOLO_RECORD=/path/out.mp4 (bitrate via
 * YOLO_RECORD_BITRATE kbps, default 4000). Encoder can be forced with
 * YOLO_RECORD_ENCODER, else it's auto-selected by [HwAccel].
 */
class H264Recorder(
    private val path: String,
    private val width: Int,
    private val height: Int,
    private val fps: Int,
    private val bitrateKbps: Int = 4000
) {
    private val queue = ArrayBlockingQueue<BufferedImage>(4)
    private val running = AtomicBoolean(false)
    private var proc: Process? = null
    private var writer: Thread? = null
    private var rgb = IntArray(width * height)
    private var row = ByteArray(width * 3)

    /** The encoder actually selected (for logging) — hardware name or "libx264". */
    val encoder: String = System.getenv("YOLO_RECORD_ENCODER")?.trim()?.ifEmpty { null }
        ?: HwAccel.h264Encoder()

    /** Starts ffmpeg and the writer thread. Returns false if ffmpeg can't launch. */
    fun start(): Boolean {
        if (!HwAccel.ffmpegAvailable) {
            System.err.println("YOLO_RECORD set but 'ffmpeg' not found on PATH — recording disabled")
            return false
        }
        val cmd = buildList {
            add("ffmpeg"); add("-hide_banner"); add("-loglevel"); add("warning"); add("-y")
            add("-f"); add("rawvideo"); add("-pix_fmt"); add("rgb24")
            add("-s"); add("${width}x$height"); add("-r"); add("$fps")
            add("-i"); add("-")
            add("-an")
            addAll(HwAccel.encoderPreFilter(encoder))
            add("-c:v"); add(encoder)
            add("-b:v"); add("${bitrateKbps}k")
            add("-g"); add("${fps * 2}")
            if (encoder != "h264_vaapi") { add("-pix_fmt"); add("yuv420p") }
            add(path)
        }
        return try {
            val p = ProcessBuilder(cmd).redirectErrorStream(false).start()
            proc = p
            Thread { runCatching { p.errorStream.bufferedReader().forEachLine { System.err.println("ffmpeg-rec: $it") } } }
                .apply { isDaemon = true; start() }
            running.set(true)
            writer = Thread(::writeLoop, "h264-record").apply { isDaemon = true; start() }
            val mode = if (HwAccel.isHardware(encoder)) "hardware" else "software"
            println("  recording: $path  (H.264 $encoder, $mode, ${bitrateKbps}kbps)")
            true
        } catch (e: Exception) {
            System.err.println("recording start failed: ${e.message}")
            false
        }
    }

    /** Hand a frame to the recorder (latest-wins; never blocks the caller). */
    fun submit(frame: BufferedImage) {
        if (!running.get()) return
        if (!queue.offer(frame)) { queue.poll(); queue.offer(frame) }   // drop oldest, keep newest
    }

    fun stop() {
        running.set(false)
        writer?.interrupt()
        val p = proc; proc = null
        runCatching { p?.outputStream?.close() }
        runCatching { p?.waitFor(2, TimeUnit.SECONDS) }
        runCatching { p?.destroy() }
    }

    private fun writeLoop() {
        val out: OutputStream = proc?.outputStream ?: return
        try {
            while (running.get() && !Thread.currentThread().isInterrupted) {
                val img = queue.poll(200, TimeUnit.MILLISECONDS) ?: continue
                writeRgb24(out, img)
            }
        } catch (_: InterruptedException) {
            // normal stop
        } catch (e: Exception) {
            if (running.get()) System.err.println("recording write error: ${e.message}")
        } finally {
            runCatching { out.flush() }
        }
    }

    /** Stream one frame as packed RGB24, scaling/padding nothing (sizes match). */
    private fun writeRgb24(out: OutputStream, img: BufferedImage) {
        val w = img.width; val h = img.height
        if (w != width || h != height) return        // geometry changed mid-run: skip
        img.getRGB(0, 0, w, h, rgb, 0, w)
        for (y in 0 until h) {
            var r = 0
            val base = y * w
            for (x in 0 until w) {
                val p = rgb[base + x]
                row[r++] = ((p ushr 16) and 0xFF).toByte()
                row[r++] = ((p ushr 8) and 0xFF).toByte()
                row[r++] = (p and 0xFF).toByte()
            }
            out.write(row)
        }
        out.flush()
    }
}
