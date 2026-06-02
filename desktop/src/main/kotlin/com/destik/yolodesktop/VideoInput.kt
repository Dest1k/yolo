package com.destik.yolodesktop

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
        if (source.startsWith("http", ignoreCase = true)) {
            runMjpegLoop()
        } else {
            runWebcamLoop()
        }
    }

    private fun runWebcamLoop() {
        val idx = source.toIntOrNull() ?: 0
        val grabber = OpenCVFrameGrabber(idx)
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
            val buf = ByteArray(4096)
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
