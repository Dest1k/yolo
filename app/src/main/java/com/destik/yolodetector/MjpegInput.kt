package com.destik.yolodetector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Reads an MJPEG-over-HTTP stream and delivers decoded Bitmap frames.
 * Supports: multipart/x-mixed-replace (standard MJPEG), also handles
 * plain JPEG responses (single frame, reconnects automatically).
 */
class MjpegInput(private val url: String) {

    private val TAG = "MjpegInput"

    @Volatile var running = false; private set
    private var thread: Thread? = null

    fun start(onFrame: (Bitmap) -> Unit, onError: (String) -> Unit) {
        if (running) return
        running = true
        thread = Thread {
            while (running) {
                try {
                    connect(onFrame)
                } catch (e: InterruptedException) {
                    break
                } catch (e: Exception) {
                    if (!running) break
                    Log.w(TAG, "stream error, reconnecting in 2s: ${e.message}")
                    onError("Переподключение… ${e.message}")
                    Thread.sleep(2_000)
                }
            }
        }.also { it.isDaemon = true; it.start() }
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
    }

    private fun connect(onFrame: (Bitmap) -> Unit) {
        Log.d(TAG, "connecting: $url")
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 10_000
            readTimeout    = 15_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "YoloDetector/1.0")
            connect()
        }
        val code = conn.responseCode
        if (code !in 200..299) throw Exception("HTTP $code")

        val contentType = conn.contentType ?: ""
        Log.d(TAG, "connected, Content-Type: $contentType")

        val boundary = if (contentType.contains("boundary=", ignoreCase = true)) {
            "--" + contentType.substringAfter("boundary=", "").trim().trimStart('-')
        } else "--frame"

        val input = BufferedInputStream(conn.inputStream, 65536)

        if (contentType.startsWith("image/jpeg", ignoreCase = true)) {
            // Single JPEG response (some cameras return one frame at a time)
            val bmp = BitmapFactory.decodeStream(input)
            if (bmp != null) onFrame(bmp)
            conn.disconnect()
            return
        }

        // multipart/x-mixed-replace stream
        readMultipart(input, boundary, onFrame)
        conn.disconnect()
    }

    private fun readMultipart(
        input: BufferedInputStream,
        boundary: String,
        onFrame: (Bitmap) -> Unit
    ) {
        val boundaryToken = boundary.trimStart('-').lowercase()   // compare case-insensitive

        while (running) {
            // ── Find boundary line ──────────────────────────────────────────
            val line = readAsciiLine(input) ?: break
            if (boundaryToken !in line.lowercase()) continue

            // ── Read part headers ───────────────────────────────────────────
            var contentLength = -1
            while (true) {
                val hdr = readAsciiLine(input) ?: return
                if (hdr.isEmpty()) break
                if (hdr.startsWith("Content-Length:", ignoreCase = true))
                    contentLength = hdr.substringAfter(":").trim().toIntOrNull() ?: -1
            }

            // ── Read JPEG bytes ─────────────────────────────────────────────
            val jpeg: ByteArray = when {
                contentLength > 0 -> readExact(input, contentLength) ?: break
                else              -> readUntilBoundary(input, boundary) ?: break
            }

            val bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: continue
            onFrame(bmp)
        }
    }

    private fun readExact(input: BufferedInputStream, length: Int): ByteArray? {
        val buf = ByteArray(length)
        var offset = 0
        while (offset < length && running) {
            val n = input.read(buf, offset, length - offset)
            if (n == -1) return null
            offset += n
        }
        return buf
    }

    private fun readUntilBoundary(input: BufferedInputStream, boundary: String): ByteArray? {
        val baos = ByteArrayOutputStream(32768)
        val sep = boundary.toByteArray(Charsets.ISO_8859_1)
        val tmp = ByteArray(4096)
        while (running) {
            val n = input.read(tmp)
            if (n == -1) return null
            baos.write(tmp, 0, n)
            if (baos.size() > 2_000_000) break   // guard against runaway
            val idx = indexOfBytes(baos.toByteArray(), sep)
            if (idx >= 0) return baos.toByteArray().copyOf(idx)
        }
        return null
    }

    /** Read one CR/LF-terminated ASCII line from a binary stream. */
    private fun readAsciiLine(input: BufferedInputStream): String? {
        val sb = StringBuilder(128)
        while (running) {
            val b = input.read()
            if (b == -1) return null
            if (b == '\n'.code) return sb.toString().trimEnd('\r')
            sb.append(b.toChar())
        }
        return null
    }

    private fun indexOfBytes(data: ByteArray, pattern: ByteArray): Int {
        outer@ for (i in 0..data.size - pattern.size) {
            for (j in pattern.indices) { if (data[i + j] != pattern[j]) continue@outer }
            return i
        }
        return -1
    }
}
