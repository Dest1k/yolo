package com.destik.yolodetector

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import java.io.File
import java.nio.ByteBuffer

class VideoRecorder(
    val width: Int,
    val height: Int,
    private val outputFile: File,
    private val frameRate: Int = 15
) {
    enum class Mode { ALWAYS, SMART }

    private val TAG = "VideoRecorder"
    private val SMART_STOP_FRAMES = frameRate * 3   // 3 seconds

    private var codec: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var videoTrack = -1
    private var frameCount = 0L
    private var muxerStarted = false

    var recording = false; private set
    private var noDetectionFrames = 0
    private val nv12 = ByteArray(width * height * 3 / 2)

    fun start() {
        if (recording) return
        outputFile.parentFile?.mkdirs()
        try {
            // Ensure even dimensions (AVC requirement)
            val w = width and 1.inv()
            val h = height and 1.inv()
            val fmt = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
                setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000)
                setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
            }
            codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).also {
                it.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                it.start()
            }
            muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            frameCount = 0L; muxerStarted = false; recording = true; noDetectionFrames = 0
            Log.d(TAG, "started → ${outputFile.name}")
        } catch (e: Exception) {
            Log.e(TAG, "start failed", e); releaseInternal()
        }
    }

    fun stop(): File? {
        if (!recording) return null
        recording = false
        return try {
            drain(eos = true)
            muxer?.stop()
            outputFile.takeIf { it.exists() && it.length() > 1024 }
        } catch (e: Exception) {
            Log.e(TAG, "stop error", e); null
        } finally {
            releaseInternal()
        }
    }

    /** Feed a composed bitmap. In SMART mode auto-starts/stops on detections. Returns true if encoded. */
    fun feedFrame(bitmap: Bitmap, hasDetections: Boolean, mode: Mode): Boolean {
        when (mode) {
            Mode.SMART -> {
                if (hasDetections) {
                    noDetectionFrames = 0
                    if (!recording) start()
                } else {
                    if (!recording) return false
                    if (++noDetectionFrames > SMART_STOP_FRAMES) { stop(); return false }
                }
            }
            Mode.ALWAYS -> if (!recording) return false
        }
        return encodeFrame(bitmap)
    }

    private fun encodeFrame(bitmap: Bitmap): Boolean {
        val c = codec ?: return false
        return try {
            argbToNv12(bitmap, nv12)
            val inIdx = c.dequeueInputBuffer(10_000)
            if (inIdx < 0) return false
            val buf: ByteBuffer = c.getInputBuffer(inIdx) ?: return false
            buf.clear(); buf.put(nv12)
            c.queueInputBuffer(inIdx, 0, nv12.size, frameCount * 1_000_000L / frameRate, 0)
            frameCount++
            drain(eos = false)
            true
        } catch (e: Exception) {
            Log.e(TAG, "encode error", e); false
        }
    }

    private fun drain(eos: Boolean) {
        val c = codec ?: return
        if (eos) {
            val inIdx = c.dequeueInputBuffer(10_000)
            if (inIdx >= 0) c.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
        val info = MediaCodec.BufferInfo()
        while (true) {
            val idx = c.dequeueOutputBuffer(info, 10_000)
            when {
                idx == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    videoTrack = muxer!!.addTrack(c.outputFormat)
                    muxer!!.start(); muxerStarted = true
                }
                idx >= 0 -> {
                    if (muxerStarted && info.size > 0 &&
                        (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                        val out = c.getOutputBuffer(idx)
                        if (out != null) {
                            out.position(info.offset); out.limit(info.offset + info.size)
                            muxer!!.writeSampleData(videoTrack, out, info)
                        }
                    }
                    c.releaseOutputBuffer(idx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
        }
    }

    fun release() { if (recording) stop(); releaseInternal() }

    private fun releaseInternal() {
        recording = false
        runCatching { codec?.stop() }; runCatching { codec?.release() }
        runCatching { muxer?.release() }
        codec = null; muxer = null
    }

    private fun argbToNv12(bmp: Bitmap, out: ByteArray) {
        val w = bmp.width.coerceAtMost(width)
        val h = bmp.height.coerceAtMost(height)
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val uvBase = width * height
        var uvIdx = 0
        for (row in 0 until h) {
            for (col in 0 until w) {
                val p = pixels[row * w + col]
                val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                out[row * width + col] = ((66 * r + 129 * g + 25 * b + 128).shr(8) + 16).coerceIn(16, 235).toByte()
            }
        }
        for (row in 0 until h step 2) {
            for (col in 0 until w step 2) {
                val p = pixels[row * w + col]
                val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                out[uvBase + uvIdx]     = ((-38 * r - 74 * g + 112 * b + 128).shr(8) + 128).coerceIn(16, 240).toByte()
                out[uvBase + uvIdx + 1] = ((112 * r - 94 * g - 18 * b + 128).shr(8) + 128).coerceIn(16, 240).toByte()
                uvIdx += 2
            }
        }
    }
}
