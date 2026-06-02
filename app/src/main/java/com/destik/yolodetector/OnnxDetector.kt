package com.destik.yolodetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.min
import kotlin.math.roundToInt

class OnnxDetector(private val config: ModelConfig) {

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var lastDiag = "onnx|init"
    private var provider = "cpu"

    // Reusable buffers — the detector runs single-threaded (inferenceExecutor),
    // so we allocate once and reuse to avoid per-frame GC pressure.
    private var padded: Bitmap? = null
    private var pixels: IntArray? = null
    private var floatArr: FloatArray? = null
    private val srcPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val dstRect = RectF()

    fun init(): Boolean {
        return try {
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(config.numThreads.coerceIn(1, 8))
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setMemoryPatternOptimization(true)

                if (config.useGPU) {
                    // NNAPI (GPU/NPU). FP16 lets the driver use half precision.
                    // Note: end-to-end / NMS-free models (with NMS/TopK ops) are
                    // partitioned by NNAPI and are usually FASTER on CPU — see below.
                    provider = try {
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                        "nnapi-fp16"
                    } catch (_: Throwable) {
                        addXnnpackOrCpu()
                    }
                } else {
                    provider = addXnnpackOrCpu()
                }
            }
            session = env.createSession(config.onnxPath, opts)
            lastDiag = "onnx|ready|$provider"
            true
        } catch (e: Exception) {
            lastDiag = "onnx|init_error|${e.message}"
            false
        }
    }

    /** Add XNNPACK CPU accelerator; returns provider tag, falling back to plain CPU. */
    private fun OrtSession.SessionOptions.addXnnpackOrCpu(): String = try {
        addXnnpack(mapOf("intra_op_num_threads" to config.numThreads.coerceIn(1, 8).toString()))
        "xnnpack"
    } catch (_: Throwable) {
        "cpu"
    }

    fun detect(bitmap: Bitmap): Array<Detection> {
        val sess = session ?: return emptyArray<Detection>().also { lastDiag = "onnx|no_session" }
        return try {
            val tPre0 = System.currentTimeMillis()
            val sz   = config.inputSize
            val bmpW = bitmap.width
            val bmpH = bitmap.height

            // Letterbox: scale to fit sz×sz, pad with gray 114 (matches YOLO training)
            val scale = min(sz.toFloat() / bmpW, sz.toFloat() / bmpH)
            val nw    = (bmpW * scale).roundToInt()
            val nh    = (bmpH * scale).roundToInt()
            val padX  = (sz - nw) / 2
            val padY  = (sz - nh) / 2

            // Reusable padded canvas — recreate only if input size changed.
            val pad = padded?.takeIf { it.width == sz && !it.isRecycled }
                ?: Bitmap.createBitmap(sz, sz, Bitmap.Config.ARGB_8888).also { padded = it }
            Canvas(pad).apply {
                drawColor(Color.rgb(114, 114, 114))
                dstRect.set(padX.toFloat(), padY.toFloat(), (padX + nw).toFloat(), (padY + nh).toFloat())
                drawBitmap(bitmap, null, dstRect, srcPaint)   // scales in one pass, no extra alloc
            }

            val n   = sz * sz
            val px  = pixels?.takeIf { it.size == n } ?: IntArray(n).also { pixels = it }
            pad.getPixels(px, 0, sz, 0, 0, sz, sz)

            val arr = floatArr?.takeIf { it.size == 3 * n } ?: FloatArray(3 * n).also { floatArr = it }
            for (i in 0 until n) {
                val p = px[i]
                arr[i]         = ((p ushr 16) and 0xFF) / 255f
                arr[n + i]     = ((p ushr 8)  and 0xFF) / 255f
                arr[2 * n + i] = ( p          and 0xFF) / 255f
            }

            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(arr),
                longArrayOf(1L, 3L, sz.toLong(), sz.toLong()))

            val preMs = System.currentTimeMillis() - tPre0
            val inputName = sess.inputNames.first()
            val tRun0 = System.currentTimeMillis()
            sess.run(mapOf(inputName to inputTensor)).use { results ->
                inputTensor.close()
                val runMs = System.currentTimeMillis() - tRun0
                (results.first().value as OnnxTensor).use { out ->
                    val dets = postprocess(out, sz, padX, padY, scale, bmpW, bmpH)
                    // pre = letterbox+normalize, run = the actual model. If run >> pre,
                    // the model itself is the bottleneck (e.g. in-graph NMS), not our code.
                    lastDiag = "$lastDiag|pre:${preMs}ms|run:${runMs}ms"
                    dets
                }
            }
        } catch (e: Exception) {
            lastDiag = "onnx|error|${e.message}"
            emptyArray()
        }
    }

    private fun postprocess(
        out: OnnxTensor, sz: Int, padX: Int, padY: Int,
        scale: Float, bmpW: Int, bmpH: Int
    ): Array<Detection> {
        val shape = out.info.shape   // [1, A, B]
        val buf   = out.floatBuffer
        val ct    = config.confThreshold
        if (shape.size < 3) return emptyArray()
        val A = shape[1].toInt()
        val B = shape[2].toInt()

        // NMS-free / end-to-end models emit 6 attributes (x1,y1,x2,y2,score,cls)
        // with a small detection count; anchor-free (v8/v11) emit 4+nc attributes
        // across thousands of anchors.
        return if ((B == 6 && A in 7..2000) || (A == 6 && B in 7..2000)) {
            parseNmsFree(buf, A, B, sz, padX, padY, scale, bmpW, bmpH, ct)
        } else {
            parseAnchorFree(buf, A, B, sz, padX, padY, scale, bmpW, bmpH, ct)
        }
    }

    private fun parseNmsFree(
        buf: FloatBuffer, A: Int, B: Int,
        sz: Int, padX: Int, padY: Int, scale: Float, bmpW: Int, bmpH: Int, ct: Float
    ): Array<Detection> {
        // Attributes (6) live on the smaller axis; detections on the larger one.
        val tr = (A < B)
        val nd = if (tr) B else A

        // Auto-detect pixel vs normalised coordinates
        var pixelCoords = false
        for (i in 0 until min(nd, 100)) {
            val x2 = if (tr) buf[2 * nd + i] else buf[i * 6 + 2]
            if (!x2.isNaN() && x2 > 1.5f) { pixelCoords = true; break }
        }
        val sc = if (pixelCoords) 1f / sz else 1f

        buf.rewind()
        var maxC = 0f
        val dets = mutableListOf<Detection>()
        for (i in 0 until nd) {
            val x1    = get6(tr, 0, nd, i, buf) * sc
            val y1    = get6(tr, 1, nd, i, buf) * sc
            val x2    = get6(tr, 2, nd, i, buf) * sc
            val y2    = get6(tr, 3, nd, i, buf) * sc
            val score = get6(tr, 4, nd, i, buf)
            val cid   = get6(tr, 5, nd, i, buf)
            if (score.isNaN() || score.isInfinite()) continue
            if (score > maxC) maxC = score
            if (score < ct || x2 <= x1 || y2 <= y1) continue
            dets += Detection(
                x  = (x1 * sz - padX) / (scale * bmpW),
                y  = (y1 * sz - padY) / (scale * bmpH),
                w  = (x2 - x1) * sz / (scale * bmpW),
                h  = (y2 - y1) * sz / (scale * bmpH),
                label      = cid.toInt().coerceIn(0, 65535),
                confidence = score
            )
        }
        lastDiag = "onnx|nms-free|$nd|maxC:${"%.2f".format(maxC)}|dets:${dets.size}|$provider"
        return dets.toTypedArray()
    }

    private fun parseAnchorFree(
        buf: FloatBuffer, A: Int, B: Int,
        sz: Int, padX: Int, padY: Int, scale: Float, bmpW: Int, bmpH: Int, ct: Float
    ): Array<Detection> {
        // Attributes (4+nc) live on the smaller axis; anchors on the larger one.
        // Derive nc from the shape so a mis-configured numClasses can't swap the axes.
        val tr  = (A < B)
        val nd  = if (tr) B else A
        val attrs = if (tr) A else B
        val nc  = (attrs - 4).coerceAtLeast(1)

        buf.rewind()
        val raw = mutableListOf<Detection>()
        for (i in 0 until nd) {
            val cx = getN(tr, 0, nd, attrs, i, buf)
            val cy = getN(tr, 1, nd, attrs, i, buf)
            val bw = getN(tr, 2, nd, attrs, i, buf)
            val bh = getN(tr, 3, nd, attrs, i, buf)
            if (bw <= 0f || bh <= 0f || cx.isNaN()) continue
            var best = ct; var cls = -1
            for (c in 0 until nc) {
                val s = getN(tr, 4 + c, nd, attrs, i, buf)
                if (s > best) { best = s; cls = c }
            }
            if (cls < 0) continue
            raw += Detection(
                x  = ((cx - bw * .5f) - padX) / (scale * bmpW),
                y  = ((cy - bh * .5f) - padY) / (scale * bmpH),
                w  = bw / (scale * bmpW),
                h  = bh / (scale * bmpH),
                label      = cls,
                confidence = best
            )
        }
        // Greedy NMS
        val sorted = raw.sortedByDescending { it.confidence }
        val keep   = BooleanArray(sorted.size) { true }
        val result = mutableListOf<Detection>()
        for (i in sorted.indices) {
            if (!keep[i]) continue
            result += sorted[i]
            for (j in i + 1 until sorted.size) {
                if (keep[j] && sorted[i].label == sorted[j].label && iou(sorted[i], sorted[j]) > config.nmsThreshold)
                    keep[j] = false
            }
        }
        lastDiag = "onnx|v8|$nd|nc=$nc|maxC:${"%.2f".format(raw.maxOfOrNull { it.confidence } ?: 0f)}|dets:${result.size}|$provider"
        return result.toTypedArray()
    }

    fun getDiagnostics(): String = lastDiag

    fun release() {
        session?.close(); session = null
        padded?.recycle(); padded = null
        pixels = null; floatArr = null
    }

    // Element access helpers
    private fun get6(tr: Boolean, attr: Int, nd: Int, i: Int, buf: FloatBuffer): Float =
        if (tr) buf[attr * nd + i] else buf[i * 6 + attr]

    private fun getN(tr: Boolean, attr: Int, nd: Int, attrs: Int, i: Int, buf: FloatBuffer): Float =
        if (tr) buf[attr * nd + i] else buf[i * attrs + attr]

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = maxOf(a.x, b.x);         val iy1 = maxOf(a.y, b.y)
        val ix2 = minOf(a.x + a.w, b.x + b.w); val iy2 = minOf(a.y + a.h, b.y + b.h)
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        val union = a.w * a.h + b.w * b.h - inter
        return if (union <= 0f) 0f else inter / union
    }
}
