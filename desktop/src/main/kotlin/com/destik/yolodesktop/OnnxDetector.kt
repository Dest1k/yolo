package com.destik.yolodesktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.roundToInt

data class Detection(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float, val cls: Int)

/**
 * ONNX Runtime detector with optional GPU acceleration.
 *
 * Handles both common YOLO export layouts:
 *  - NMS-free / end-to-end [1,N,6] = x1,y1,x2,y2,score,cls (YOLOv10)
 *  - anchor-free [1,4+nc,N] / [1,N,4+nc] = cx,cy,w,h,cls… (YOLOv8/v9/v11, NMS here)
 *
 * Per-frame buffers (padded image, pixel + float arrays) are reused so a slow
 * single-board CPU isn't fighting the garbage collector on every frame.
 *
 * GPU modes: AUTO (CUDA→DirectML→CPU), CUDA, DIRECTML, CPU.
 */
class OnnxDetector {

    enum class GpuMode { CPU, CUDA, DIRECTML, AUTO }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputSize = 640
    // Actual model input dims (read from the graph; may be non-square). Fall back
    // to the requested square size when the model declares a dynamic input.
    private var inW = 640
    private var inH = 640
    var confThreshold = 0.25f
    var numClasses = 80
    var nmsThreshold = 0.45f

    // ── NanoDet-Plus (GFL/DFL head) mode ──────────────────────────────────────
    // Off by default; enabled via configureNanoDet (Headless: YOLO_DECODE=nanodet).
    // NanoDet needs BGR mean/std preprocessing (not /255 RGB) and a single-output
    // GFL/DFL decode, so it gets its own preprocessing + parser branch.
    private var nanodet = false
    private var regMax = 7
    private var ndStrides = intArrayOf(8, 16, 32, 64)
    private var ndMean = floatArrayOf(103.53f, 116.28f, 123.675f)   // BGR
    private var ndInvStd = floatArrayOf(1f / 57.375f, 1f / 57.12f, 1f / 58.395f)
    private var ndClsSigmoid: Boolean? = null

    /** Switch this detector to NanoDet-Plus decoding (call after load()). */
    fun configureNanoDet(regMax: Int, strides: IntArray, mean: FloatArray, std: FloatArray) {
        nanodet = true; this.regMax = regMax; this.ndStrides = strides
        this.ndMean = mean; this.ndInvStd = FloatArray(3) { 1f / std[it] }
        this.ndClsSigmoid = null
    }

    /** Human-readable description of the execution provider actually used. */
    var activeProvider = "CPU"
        private set

    /** Model's input geometry actually used for letterboxing (after load). */
    val modelInputW get() = inW
    val modelInputH get() = inH

    // Reusable per-frame buffers.
    private var padded: BufferedImage? = null
    private var pixels: IntArray? = null
    private var floats: FloatArray? = null

    fun load(
        modelPath: String,
        inputSize: Int = 640,
        numClasses: Int = 80,
        confThreshold: Float = 0.25f,
        gpuMode: GpuMode = GpuMode.AUTO
    ) {
        close()
        this.inputSize     = inputSize
        this.numClasses    = numClasses
        this.confThreshold = confThreshold

        env = OrtEnvironment.getEnvironment()
        val opts = buildSessionOptions(gpuMode)
        session = env!!.createSession(modelPath, opts)

        // Read the model's real input geometry so letterboxing matches the graph
        // regardless of what YOLO_INPUT was set to. NCHW assumed: [batch,3,H,W].
        inW = inputSize; inH = inputSize
        runCatching {
            val info = session!!.inputInfo.values.iterator().next().info
            if (info is ai.onnxruntime.TensorInfo) {
                val s = info.shape
                if (s.size == 4) {
                    val hh = s[2].toInt(); val ww = s[3].toInt()
                    if (hh in 32..8192) inH = hh
                    if (ww in 32..8192) inW = ww
                }
            }
        }
    }

    fun detect(image: BufferedImage): List<Detection> {
        val sess = session ?: return emptyList()
        val env  = env     ?: return emptyList()

        val iw = inW
        val ih = inH
        val w  = image.width
        val h  = image.height
        // Uniform-scale letterbox (preserves aspect ratio) into the model's own
        // input rectangle, so detections map back to the exact source frame —
        // works for any video resolution / aspect and any model input size.
        val scale = min(iw.toFloat() / w, ih.toFloat() / h)
        val nw = (w * scale).roundToInt()
        val nh = (h * scale).roundToInt()
        val padX = (iw - nw) / 2
        val padY = (ih - nh) / 2

        // Reusable letterbox canvas (gray 114 padding, matches training).
        val pad = padded?.takeIf { it.width == iw && it.height == ih }
            ?: BufferedImage(iw, ih, BufferedImage.TYPE_INT_RGB).also { padded = it }
        pad.createGraphics().apply {
            color = Color(114, 114, 114); fillRect(0, 0, iw, ih)
            drawImage(image, padX, padY, nw, nh, null); dispose()
        }

        val n  = iw * ih
        val px = pixels?.takeIf { it.size == n } ?: IntArray(n).also { pixels = it }
        pad.getRGB(0, 0, iw, ih, px, 0, iw)

        val arr = floats?.takeIf { it.size == 3 * n } ?: FloatArray(3 * n).also { floats = it }
        if (nanodet) {
            // NanoDet: BGR order, (pixel - mean) / std (plane0=B, plane1=G, plane2=R).
            for (i in 0 until n) {
                val p = px[i]
                arr[i]         = ((p and 0xFF)          - ndMean[0]) * ndInvStd[0]
                arr[i + n]     = (((p shr 8) and 0xFF)  - ndMean[1]) * ndInvStd[1]
                arr[i + 2 * n] = (((p shr 16) and 0xFF) - ndMean[2]) * ndInvStd[2]
            }
        } else {
            for (i in 0 until n) {
                val p = px[i]
                arr[i]         = ((p shr 16) and 0xFF) / 255f
                arr[i + n]     = ((p shr 8)  and 0xFF) / 255f
                arr[i + 2 * n] = ( p         and 0xFF) / 255f
            }
        }

        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(arr),
            longArrayOf(1L, 3L, ih.toLong(), iw.toLong()))
        return try {
            val results = sess.run(mapOf(sess.inputNames.iterator().next() to tensor))
            results.use {
                val out   = it[0] as OnnxTensor
                val shape = out.info.shape          // [1, A, B]
                if (shape.size < 3) return emptyList()
                val a = shape[1].toInt()
                val b = shape[2].toInt()
                val buf = out.floatBuffer
                if (nanodet)
                    parseNanoDet(buf, a, b, iw, ih, padX, padY, scale, w, h)
                else if ((b == 6 && a in 7..4000) || (a == 6 && b in 7..4000))
                    parseNmsFree(buf, a, b, iw, ih, padX, padY, scale, w, h)
                else
                    parseAnchorFree(buf, a, b, iw, ih, padX, padY, scale, w, h)
            }
        } catch (e: Exception) {
            emptyList()
        } finally {
            tensor.close()
        }
    }

    /**
     * NanoDet-Plus decode: single output [1, numPoints, nc + 4*(reg_max+1)] (or its
     * transpose), concatenated over the FPN strides (8/16/32/64), row-major (y,x) per
     * level. Per point: argmax class (sigmoid auto-applied if the export left logits),
     * then each of the 4 box sides is a softmax-integral over (reg_max+1) bins → a
     * distance from the cell centre (x*stride, y*stride). Mirrors the python sidecar.
     */
    private fun parseNanoDet(
        buf: FloatBuffer, a: Int, b: Int,
        iw: Int, ih: Int, padX: Int, padY: Int, scale: Float, ow: Int, oh: Int
    ): List<Detection> {
        // numPoints (thousands) ≫ channels (nc+32), so the larger axis is the points.
        val tr = (a < b)
        val numPoints = if (tr) b else a
        val cc = if (tr) a else b
        fun g(pt: Int, ch: Int) = if (tr) buf[ch * numPoints + pt] else buf[pt * cc + ch]
        val rm1 = regMax + 1
        val nc = cc - 4 * rm1
        if (nc < 1) return emptyList()

        // Grid centre priors in nanodet order (strides ascending, y outer, x inner).
        val gx = IntArray(numPoints); val gy = IntArray(numPoints); val gs = IntArray(numPoints)
        var idx = 0
        for (s in ndStrides) {
            val fw = (iw + s - 1) / s; val fh = (ih + s - 1) / s
            var yy = 0
            while (yy < fh) {
                var xx = 0
                while (xx < fw) {
                    if (idx < numPoints) { gx[idx] = xx; gy[idx] = yy; gs[idx] = s; idx++ }
                    xx++
                }
                yy++
            }
        }
        if (idx != numPoints) return emptyList()   // grid ≠ points → wrong input/strides

        if (ndClsSigmoid == null) {
            var raw = false
            val lim = min(numPoints, 200)
            var i = 0
            loop@ while (i < lim) {
                for (c in 0 until nc) { val v = g(i, c); if (v < -0.01f || v > 1.01f) { raw = true; break@loop } }
                i++
            }
            ndClsSigmoid = raw
        }
        val sig = ndClsSigmoid == true

        val out = ArrayList<Detection>()
        val sm = FloatArray(rm1)
        for (i in 0 until numPoints) {
            var best = -1e30f; var cls = -1
            for (c in 0 until nc) { val v = g(i, c); if (v > best) { best = v; cls = c } }
            val score = if (sig) (1.0 / (1.0 + kotlin.math.exp(-best.toDouble()))).toFloat() else best
            if (cls < 0 || score < confThreshold) continue
            val s = gs[i]
            val d = FloatArray(4)
            for (side in 0 until 4) {
                var mx = -1e30f
                for (j in 0 until rm1) { val v = g(i, nc + side * rm1 + j); sm[j] = v; if (v > mx) mx = v }
                var sum = 0f
                for (j in 0 until rm1) { val e = kotlin.math.exp((sm[j] - mx).toDouble()).toFloat(); sm[j] = e; sum += e }
                var acc = 0f
                for (j in 0 until rm1) acc += j * sm[j]
                d[side] = if (sum > 0f) (acc / sum) * s else 0f
            }
            val ctx = (gx[i] * s).toFloat(); val cty = (gy[i] * s).toFloat()
            val x1 = (ctx - d[0] - padX) / scale
            val y1 = (cty - d[1] - padY) / scale
            val x2 = (ctx + d[2] - padX) / scale
            val y2 = (cty + d[3] - padY) / scale
            if (x2 <= x1 || y2 <= y1 || x1.isNaN()) continue
            out += Detection(
                x1 = x1.coerceIn(0f, ow.toFloat()), y1 = y1.coerceIn(0f, oh.toFloat()),
                x2 = x2.coerceIn(0f, ow.toFloat()), y2 = y2.coerceIn(0f, oh.toFloat()),
                conf = score, cls = cls
            )
        }
        return nms(out)
    }

    private fun parseNmsFree(
        buf: FloatBuffer, a: Int, b: Int,
        iw: Int, ih: Int, padX: Int, padY: Int, scale: Float, ow: Int, oh: Int
    ): List<Detection> {
        val tr = (a < b)
        val nd = if (tr) b else a
        fun g(attr: Int, i: Int) = if (tr) buf[attr * nd + i] else buf[i * 6 + attr]

        // Auto-detect pixel vs normalised coords; normalised scale up per-axis to
        // the model input rectangle (x by width, y by height).
        val pixel = looksPixel(nd) { g(2, it) }
        val scX = if (pixel) 1f else iw.toFloat()
        val scY = if (pixel) 1f else ih.toFloat()

        val dets = ArrayList<Detection>()
        for (i in 0 until nd) {
            val score = g(4, i)
            if (score.isNaN() || score < confThreshold) continue
            val x1 = g(0, i) * scX; val y1 = g(1, i) * scY
            val x2 = g(2, i) * scX; val y2 = g(3, i) * scY
            if (x2 <= x1 || y2 <= y1) continue
            val cls = g(5, i).toInt().coerceIn(0, numClasses - 1)
            dets += Detection(
                x1 = ((x1 - padX) / scale).coerceIn(0f, ow.toFloat()),
                y1 = ((y1 - padY) / scale).coerceIn(0f, oh.toFloat()),
                x2 = ((x2 - padX) / scale).coerceIn(0f, ow.toFloat()),
                y2 = ((y2 - padY) / scale).coerceIn(0f, oh.toFloat()),
                conf = score, cls = cls
            )
        }
        return dets
    }

    private fun parseAnchorFree(
        buf: FloatBuffer, a: Int, b: Int,
        iw: Int, ih: Int, padX: Int, padY: Int, scale: Float, ow: Int, oh: Int
    ): List<Detection> {
        val tr    = (a < b)
        val nd    = if (tr) b else a
        val attrs = if (tr) a else b
        // YOLOv5/v6 carry an objectness channel (cx,cy,w,h,obj,cls…) → attrs = 5+nc;
        // YOLOv8/v9/v11 don't (cx,cy,w,h,cls…) → attrs = 4+nc. Tell them apart by the
        // known class count so all of v5/v6/v8/v11 decode correctly.
        val hasObj   = attrs == numClasses + 5
        val clsStart = if (hasObj) 5 else 4
        val nc       = if (hasObj) numClasses else (attrs - 4).coerceAtLeast(1)
        fun g(attr: Int, i: Int) = if (tr) buf[attr * nd + i] else buf[i * attrs + attr]

        // Auto-detect pixel vs normalised box coords (some exports emit 0..1).
        val pixel = looksPixel(nd) { g(2, it) }
        val scX = if (pixel) 1f else iw.toFloat()
        val scY = if (pixel) 1f else ih.toFloat()

        val raw = ArrayList<Detection>()
        for (i in 0 until nd) {
            var bestProb = 0f; var cls = -1
            for (c in 0 until nc) { val s = g(clsStart + c, i); if (s > bestProb) { bestProb = s; cls = c } }
            val score = if (hasObj) g(4, i) * bestProb else bestProb
            if (cls < 0 || score < confThreshold) continue
            val cx = g(0, i) * scX; val cy = g(1, i) * scY
            val bw = g(2, i) * scX; val bh = g(3, i) * scY
            if (bw <= 0f || bh <= 0f || cx.isNaN()) continue
            val x1 = (cx - bw / 2 - padX) / scale
            val y1 = (cy - bh / 2 - padY) / scale
            val x2 = (cx + bw / 2 - padX) / scale
            val y2 = (cy + bh / 2 - padY) / scale
            raw += Detection(
                x1 = x1.coerceIn(0f, ow.toFloat()), y1 = y1.coerceIn(0f, oh.toFloat()),
                x2 = x2.coerceIn(0f, ow.toFloat()), y2 = y2.coerceIn(0f, oh.toFloat()),
                conf = score, cls = cls
            )
        }
        return nms(raw)
    }

    /** Heuristic: coords are raw pixels if any sampled value exceeds ~1.5. */
    private inline fun looksPixel(nd: Int, value: (Int) -> Float): Boolean {
        for (i in 0 until min(nd, 100)) { val v = value(i); if (!v.isNaN() && v > 1.5f) return true }
        return false
    }

    /** Greedy per-class NMS. */
    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.conf }
        val keep = BooleanArray(sorted.size) { true }
        val out = ArrayList<Detection>()
        for (i in sorted.indices) {
            if (!keep[i]) continue
            out += sorted[i]
            for (j in i + 1 until sorted.size)
                if (keep[j] && sorted[i].cls == sorted[j].cls && iou(sorted[i], sorted[j]) > nmsThreshold)
                    keep[j] = false
        }
        return out
    }

    private fun iou(p: Detection, q: Detection): Float {
        val ix1 = maxOf(p.x1, q.x1); val iy1 = maxOf(p.y1, q.y1)
        val ix2 = minOf(p.x2, q.x2); val iy2 = minOf(p.y2, q.y2)
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        val union = (p.x2 - p.x1) * (p.y2 - p.y1) + (q.x2 - q.x1) * (q.y2 - q.y1) - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() {
        session?.close(); session = null
        env?.close(); env = null
        padded = null; pixels = null; floats = null
    }

    private fun buildSessionOptions(mode: GpuMode): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
        when (mode) {
            GpuMode.CPU      -> activeProvider = "CPU"
            GpuMode.CUDA     -> { opts.addCUDA(0);     activeProvider = "CUDA GPU:0" }
            GpuMode.DIRECTML -> { opts.addDirectML(0); activeProvider = "DirectML GPU:0" }
            GpuMode.AUTO     -> activeProvider = tryAddGpuAuto(opts)
        }
        return opts
    }

    private fun tryAddGpuAuto(opts: OrtSession.SessionOptions): String {
        try { opts.addCUDA(0); return "CUDA GPU:0" } catch (_: Throwable) {}
        try { opts.addDirectML(0); return "DirectML GPU:0" } catch (_: Throwable) {}
        return "CPU (no GPU available)"
    }
}
