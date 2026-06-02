package com.destik.yolodesktop

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.translate.Batchifier
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import java.awt.image.BufferedImage
import java.nio.file.Paths
import kotlin.math.min

/**
 * TorchScript (.pt) detector via DJL PyTorch engine.
 * Supports YOLOv5 anchor-free output [1,N,4+C] and YOLOv10 NMS-free [1,N,6].
 * GPU selection: Device.gpu(0) for CUDA; falls back to CPU automatically.
 */
class PtDetector {

    private var model: ai.djl.Model? = null
    private var predictor: ai.djl.inference.Predictor<BufferedImage, List<Detection>>? = null

    var inputSize = 640
    var confThreshold = 0.25f
    var numClasses = 80

    enum class GpuMode { CPU, CUDA, AUTO }

    fun load(
        modelPath: String,
        inputSize: Int = 640,
        numClasses: Int = 80,
        confThreshold: Float = 0.25f,
        gpuMode: GpuMode = GpuMode.AUTO
    ) {
        close()
        this.inputSize = inputSize
        this.numClasses = numClasses
        this.confThreshold = confThreshold

        val device = resolveDevice(gpuMode)
        model = ai.djl.Model.newInstance("yolo", device)
        model!!.load(Paths.get(modelPath))
        predictor = model!!.newPredictor(YoloTranslator(inputSize, numClasses, confThreshold))
    }

    fun detect(image: BufferedImage): List<Detection> =
        predictor?.predict(image) ?: emptyList()

    fun close() {
        predictor?.close(); predictor = null
        model?.close(); model = null
    }

    private fun resolveDevice(mode: GpuMode): Device {
        if (mode == GpuMode.CPU) return Device.cpu()
        return try {
            val gpuCount = Engine.getInstance().gpuCount
            if (gpuCount > 0) Device.gpu(0) else Device.cpu()
        } catch (_: Exception) {
            Device.cpu()
        }
    }

    private inner class YoloTranslator(
        private val size: Int,
        private val nClasses: Int,
        private val conf: Float
    ) : Translator<BufferedImage, List<Detection>> {

        override fun getBatchifier(): Batchifier = Batchifier.STACK

        override fun processInput(ctx: TranslatorContext, input: BufferedImage): NDList {
            val (lb, scale, padX, padY) = letterbox(input, size)
            ctx.setAttachment("scale", scale)
            ctx.setAttachment("padX", padX)
            ctx.setAttachment("padY", padY)
            ctx.setAttachment("origW", input.width)
            ctx.setAttachment("origH", input.height)

            val pixels = IntArray(size * size)
            lb.getRGB(0, 0, size, size, pixels, 0, size)
            val n = size * size
            val floats = FloatArray(3 * n)
            for (i in 0 until n) {
                val px = pixels[i]
                floats[i]         = (px shr 16 and 0xFF) / 255f
                floats[i + n]     = (px shr 8  and 0xFF) / 255f
                floats[i + n * 2] = (px        and 0xFF) / 255f
            }
            return NDList(ctx.ndManager.create(floats, Shape(1, 3, size.toLong(), size.toLong())))
        }

        override fun processOutput(ctx: TranslatorContext, list: NDList): List<Detection> {
            val scale = ctx.getAttachment("scale") as Float
            val padX  = ctx.getAttachment("padX")  as Float
            val padY  = ctx.getAttachment("padY")  as Float
            val origW = ctx.getAttachment("origW") as Int
            val origH = ctx.getAttachment("origH") as Int

            val out   = list.singletonOrThrow()
            val rows  = out.shape[1].toInt()
            val cols  = out.shape[2].toInt()
            val dets  = mutableListOf<Detection>()

            for (i in 0 until rows) {
                val il = i.toLong()
                if (cols == 6) {
                    // NMS-free v10: [cx, cy, w, h, conf, cls]
                    val c = out.getFloat(0L, il, 4L)
                    if (c < conf) continue
                    val cls = out.getFloat(0L, il, 5L).toInt().coerceIn(0, nClasses - 1)
                    addBox(dets, out.getFloat(0L,il,0L), out.getFloat(0L,il,1L),
                           out.getFloat(0L,il,2L), out.getFloat(0L,il,3L),
                           c, cls, scale, padX, padY, origW, origH)
                } else {
                    // Anchor-free v8: [cx, cy, w, h, cls0..clsN]
                    var maxConf = 0f; var maxCls = 0
                    for (c in 4 until cols) {
                        val v = out.getFloat(0L, il, c.toLong())
                        if (v > maxConf) { maxConf = v; maxCls = c - 4 }
                    }
                    if (maxConf < conf) continue
                    addBox(dets, out.getFloat(0L,il,0L), out.getFloat(0L,il,1L),
                           out.getFloat(0L,il,2L), out.getFloat(0L,il,3L),
                           maxConf, maxCls, scale, padX, padY, origW, origH)
                }
            }
            return dets
        }

        private fun addBox(
            dets: MutableList<Detection>,
            cx: Float, cy: Float, w: Float, h: Float,
            conf: Float, cls: Int,
            scale: Float, padX: Float, padY: Float, origW: Int, origH: Int
        ) {
            val rx = (cx - padX) / scale
            val ry = (cy - padY) / scale
            val rw = w / scale
            val rh = h / scale
            dets.add(Detection(
                x1 = (rx - rw / 2).coerceIn(0f, origW.toFloat()),
                y1 = (ry - rh / 2).coerceIn(0f, origH.toFloat()),
                x2 = (rx + rw / 2).coerceIn(0f, origW.toFloat()),
                y2 = (ry + rh / 2).coerceIn(0f, origH.toFloat()),
                conf = conf, cls = cls
            ))
        }
    }

    private data class LBResult(val img: BufferedImage, val scale: Float, val padX: Float, val padY: Float)

    private fun letterbox(src: BufferedImage, size: Int): LBResult {
        val scale = min(size.toFloat() / src.width, size.toFloat() / src.height)
        val newW  = (src.width  * scale).toInt()
        val newH  = (src.height * scale).toInt()
        val padX  = (size - newW) / 2f
        val padY  = (size - newH) / 2f
        val dst   = BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
        val g     = dst.createGraphics()
        g.color   = java.awt.Color(114, 114, 114)
        g.fillRect(0, 0, size, size)
        g.drawImage(src, padX.toInt(), padY.toInt(), newW, newH, null)
        g.dispose()
        return LBResult(dst, scale, padX, padY)
    }
}
