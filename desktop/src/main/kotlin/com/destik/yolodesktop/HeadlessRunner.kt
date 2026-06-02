package com.destik.yolodesktop

import org.bytedeco.javacv.FFmpegFrameGrabber
import org.bytedeco.javacv.Java2DFrameConverter
import org.bytedeco.javacv.OpenCVFrameGrabber
import java.awt.image.BufferedImage

/**
 * Headless server mode: MJPEG inference without Compose UI.
 * Usage:
 *   --headless
 *   --model <path>          path to .onnx or .pt model file
 *   --source <cam|url>      webcam index (0) or http://... MJPEG URL
 *   --input-size <n>        model input size (default 640)
 *   --conf <f>              confidence threshold (default 0.25)
 *   --classes <n>           number of classes (default 80)
 *   --gpu <auto|cpu|cuda>   GPU mode (default auto)
 *   --port <n>              MJPEG server port (default 8080)
 *   --skip <n>              run inference every N frames (default 1)
 */
object HeadlessRunner {

    fun run(args: Array<String>) {
        val arg = args.toList()
        fun get(key: String, default: String) = arg.indexOf(key).takeIf { it >= 0 }?.let { arg.getOrNull(it + 1) } ?: default

        val modelPath  = get("--model", "")
        val source     = get("--source", "0")
        val inputSize  = get("--input-size", "640").toInt()
        val conf       = get("--conf", "0.25").toFloat()
        val numClasses = get("--classes", "80").toInt()
        val gpuArg     = get("--gpu", "auto").lowercase()
        val port       = get("--port", "8080").toInt()
        val skip       = get("--skip", "1").toInt().coerceAtLeast(1)

        if (modelPath.isEmpty()) {
            System.err.println("--model <path> is required"); return
        }

        val isOnnx = modelPath.endsWith(".onnx", ignoreCase = true)

        val onnxDetector: OnnxDetector?
        val ptDetector: PtDetector?

        if (isOnnx) {
            val gpuMode = when (gpuArg) {
                "cpu"      -> OnnxDetector.GpuMode.CPU
                "cuda"     -> OnnxDetector.GpuMode.CUDA
                "directml" -> OnnxDetector.GpuMode.DIRECTML
                else       -> OnnxDetector.GpuMode.AUTO
            }
            onnxDetector = OnnxDetector()
            onnxDetector.load(modelPath, inputSize, numClasses, conf, gpuMode)
            ptDetector = null
            println("[headless] ONNX loaded, provider=${onnxDetector.activeProvider}")
        } else {
            val gpuMode = when (gpuArg) {
                "cpu"  -> PtDetector.GpuMode.CPU
                "cuda" -> PtDetector.GpuMode.CUDA
                else   -> PtDetector.GpuMode.AUTO
            }
            ptDetector = PtDetector()
            ptDetector.load(modelPath, inputSize, numClasses, conf, gpuMode)
            onnxDetector = null
            println("[headless] PT loaded")
        }

        val server = MjpegServer(port)
        server.start()
        println("[headless] MJPEG server on :$port")

        val grabber = if (source.startsWith("http", ignoreCase = true))
            FFmpegFrameGrabber(source)
        else
            OpenCVFrameGrabber(source.toIntOrNull() ?: 0)
        grabber.start()
        println("[headless] Grabber started: $source")

        val converter = Java2DFrameConverter()
        var lastDets: List<Detection> = emptyList()
        var frameIdx = 0

        Runtime.getRuntime().addShutdownHook(Thread {
            grabber.runCatching { stop() }
            server.stop()
            onnxDetector?.close()
            ptDetector?.close()
            println("[headless] shutdown")
        })

        while (true) {
            val frame = grabber.grab() ?: break
            val img: BufferedImage = converter.convert(frame) ?: continue
            frameIdx++
            if (frameIdx % skip == 0) {
                lastDets = try {
                    onnxDetector?.detect(img) ?: ptDetector?.detect(img) ?: emptyList()
                } catch (e: Exception) {
                    System.err.println("[headless] inference error: ${e.message}")
                    emptyList()
                }
            }
            val annotated = drawDetections(img, lastDets)
            server.pushFrame(annotated)
        }

        println("[headless] stream ended")
    }

    private fun drawDetections(src: BufferedImage, dets: List<Detection>): BufferedImage {
        if (dets.isEmpty()) return src
        val out = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_RGB)
        val g = out.createGraphics()
        g.drawImage(src, 0, 0, null)
        g.color = java.awt.Color(0, 230, 118)
        g.stroke = java.awt.BasicStroke(2f)
        for (d in dets) {
            g.drawRect(d.x1.toInt(), d.y1.toInt(), (d.x2 - d.x1).toInt(), (d.y2 - d.y1).toInt())
            g.drawString("${d.cls} ${"%.2f".format(d.conf)}", d.x1.toInt() + 2, d.y1.toInt() - 3)
        }
        g.dispose()
        return out
    }
}
