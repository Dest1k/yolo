package com.destik.yolodesktop

import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.concurrent.CountDownLatch

/**
 * Headless entry point for single-board computers (Raspberry Pi, etc.) driven
 * over SSH with no desktop. Loads a model, opens a video source, runs detection
 * and broadcasts the annotated stream over MJPEG on the LAN — no GUI, no display
 * needed. Meant to be launched on boot (e.g. via systemd).
 *
 * Configuration is read from environment variables (all optional except the
 * model), so it slots cleanly into a systemd unit:
 *   YOLO_MODEL       path to model file (.onnx or .pt)        [required]
 *   YOLO_MODEL_TYPE  onnx | pt              (default: inferred from extension)
 *   YOLO_SOURCE      "0"/"1"… USB webcam, "rpicam" for a Pi CSI camera, or an
 *                    http MJPEG URL                           (default: "0")
 *   YOLO_CAM_W/H/FPS rpicam capture geometry           (default: 1280x720@15)
 *   YOLO_INPUT       model input size                         (default: 320)
 *   YOLO_CLASSES     number of classes                        (default: 80)
 *   YOLO_CONF        confidence threshold 0..1                (default: 0.25)
 *   YOLO_PORT        MJPEG server port                        (default: 8080)
 *   YOLO_GPU         cpu | auto                               (default: cpu)
 */
fun main() {
    fun env(k: String) = System.getenv(k)?.trim()?.takeIf { it.isNotEmpty() }

    val modelPath = env("YOLO_MODEL") ?: run {
        System.err.println("ERROR: set YOLO_MODEL to your model file (.onnx or .pt)")
        kotlin.system.exitProcess(2)
    }
    val isPt       = (env("YOLO_MODEL_TYPE")?.lowercase()
        ?: if (modelPath.endsWith(".pt", true)) "pt" else "onnx") == "pt"
    val source     = env("YOLO_SOURCE") ?: "0"
    val inputSize  = env("YOLO_INPUT")?.toIntOrNull() ?: 320
    val numClasses = env("YOLO_CLASSES")?.toIntOrNull() ?: 80
    val conf       = env("YOLO_CONF")?.toFloatOrNull() ?: 0.25f
    val port       = env("YOLO_PORT")?.toIntOrNull() ?: 8080
    val gpuAuto    = env("YOLO_GPU")?.lowercase() == "auto"

    println("YOLO Detector — headless")
    println("  model=$modelPath type=${if (isPt) "pt" else "onnx"} source=$source")
    println("  input=$inputSize classes=$numClasses conf=$conf port=$port gpu=${if (gpuAuto) "auto" else "cpu"}")

    val onnx = OnnxDetector()
    val pt   = PtDetector()
    val provider: String = try {
        if (isPt) {
            pt.load(modelPath, inputSize, numClasses, conf,
                if (gpuAuto) PtDetector.GpuMode.AUTO else PtDetector.GpuMode.CPU)
            if (gpuAuto) "PyTorch auto" else "CPU"
        } else {
            onnx.load(modelPath, inputSize, numClasses, conf,
                if (gpuAuto) OnnxDetector.GpuMode.AUTO else OnnxDetector.GpuMode.CPU)
            onnx.activeProvider
        }
    } catch (e: Exception) {
        System.err.println("ERROR: model load failed: ${e.message}")
        kotlin.system.exitProcess(1)
    }
    println("  provider=$provider")

    val mjpeg = MjpegServer(port)
    mjpeg.start()
    for (ip in lanAddresses()) println("  stream: http://$ip:$port")
    println("  (open a stream URL above in a browser or VLC on the same network)")

    val video = VideoInput(
        source  = source,
        onFrame = { img ->
            // Only spend CPU on inference + drawing while someone is watching;
            // the server still keeps listening so the feed is instant on connect.
            if (mjpeg.hasClients()) {
                val dets = runCatching { if (isPt) pt.detect(img) else onnx.detect(img) }
                    .getOrDefault(emptyList())
                mjpeg.pushFrame(Render.draw(img, dets))
            }
        },
        onError = { msg -> System.err.println(msg) }
    )

    Runtime.getRuntime().addShutdownHook(Thread {
        println("shutting down…")
        runCatching { video.stop() }
        runCatching { mjpeg.stop() }
        runCatching { onnx.close() }
        runCatching { pt.close() }
    })

    video.start()
    println("running — Ctrl+C to stop")
    CountDownLatch(1).await()   // block forever; the shutdown hook cleans up
}

/** Site-local IPv4 addresses, for printing reachable stream URLs. */
private fun lanAddresses(): List<String> = buildList {
    runCatching {
        for (ni in NetworkInterface.getNetworkInterfaces()) {
            if (!ni.isUp || ni.isLoopback) continue
            for (addr in ni.inetAddresses) {
                if (addr is Inet4Address && addr.isSiteLocalAddress) add(addr.hostAddress)
            }
        }
    }
}.ifEmpty { listOf("<pi-ip-address>") }
