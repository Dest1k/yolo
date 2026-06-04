package com.destik.yolodesktop

import java.awt.image.BufferedImage
import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

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
 *   YOLO_CAM_W/H/FPS capture geometry (all source types)  (default: 1280x720@30)
 *   YOLO_JPEG_Q      MJPEG stream quality 1..100                    (default: 80)
 *   YOLO_TRACK       on | off  — IoU tracking / box persistence     (default: on)
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
    if (!isPt) println("  model input: ${onnx.modelInputW}x${onnx.modelInputH}")

    val trackOn = env("YOLO_TRACK")?.lowercase() != "off"
    val jpegQ   = env("YOLO_JPEG_Q")?.toIntOrNull()?.coerceIn(1, 100) ?: 80
    val tracker = DetectionTracker()
    val mjpeg = MjpegServer(port)
    mjpeg.start()
    for (ip in lanAddresses()) println("  stream: http://$ip:$port")
    println("  (open a stream URL above in a browser or VLC on the same network)")

    // Decoupled pipeline: the capture thread streams every frame at full camera
    // FPS with the last known boxes; inference runs on a separate thread over the
    // *latest* frame only (intermediate frames are dropped). This kills the input
    // lag — the stream never waits on the slow CPU detector — and lets the video
    // run at camera FPS while detection updates as fast as it can.
    val latestDets  = AtomicReference<List<Detection>>(emptyList())
    val inferencing = AtomicBoolean(false)
    val inferExec   = Executors.newSingleThreadExecutor()
    val streamFps   = AtomicInteger(0)
    val detFps      = AtomicInteger(0)
    val streamMeter = RateMeter()
    val detMeter    = RateMeter()
    var loggedFrame = false

    val video = VideoInput(
        source  = source,
        onFrame = { img ->
            // Only spend CPU while someone is watching; the server keeps listening
            // so the feed is instant on connect.
            if (mjpeg.hasClients()) {
                if (!loggedFrame) {
                    loggedFrame = true
                    println("  video frame: ${img.width}x${img.height}")
                }
                val hud = "FPS ${streamFps.get()}  |  det ${detFps.get()}"
                mjpeg.pushFrame(Render.draw(img, latestDets.get(), hud), jpegQ)
                streamMeter.tick()?.let { streamFps.set(it) }

                // Kick inference on the latest frame if the detector is free.
                if (inferencing.compareAndSet(false, true)) {
                    val snap = copyImage(img)   // detach from the capture buffer
                    inferExec.execute {
                        try {
                            val raw = runCatching { if (isPt) pt.detect(snap) else onnx.detect(snap) }
                                .getOrDefault(emptyList())
                            latestDets.set(if (trackOn) tracker.update(raw, System.currentTimeMillis()) else raw)
                            detMeter.tick()?.let { detFps.set(it) }
                        } finally {
                            inferencing.set(false)
                        }
                    }
                }
            }
        },
        onError = { msg -> System.err.println(msg) }
    )

    Runtime.getRuntime().addShutdownHook(Thread {
        println("shutting down…")
        runCatching { video.stop() }
        runCatching { inferExec.shutdownNow() }
        runCatching { mjpeg.stop() }
        runCatching { onnx.close() }
        runCatching { pt.close() }
    })

    video.start()
    println("running — Ctrl+C to stop")
    CountDownLatch(1).await()   // block forever; the shutdown hook cleans up
}

/** Deep copy so an async inference task is safe from the recycled capture buffer. */
private fun copyImage(src: BufferedImage): BufferedImage {
    val c = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_RGB)
    c.createGraphics().apply { drawImage(src, 0, 0, null); dispose() }
    return c
}

/** Counts ticks and reports an integer rate roughly once per second. */
private class RateMeter {
    private var count = 0
    private var t0 = System.currentTimeMillis()
    /** Returns the new fps value when a ~1s window closes, else null. */
    fun tick(): Int? {
        count++
        val dt = System.currentTimeMillis() - t0
        if (dt >= 1000) { val fps = (count * 1000 / dt).toInt(); count = 0; t0 += dt; return fps }
        return null
    }
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
