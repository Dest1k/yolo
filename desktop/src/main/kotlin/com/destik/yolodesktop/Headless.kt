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
 *   YOLO_DECODE      set to "nanodet" for a NanoDet-Plus ONNX (GFL/DFL head). Then
 *                    ND_REG_MAX (def 7) and ND_STRIDES (def 8,16,32,64) tune the
 *                    decode; YOLO_INPUT must match the export. (default: auto YOLO)
 *   YOLO_SOURCE      "0"/"1"… USB webcam, "rpicam" for a Pi CSI camera, an
 *                    rtsp:// URL (e.g. SIYI ZR10), or an http MJPEG URL  (def: "0")
 *   YOLO_HWDEC       on | off — decode rtsp/http via the system ffmpeg so it can use
 *                    the board's HARDWARE decoder (e.g. Pi 5 HEVC). YOLO_FFMPEG_DECODER
 *                    forces a decoder (e.g. hevc_v4l2m2m); size = YOLO_CAM_W/H. (def: off)
 *   YOLO_RTSP_TRANSPORT  tcp | udp — RTSP transport. Use udp if a TCP feed (SIYI/
 *                    LIVE555) loses packets and stalls after the first frame. (def: tcp)
 *   YOLO_GIMBAL      on | off — SIYI gimbal control (auto-on for the SIYI RTSP
 *                    source). YOLO_GIMBAL_HOST/PORT, YOLO_CONTROL_PORT. The web
 *                    panel always runs (YOLO_CONTROL_PORT, default YOLO_PORT+1):
 *                    live video + manual target capture (drag a box to lock; C/Esc
 *                    clears it; H toggles the gimbal controls, hidden for CSI).
 *   YOLO_RECORD      path to .mp4 — record annotated video with the board's
 *                    HARDWARE H.264 encoder (Pi v4l2m2m/nvenc/vaapi…), libx264 as
 *                    fallback. Needs ffmpeg on PATH. YOLO_RECORD_BITRATE (kbps,
 *                    def 4000), YOLO_RECORD_ENCODER forces a codec.
 *   YOLO_CAM_W/H/FPS capture geometry (all source types)  (default: 1280x720@30)
 *   YOLO_JPEG_Q      MJPEG stream quality 1..100                    (default: 80)
 *   YOLO_TRACK       on | off  — IoU tracking / box persistence     (default: on)
 *   YOLO_INPUT       model input size                         (default: 320)
 *   YOLO_CLASSES     number of classes              (default: labels count, else 80)
 *   YOLO_LABELS      path to a labels.txt (one class name per line) for custom
 *                    models — overrides the built-in COCO names
 *   YOLO_FILTER      keep only these classes (comma-separated indices or names,
 *                    e.g. "person" or "0,2"); applies to drawing, tracking, follow
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
    // Custom class names from a labels file (one per line) — for non-COCO models.
    val labels     = env("YOLO_LABELS")?.let { path ->
        runCatching { java.io.File(path).readLines().map { it.trim() }.filter { it.isNotEmpty() } }
            .getOrElse { System.err.println("WARNING: can't read labels '$path': ${it.message}"); null }
    }
    // Class count drives v5/v6 objectness detection, so derive it from the labels
    // file when present (unless YOLO_CLASSES is set explicitly).
    val numClasses = env("YOLO_CLASSES")?.toIntOrNull() ?: labels?.size ?: 80
    val conf       = env("YOLO_CONF")?.toFloatOrNull() ?: 0.25f
    val port       = env("YOLO_PORT")?.toIntOrNull() ?: 8080
    val gpuAuto    = env("YOLO_GPU")?.lowercase() == "auto"
    // Keep only these classes (comma-separated indices or names, e.g. "person" or
    // "0,2"). Empty/unset = keep all. Filtering happens before tracking/drawing/
    // follow, so it limits everything (e.g. follow only people).
    val filterSet: Set<Int>? = env("YOLO_FILTER")?.let { spec ->
        val names = labels ?: Render.cocoLabels.toList()
        spec.split(",").mapNotNull { tok ->
            val t = tok.trim()
            t.toIntOrNull() ?: names.indexOfFirst { it.equals(t, true) }.takeIf { it >= 0 }
        }.toSet().ifEmpty { null }
    }

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
            if (env("YOLO_DECODE")?.lowercase() == "nanodet") {
                val rm = env("ND_REG_MAX")?.toIntOrNull() ?: 7
                val strides = env("ND_STRIDES")?.split(",")
                    ?.mapNotNull { it.trim().toIntOrNull() }?.toIntArray()
                    ?: intArrayOf(8, 16, 32, 64)
                onnx.configureNanoDet(rm, strides,
                    floatArrayOf(103.53f, 116.28f, 123.675f), floatArrayOf(57.375f, 57.12f, 58.395f))
                println("  decode=nanodet reg_max=$rm strides=${strides.joinToString(",")}")
            }
            onnx.activeProvider
        }
    } catch (e: Exception) {
        System.err.println("ERROR: model load failed: ${e.message}")
        kotlin.system.exitProcess(1)
    }
    println("  provider=$provider")
    if (!isPt) println("  model input: ${onnx.modelInputW}x${onnx.modelInputH}")
    if (labels != null) println("  labels: ${labels.size} custom classes")
    if (filterSet != null) println("  filter: only classes $filterSet")

    val trackOn = env("YOLO_TRACK")?.lowercase() != "off"
    // Box stickiness: how long a box lingers after it stops being detected. Lower =
    // tighter / less ghosting on fast motion (matches the python sidecars' default).
    val trackHoldMs = ((env("YOLO_TRACK_HOLD")?.toFloatOrNull() ?: 0.3f) * 1000).toLong()
    val jpegQ   = env("YOLO_JPEG_Q")?.toIntOrNull()?.coerceIn(1, 100) ?: 80
    val tracker = DetectionTracker(holdMs = trackHoldMs)
    val mjpeg = MjpegServer(port)
    mjpeg.start()
    for (ip in lanAddresses()) println("  stream: http://$ip:$port")
    println("  (open a stream URL above in a browser or VLC on the same network)")

    // ── Manual target capture ────────────────────────────────────────────────
    // Draw a rectangle on the panel video → a firmly attached box that follows the
    // object (a single-object visual tracker, independent of YOLO — "in help of"
    // it). Requests are handed in from the HTTP thread and applied on the stream
    // thread (which owns the frames + tracker), so no locking is needed here.
    val objTracker   = ObjectTracker()
    val manualLockReq = AtomicReference<FloatArray?>(null)   // normalised x1,y1,x2,y2
    val manualClear   = AtomicBoolean(false)
    val manualTarget  = AtomicReference<ObjectTracker.Box?>(null)
    val tracking = AtomicBoolean(false)            // gimbal auto-follow mode (toggled from panel)

    // ── SIYI gimbal control (e.g. ZR10) ──────────────────────────────────────
    // Enabled with YOLO_GIMBAL=on (auto-on when the source is the SIYI RTSP feed).
    // For a CSI camera there's no gimbal, so the gimbal UI is useless: the panel
    // hides it (toggle with 'h') and only shows the video + manual-capture overlay.
    var gimbal: SiyiGimbal? = null
    var follower: GimbalFollower? = null
    val gimbalEnv = env("YOLO_GIMBAL")?.lowercase()
    val gimbalOn = gimbalEnv == "on" ||
        (gimbalEnv != "off" && source.contains("192.168.144.25"))
    if (gimbalOn) {
        val gHost = env("YOLO_GIMBAL_HOST") ?: "192.168.144.25"
        val gPort = env("YOLO_GIMBAL_PORT")?.toIntOrNull() ?: 37260
        val g = SiyiGimbal(gHost, gPort).also { it.start() }
        gimbal = g
        follower = GimbalFollower(
            g,
            maxSpeed = env("YOLO_TRACK_SPEED")?.toIntOrNull() ?: 40,
            invertYaw = env("YOLO_TRACK_INVERT_YAW")?.lowercase() == "on",
            invertPitch = env("YOLO_TRACK_INVERT_PITCH")?.lowercase() == "on"
        )
    }

    // The control/viewer panel always runs (even without a gimbal) so a CSI/USB
    // user still gets the live video + manual-capture overlay in the browser.
    val cPort = env("YOLO_CONTROL_PORT")?.toIntOrNull() ?: (port + 1)
    val control = SiyiControlServer(
        gimbal, cPort, port, tracking, follower,
        onLock = { norm -> manualLockReq.set(norm) },
        onClearLock = { manualClear.set(true) }
    ).also { it.start() }
    val panelKind = if (gimbal != null) "(gimbal control + manual capture)" else "(video + manual capture)"
    for (ip in lanAddresses()) println("  panel: http://$ip:$cPort  $panelKind")

    // ── Optional hardware H.264 recording (YOLO_RECORD=/path/out.mp4) ──────────
    // Encodes the annotated video with the board's hardware H.264 encoder (Pi
    // v4l2m2m / nvenc / vaapi…), software libx264 only as a fallback — so the
    // encode load stays off the CPU that's running inference. Created lazily on
    // the first frame, once the real capture geometry is known.
    val recordPath    = env("YOLO_RECORD")
    val recordBitrate = env("YOLO_RECORD_BITRATE")?.toIntOrNull() ?: 4000
    val recordFps     = env("YOLO_CAM_FPS")?.toIntOrNull() ?: 30
    val recorder      = AtomicReference<H264Recorder?>(null)
    val recordDisabled = AtomicBoolean(recordPath == null)   // true once we've given up / never asked

    // Decoupled pipeline: the capture thread streams every frame at full camera
    // FPS with the last known boxes; inference runs on a separate thread over the
    // *latest* frame only (intermediate frames are dropped). This kills the input
    // lag — the stream never waits on the slow CPU detector — and lets the video
    // run at camera FPS while detection updates as fast as it can.
    val latestDets  = AtomicReference<List<Detection>>(emptyList())
    val latestFrame = AtomicReference<BufferedImage?>(null)
    val targetBox   = AtomicReference<Detection?>(null)   // currently tracked target (for drawing)
    val frameW = AtomicInteger(0); val frameH = AtomicInteger(0)
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
            // Spend CPU while someone is watching, or while we're recording to a
            // file (which must run regardless of viewers); otherwise idle — the
            // server keeps listening so the feed is instant on connect.
            if (mjpeg.hasClients() || recordPath != null) {
                if (!loggedFrame) {
                    loggedFrame = true
                    println("  video frame: ${img.width}x${img.height}")
                }
                // Keep the capture thread light so it drains the decoder and RTSP
                // latency stays low: copy the frame (capture buffers are reused)
                // and hand it off. Drawing + JPEG encode happen on the stream
                // thread; inference on its own thread.
                val snap = copyImage(img)
                latestFrame.set(snap)
                frameW.set(snap.width); frameH.set(snap.height)
                if (inferencing.compareAndSet(false, true)) {
                    inferExec.execute {
                        try {
                            var raw = runCatching { if (isPt) pt.detect(snap) else onnx.detect(snap) }
                                .getOrDefault(emptyList())
                            if (filterSet != null) raw = raw.filter { it.cls in filterSet }
                            latestDets.set(
                                if (trackOn) runCatching { tracker.update(raw, System.currentTimeMillis()) }
                                    .getOrDefault(raw) else raw
                            )
                            detMeter.tick()?.let { detFps.set(it) }
                        } catch (e: Throwable) {           // never let a bug kill the worker
                            System.err.println("inference error: ${e.message}")
                        } finally {
                            inferencing.set(false)
                        }
                    }
                }
            }
        },
        onError = { msg -> System.err.println(msg) }
    )

    // Stream thread: draw boxes + HUD and push at encode speed, always using the
    // newest captured frame (stale frames are dropped) so the stream never lags
    // behind capture even when encoding can't keep up with the camera FPS.
    val streamThread = Thread({
        while (!Thread.currentThread().isInterrupted) {
            val f = latestFrame.getAndSet(null)
            if (f == null) { try { Thread.sleep(2) } catch (e: InterruptedException) { break }; continue }

            // Apply pending manual-capture requests (owned by this thread) and
            // advance the visual tracker so its box stays glued to the object.
            manualLockReq.getAndSet(null)?.let { r ->
                objTracker.lock(f, r[0] * f.width, r[1] * f.height, r[2] * f.width, r[3] * f.height)
            }
            if (manualClear.getAndSet(false)) { objTracker.reset(); manualTarget.set(null) }
            val mbox = if (objTracker.locked) objTracker.update(f) else null
            manualTarget.set(mbox)

            // Lazily bring up the hardware recorder now that we know the geometry.
            if (!recordDisabled.get() && recorder.get() == null) {
                val rec = H264Recorder(recordPath!!, f.width, f.height, recordFps, recordBitrate)
                if (rec.start()) recorder.set(rec) else recordDisabled.set(true)
            }

            val hud = "FPS ${streamFps.get()}  |  det ${detFps.get()}"
            runCatching {
                val annotated = Render.draw(f, latestDets.get(), hud, labels,
                    targetBox.get(), tracking.get(), mbox)
                mjpeg.pushFrame(annotated, jpegQ)
                recorder.get()?.submit(annotated)
            }
            streamMeter.tick()?.let { streamFps.set(it) }
        }
    }, "stream").apply { isDaemon = true; start() }

    // Target-follow thread: when tracking is on, steer the gimbal to keep the
    // YOLO-detected target centred. Toggled from the panel (spacebar) via /track.
    val followThread = follower?.let { fol ->
        Thread({
            while (!Thread.currentThread().isInterrupted) {
                if (tracking.get()) {
                    // A manual lock wins: steer the gimbal onto the hand-picked box
                    // (works for anything, even classes YOLO never detects). With no
                    // manual lock, fall back to following the YOLO detections.
                    val mb = manualTarget.get()
                    val dets = if (mb != null) listOf(Detection(mb.x1, mb.y1, mb.x2, mb.y2, mb.conf, -1))
                               else latestDets.get()
                    targetBox.set(fol.step(dets, frameW.get(), frameH.get()))
                } else { fol.stop(); targetBox.set(null) }
                try { Thread.sleep(66) } catch (e: InterruptedException) { break }
            }
        }, "gimbal-follow").apply { isDaemon = true; start() }
    }

    Runtime.getRuntime().addShutdownHook(Thread {
        println("shutting down…")
        runCatching { video.stop() }
        runCatching { streamThread.interrupt() }
        runCatching { followThread?.interrupt() }
        runCatching { follower?.stop() }
        runCatching { inferExec.shutdownNow() }
        runCatching { recorder.get()?.stop() }
        runCatching { control.stop() }
        runCatching { gimbal?.close() }
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
