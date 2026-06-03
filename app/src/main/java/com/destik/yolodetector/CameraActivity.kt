package com.destik.yolodetector

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.destik.yolodetector.databinding.ActivityCameraBinding
import com.google.gson.Gson
import java.io.File
import java.net.Inet4Address
import java.net.NetworkInterface
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var config: ModelConfig
    private val ncnnDetector = YoloDetector()
    private var onnxDetector: OnnxDetector? = null

    // Stream executor: runs at full camera FPS, handles compositing + MJPEG push
    private val streamExecutor    = Executors.newSingleThreadExecutor()
    // Inference executor: runs YOLO, may skip frames when busy
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val inferencing = AtomicBoolean(false)

    private var lastFps = 0f
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var camera: Camera? = null

    // Last known detections — written by inferenceExecutor, read by streamExecutor
    @Volatile private var lastKnownDets: Array<Detection> = emptyArray()
    @Volatile private var lastKnownDiag: String = ""
    // Latest inference-composed frame for screenshot / recording
    @Volatile private var latestComposed: Bitmap? = null
    // Set by the screenshot button; the next inference frame composes + saves it.
    @Volatile private var pendingShot: Boolean = false

    private val resolutions = listOf("480p" to Size(640, 480), "720p" to Size(1280, 720), "1080p" to Size(1920, 1080))
    private var resolutionIdx = 1

    private var recorder: VideoRecorder? = null
    private var smartMode = false

    private val mjpegServer = MjpegServer(port = 8080)
    private var mjpegInput: MjpegInput? = null
    private val streamMode get() = config.streamUrl.isNotEmpty()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        config = Gson().fromJson(intent.getStringExtra("config") ?: "{}", ModelConfig::class.java)

        inferenceExecutor.execute {
            val hasModel = config.streamUrl.isEmpty() ||
                (config.engine == "onnx" && config.onnxPath.isNotEmpty()) ||
                (config.engine != "onnx" && config.paramPath.isNotEmpty())

            val ok = if (!hasModel) true   // stream without model: still show frames
            else if (config.engine == "onnx") {
                onnxDetector = OnnxDetector(config)
                runCatching { onnxDetector!!.init() }.getOrDefault(false)
            } else {
                runCatching { ncnnDetector.init(config) }.getOrDefault(false)
            }
            runOnUiThread {
                if (ok) {
                    if (streamMode) startStreamInput() else startCamera()
                } else {
                    Toast.makeText(this, "Ошибка загрузки модели", Toast.LENGTH_LONG).show(); finish()
                }
            }
        }

        binding.fabSettings.setOnClickListener { showSettingsSheet() }
        binding.btnFlip.setOnClickListener { if (!streamMode) flipCamera() }
        binding.btnManual.setOnClickListener {
            val cam = camera
            if (cam == null) toast("Камера ещё не готова")
            else CameraControlsSheet(cam).show(supportFragmentManager, "camControls")
        }
        // Hide camera-only controls when using stream input
        if (streamMode) {
            binding.btnFlip.visibility = View.GONE
            binding.btnResolution.visibility = View.GONE
            binding.btnManual.visibility = View.GONE
            binding.previewView.visibility = View.GONE
            binding.streamView.visibility = View.VISIBLE
        }
        binding.btnScreenshot.setOnClickListener { takeScreenshot() }
        binding.btnRecord.setOnClickListener { toggleRecording() }
        binding.btnSmartRecord.setOnClickListener { toggleSmartMode() }
        binding.btnResolution.setOnClickListener { cycleResolution() }
        binding.btnStream.setOnClickListener { toggleStream() }
        updateRecordingUI()
    }

    // ── Stream ────────────────────────────────────────────────────────────────

    private fun toggleStream() {
        if (mjpegServer.running) {
            mjpegServer.stop()
            binding.tvStreamUrl.visibility = View.GONE
            binding.btnStream.backgroundTintList = null
            toast("Стрим остановлен")
        } else {
            try {
                mjpegServer.start()
                val ip = getLocalIpAddress() ?: "?"
                val url = "http://$ip:${mjpegServer.port}/stream"
                binding.tvStreamUrl.text = "📡 $url"
                binding.tvStreamUrl.visibility = View.VISIBLE
                binding.btnStream.backgroundTintList =
                    ContextCompat.getColorStateList(this, android.R.color.holo_green_dark)
                toast("Стрим: $url")
            } catch (e: Exception) {
                toast("Ошибка стрима: ${e.message}")
            }
        }
    }

    private fun getLocalIpAddress(): String? {
        return try {
            NetworkInterface.getNetworkInterfaces()?.toList()
                ?.filter { it.isUp && !it.isLoopback }
                ?.flatMap { it.inetAddresses.toList() }
                ?.firstOrNull { !it.isLoopbackAddress && it is Inet4Address }
                ?.hostAddress
        } catch (e: Exception) { null }
    }

    // ── Stream input ──────────────────────────────────────────────────────────

    private fun startStreamInput() {
        val input = MjpegInput(config.streamUrl).also { mjpegInput = it }
        input.start(
            onFrame = { bmp ->
                // Update ImageView with raw frame on UI thread
                runOnUiThread { binding.streamView.setImageBitmap(bmp) }
                // Submit inference if not already running
                if (inferencing.compareAndSet(false, true)) {
                    val copy = bmp.copy(bmp.config, false)
                    try {
                        inferenceExecutor.execute {
                            runInference(copy, rotation = 0, isFront = false,
                                imgW = copy.width, imgH = copy.height)
                        }
                    } catch (e: RejectedExecutionException) {
                        inferencing.set(false); copy.recycle()
                    }
                }
                // Update overlay aspect ratio
                runOnUiThread {
                    binding.overlay.setImageAspect(bmp.width.toFloat() / bmp.height.toFloat())
                }
            },
            onError = { msg ->
                runOnUiThread { binding.overlay.setDebugLine("⚠ $msg") }
            }
        )
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun cycleResolution() {
        if (recorder?.recording == true) { toast("Остановите запись перед сменой разрешения"); return }
        resolutionIdx = (resolutionIdx + 1) % resolutions.size
        binding.btnResolution.text = resolutions[resolutionIdx].first
        startCamera()
    }

    private fun flipCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
            CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        startCamera()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val targetSize = resolutions[resolutionIdx].second
            // Give the preview the same target resolution as the analysis stream so both
            // share an aspect ratio — otherwise FILL_CENTER crops them differently and the
            // overlay boxes land slightly off the objects shown in the preview.
            val preview = Preview.Builder().setTargetResolution(targetSize).build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(targetSize)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            val isFront = (lensFacing == CameraSelector.LENS_FACING_FRONT)

            // Stream thread: runs at full camera FPS
            analysis.setAnalyzer(streamExecutor) { proxy ->
                val rotation = proxy.imageInfo.rotationDegrees
                val rawW = proxy.width; val rawH = proxy.height

                val bmp: Bitmap = try { proxy.toBitmap() }
                catch (e: Exception) {
                    Log.e("Camera", "toBitmap", e); proxy.close(); return@setAnalyzer
                }
                proxy.close()   // release immediately after copy

                // Push to MJPEG at full camera FPS using last known detections
                // JPEG encoding is skipped entirely if no clients are connected
                if (mjpegServer.running) {
                    val streamFrame = composeFrame(bmp, rotation, isFront)
                    binding.overlay.drawBoxesOnBitmap(streamFrame, lastKnownDets, config.classNames)
                    mjpegServer.pushFrame(streamFrame)
                    streamFrame.recycle()
                }

                // Submit inference if not already running; otherwise drop this frame
                if (inferencing.compareAndSet(false, true)) {
                    val imgW = if (rotation == 90 || rotation == 270) rawH else rawW
                    val imgH = if (rotation == 90 || rotation == 270) rawW else rawH
                    try {
                        inferenceExecutor.execute {
                            runInference(bmp, rotation, isFront, imgW, imgH)
                        }
                    } catch (e: RejectedExecutionException) {   // executor shutting down (leaving camera)
                        inferencing.set(false); bmp.recycle()
                    }
                } else {
                    bmp.recycle()
                }
            }

            runCatching { provider.unbindAll() }
            camera = provider.bindToLifecycle(
                this, CameraSelector.Builder().requireLensFacing(lensFacing).build(),
                preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun runInference(bmp: Bitmap, rotation: Int, isFront: Boolean, imgW: Int, imgH: Int) {
        try {
            val t0 = System.currentTimeMillis()

            val rawDets: Array<Detection> = runCatching {
                if (config.engine == "onnx") onnxDetector?.detect(bmp) ?: emptyArray()
                else ncnnDetector.detect(bmp, config)
            }.getOrDefault(emptyArray())

            val diag = runCatching {
                if (config.engine == "onnx") onnxDetector?.getDiagnostics() ?: ""
                else ncnnDetector.getDiagnostics()
            }.getOrDefault("")

            var dets = rotateDetections(rawDets, rotation)
            if (isFront) dets = dets.map {
                Detection(1f - it.x - it.w, it.y, it.w, it.h, it.label, it.confidence)
            }.toTypedArray()

            lastKnownDets = dets
            lastKnownDiag = diag
            val infMs = (System.currentTimeMillis() - t0).coerceAtLeast(1)
            lastFps = 1000f / infMs

            // Building the full-frame composited bitmap (rotate + draw boxes) is costly,
            // so only do it when something actually consumes it: recording, smart mode,
            // an MJPEG client, or a pending screenshot. When idle, skip it entirely —
            // this is pure per-frame overhead that was dragging the FPS down.
            val needComposite = smartMode || recorder?.recording == true ||
                mjpegServer.running || pendingShot
            if (needComposite) {
                val composed = composeFrame(bmp, rotation, isFront)
                binding.overlay.drawBoxesOnBitmap(composed, dets, config.classNames)
                latestComposed?.recycle()
                latestComposed = composed

                if (smartMode) {
                    ensureRecorderForSmart().feedFrame(composed, dets.isNotEmpty(), VideoRecorder.Mode.SMART)
                } else {
                    recorder?.feedFrame(composed, dets.isNotEmpty(), VideoRecorder.Mode.ALWAYS)
                }

                if (pendingShot) { pendingShot = false; saveSnapshot(composed) }
            }

            runOnUiThread {
                binding.overlay.setImageAspect(imgW.toFloat() / imgH.toFloat())
                binding.overlay.setDebugLine("rot=${rotation}° | ${infMs}ms | $diag")
                binding.overlay.update(dets, config.classNames, lastFps)
                updateRecordingUI()
                if (mjpegServer.running) {
                    val n = mjpegServer.clientCount()
                    val base = binding.tvStreamUrl.text.toString().substringBefore(" (")
                    binding.tvStreamUrl.text = if (n > 0) "$base ($n клиент${clientsSuffix(n)})" else base
                }
            }
        } catch (e: Exception) {
            Log.e("Camera", "inference", e)
        } finally {
            bmp.recycle()
            inferencing.set(false)
        }
    }

    private fun clientsSuffix(n: Int) = when {
        n % 10 == 1 && n % 100 != 11 -> ""
        n % 10 in 2..4 && n % 100 !in 12..14 -> "а"
        else -> "ов"
    }

    private fun composeFrame(raw: Bitmap, rotation: Int, isFront: Boolean): Bitmap {
        val matrix = Matrix().apply {
            postRotate(rotation.toFloat())
            if (isFront) postScale(-1f, 1f, raw.width / 2f, raw.height / 2f)
        }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
    }

    // ── Screenshot ────────────────────────────────────────────────────────────

    private fun takeScreenshot() {
        // The composited frame isn't built every frame anymore (FPS optimisation), so
        // ask the next inference frame to compose + save one.
        pendingShot = true
        toast("Скриншот будет сохранён")
    }

    /** Saves an already-composited (rotated, boxed) frame to the gallery. Runs on inferenceExecutor. */
    private fun saveSnapshot(composed: Bitmap) {
        try {
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "YOLO_$ts.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/YoloDetector")
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { composed.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                runOnUiThread { toast("Скриншот сохранён") }
            } else runOnUiThread { toast("Ошибка сохранения") }
        } catch (e: Exception) {
            Log.e("Camera", "screenshot", e)
            runOnUiThread { toast("Ошибка скриншота") }
        }
    }

    // ── Recording ─────────────────────────────────────────────────────────────

    private fun toggleRecording() {
        if (smartMode) { toast("Выключите умный режим для ручной записи"); return }
        val rec = recorder
        if (rec?.recording == true) {
            inferenceExecutor.execute {
                val file = rec.stop()
                runOnUiThread {
                    updateRecordingUI()
                    if (file != null) saveVideoToGallery(file) else toast("Ошибка записи")
                }
            }
        } else {
            val w = (latestComposed?.width ?: 720).let { if (it % 2 == 0) it else it - 1 }
            val h = (latestComposed?.height ?: 1280).let { if (it % 2 == 0) it else it - 1 }
            recorder = VideoRecorder(w, h, newFile = { newRecordingFile() }).also { it.start() }
            updateRecordingUI()
        }
    }

    /** A fresh timestamped clip in the app's private movies dir (then exported to Downloads). */
    private fun newRecordingFile(): File {
        val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(dir, "YOLO_$ts.mp4")
    }

    private fun toggleSmartMode() {
        if (recorder?.recording == true) { toast("Остановите запись для смены режима"); return }
        smartMode = !smartMode
        if (!smartMode) inferenceExecutor.execute { recorder?.release(); recorder = null }
        updateRecordingUI()
    }

    private fun ensureRecorderForSmart(): VideoRecorder {
        recorder?.let { return it }
        val w = (latestComposed?.width ?: 720).let { if (it % 2 == 0) it else it - 1 }
        val h = (latestComposed?.height ?: 1280).let { if (it % 2 == 0) it else it - 1 }
        // Each detection segment gets its own file (newFile), and when smart mode
        // auto-stops a segment we export the finished clip to Downloads.
        return VideoRecorder(
            w, h,
            newFile = { newRecordingFile() },
            onAutoStop = { file -> if (file != null) saveVideoToGallery(file) }
        ).also { recorder = it }
    }

    private fun updateRecordingUI() {
        val isRecording = recorder?.recording == true
        binding.tvRecIndicator.visibility = if (isRecording) View.VISIBLE else View.GONE
        binding.btnRecord.backgroundTintList = ContextCompat.getColorStateList(
            this, if (isRecording) android.R.color.holo_red_light else R.color.card_bg)
        binding.btnSmartRecord.backgroundTintList = ContextCompat.getColorStateList(
            this, if (smartMode) android.R.color.holo_orange_light else R.color.card_bg)
    }

    private fun saveVideoToGallery(file: File) {
        try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                // API 29+: write into the OS-standard Downloads collection via MediaStore.
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, file.name)
                    put(MediaStore.Downloads.MIME_TYPE, "video/mp4")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/YoloDetector")
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
                    file.delete(); toast("Видео сохранено в Downloads/YoloDetector")
                } else toast("Ошибка сохранения видео")
            } else {
                // Legacy: copy straight into the public Downloads directory.
                val dl = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val dir = File(dl, "YoloDetector").also { it.mkdirs() }
                val dest = File(dir, file.name)
                file.inputStream().use { inp -> dest.outputStream().use { inp.copyTo(it) } }
                file.delete(); toast("Видео сохранено: ${dest.absolutePath}")
            }
        } catch (e: Exception) { toast("Видео: ${file.absolutePath}") }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun rotateDetections(dets: Array<Detection>, degrees: Int): Array<Detection> =
        when (degrees) {
            90  -> dets.map { Detection(1f - it.y - it.h, it.x,              it.h, it.w, it.label, it.confidence) }.toTypedArray()
            180 -> dets.map { Detection(1f - it.x - it.w, 1f - it.y - it.h,  it.w, it.h, it.label, it.confidence) }.toTypedArray()
            270 -> dets.map { Detection(it.y,              1f - it.x - it.w,  it.h, it.w, it.label, it.confidence) }.toTypedArray()
            else -> dets
        }

    private fun showSettingsSheet() {
        SettingsSheet(config) { updated ->
            config = updated
            inferenceExecutor.execute {
                ncnnDetector.release(); onnxDetector?.release()
                if (config.engine == "onnx") {
                    onnxDetector = OnnxDetector(config); runCatching { onnxDetector!!.init() }
                } else runCatching { ncnnDetector.init(config) }
            }
        }.show(supportFragmentManager, "settings")
    }

    private fun toast(msg: String) = runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }

    override fun onDestroy() {
        super.onDestroy()
        mjpegInput?.stop()
        mjpegServer.stop()
        // Stop feeding new frames first so nothing new is submitted for inference.
        streamExecutor.shutdownNow()
        // Tear down native detectors + bitmaps ON the inference thread, so release()
        // (which clears the native ncnn net / recycles bitmaps) can never run while a
        // detect() is still in flight on that same thread. Doing it on the main thread
        // was a use-after-free → native crash when leaving the camera with Back.
        runCatching {
            inferenceExecutor.execute {
                runCatching { recorder?.release() }
                runCatching { ncnnDetector.release() }
                runCatching { onnxDetector?.release() }
                runCatching { latestComposed?.recycle() }
            }
        }
        inferenceExecutor.shutdown()
        runCatching { inferenceExecutor.awaitTermination(2, TimeUnit.SECONDS) }
    }
}
