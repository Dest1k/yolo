package com.destik.yolodetector

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.wifi.WifiManager
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
import java.net.NetworkInterface
import java.net.Inet4Address
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var config: ModelConfig
    private val ncnnDetector = YoloDetector()
    private var onnxDetector: OnnxDetector? = null
    private lateinit var executor: ExecutorService
    private val processing = AtomicBoolean(false)
    private var lastFps = 0f
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    private val resolutions = listOf("480p" to Size(640, 480), "720p" to Size(1280, 720), "1080p" to Size(1920, 1080))
    private var resolutionIdx = 1

    private var recorder: VideoRecorder? = null
    private var smartMode = false
    @Volatile private var latestComposed: Bitmap? = null

    private val mjpegServer = MjpegServer(port = 8080)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        config = Gson().fromJson(intent.getStringExtra("config") ?: "{}", ModelConfig::class.java)
        executor = Executors.newSingleThreadExecutor()

        executor.execute {
            val ok = if (config.engine == "onnx") {
                onnxDetector = OnnxDetector(config)
                runCatching { onnxDetector!!.init() }.getOrDefault(false)
            } else {
                runCatching { ncnnDetector.init(config) }.getOrDefault(false)
            }
            runOnUiThread {
                if (ok) startCamera()
                else { Toast.makeText(this, "Ошибка загрузки модели", Toast.LENGTH_LONG).show(); finish() }
            }
        }

        binding.fabSettings.setOnClickListener { showSettingsSheet() }
        binding.btnFlip.setOnClickListener { flipCamera() }
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
                toast("Стрим запущен: $url")
            } catch (e: Exception) {
                toast("Ошибка запуска стрима: ${e.message}")
            }
        }
    }

    private fun getLocalIpAddress(): String? {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces() ?: return null
            for (intf in interfaces) {
                if (!intf.isUp || intf.isLoopback) continue
                for (addr in intf.inetAddresses) {
                    if (!addr.isLoopbackAddress && addr is Inet4Address) return addr.hostAddress
                }
            }
        } catch (e: Exception) {
            Log.w("Camera", "getLocalIpAddress: ${e.message}")
        }
        return null
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
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val targetSize = resolutions[resolutionIdx].second
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(targetSize)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            val isFront = (lensFacing == CameraSelector.LENS_FACING_FRONT)

            analysis.setAnalyzer(executor) { proxy ->
                if (!processing.compareAndSet(false, true)) { proxy.close(); return@setAnalyzer }

                val rotation = proxy.imageInfo.rotationDegrees
                val rawW = proxy.width; val rawH = proxy.height

                val bmp: Bitmap = try { proxy.toBitmap() }
                catch (e: Exception) {
                    Log.e("Camera", "toBitmap", e)
                    proxy.close(); processing.set(false); return@setAnalyzer
                }

                try {
                    val t0 = System.currentTimeMillis()

                    val rawDets: Array<Detection> = try {
                        if (config.engine == "onnx") onnxDetector?.detect(bmp) ?: emptyArray()
                        else ncnnDetector.detect(bmp, config)
                    } catch (e: Exception) { Log.e("Camera", "detect", e); emptyArray() }

                    val diag = try {
                        if (config.engine == "onnx") onnxDetector?.getDiagnostics() ?: ""
                        else ncnnDetector.getDiagnostics()
                    } catch (_: Exception) { "" }

                    var dets = rotateDetections(rawDets, rotation)
                    if (isFront) dets = dets.map {
                        Detection(1f - it.x - it.w, it.y, it.w, it.h, it.label, it.confidence)
                    }.toTypedArray()

                    val imgW = if (rotation == 90 || rotation == 270) rawH else rawW
                    val imgH = if (rotation == 90 || rotation == 270) rawW else rawH
                    lastFps = 1000f / (System.currentTimeMillis() - t0).coerceAtLeast(1)

                    val composed = composeFrame(bmp, rotation, isFront)
                    binding.overlay.drawBoxesOnBitmap(composed, dets, config.classNames)
                    latestComposed?.recycle()
                    latestComposed = composed

                    // Feed MJPEG server (rate-limited internally, zero cost if no clients)
                    if (mjpegServer.running) mjpegServer.pushFrame(composed)

                    // Feed video recorder
                    if (smartMode) {
                        ensureRecorderForSmart().feedFrame(composed, dets.isNotEmpty(), VideoRecorder.Mode.SMART)
                    } else {
                        recorder?.feedFrame(composed, dets.isNotEmpty(), VideoRecorder.Mode.ALWAYS)
                    }

                    runOnUiThread {
                        binding.overlay.setImageAspect(imgW.toFloat() / imgH.toFloat())
                        binding.overlay.setDebugLine("rot=${rotation}° | $diag")
                        binding.overlay.update(dets, config.classNames, lastFps)
                        updateRecordingUI()
                        // Update client count in URL label
                        if (mjpegServer.running) {
                            val n = mjpegServer.clientCount()
                            val base = binding.tvStreamUrl.text.toString().substringBefore(" (")
                            binding.tvStreamUrl.text = if (n > 0) "$base ($n клиент${clientsSuffix(n)})" else base
                        }
                    }
                } catch (e: Exception) {
                    Log.e("Camera", "frame", e)
                } finally {
                    bmp.recycle(); proxy.close(); processing.set(false)
                }
            }

            runCatching { provider.unbindAll() }
            provider.bindToLifecycle(
                this, CameraSelector.Builder().requireLensFacing(lensFacing).build(),
                preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
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
        val snap = latestComposed ?: run { toast("Нет кадра"); return }
        executor.execute {
            try {
                val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, "YOLO_$ts.jpg")
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/YoloDetector")
                }
                val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { snap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                    runOnUiThread { toast("Скриншот сохранён") }
                } else runOnUiThread { toast("Ошибка сохранения") }
            } catch (e: Exception) {
                Log.e("Camera", "screenshot", e)
                runOnUiThread { toast("Ошибка скриншота") }
            }
        }
    }

    // ── Recording ─────────────────────────────────────────────────────────────

    private fun toggleRecording() {
        if (smartMode) { toast("Выключите умный режим для ручной записи"); return }
        val rec = recorder
        if (rec?.recording == true) {
            executor.execute {
                val file = rec.stop()
                runOnUiThread {
                    updateRecordingUI()
                    if (file != null) saveVideoToGallery(file) else toast("Ошибка записи")
                }
            }
        } else {
            val w = (latestComposed?.width ?: 720).let { if (it % 2 == 0) it else it - 1 }
            val h = (latestComposed?.height ?: 1280).let { if (it % 2 == 0) it else it - 1 }
            val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            recorder = VideoRecorder(w, h, File(dir, "YOLO_$ts.mp4")).also { it.start() }
            updateRecordingUI()
        }
    }

    private fun toggleSmartMode() {
        if (recorder?.recording == true) { toast("Остановите запись для смены режима"); return }
        smartMode = !smartMode
        if (!smartMode) executor.execute { recorder?.release(); recorder = null }
        updateRecordingUI()
    }

    private fun ensureRecorderForSmart(): VideoRecorder {
        recorder?.let { return it }
        val w = (latestComposed?.width ?: 720).let { if (it % 2 == 0) it else it - 1 }
        val h = (latestComposed?.height ?: 1280).let { if (it % 2 == 0) it else it - 1 }
        val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return VideoRecorder(w, h, File(dir, "YOLO_$ts.mp4")).also { recorder = it }
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
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/YoloDetector")
            }
            val uri = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
                file.delete(); toast("Видео сохранено в галерею")
            } else toast("Ошибка сохранения видео")
        } catch (e: Exception) {
            Log.e("Camera", "saveVideo", e)
            toast("Видео: ${file.absolutePath}")
        }
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
            executor.execute {
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
        mjpegServer.stop()
        executor.execute { recorder?.release() }
        executor.shutdown()
        ncnnDetector.release(); onnxDetector?.release()
        latestComposed?.recycle()
    }
}
