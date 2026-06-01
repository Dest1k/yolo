package com.destik.yolodetector

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.destik.yolodetector.databinding.ActivityCameraBinding
import com.google.gson.Gson
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var config: ModelConfig
    private val detector = YoloDetector()
    private lateinit var executor: ExecutorService
    private val processing = AtomicBoolean(false)
    private var lastFps = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        config = Gson().fromJson(
            intent.getStringExtra("config") ?: "{}",
            ModelConfig::class.java
        )

        executor = Executors.newSingleThreadExecutor()

        executor.execute {
            val ok = runCatching { detector.init(config) }.getOrDefault(false)
            runOnUiThread {
                if (ok) startCamera()
                else {
                    Toast.makeText(this, "Ошибка загрузки модели", Toast.LENGTH_LONG).show()
                    finish()
                }
            }
        }

        binding.fabSettings.setOnClickListener { showSettingsSheet() }
        binding.btnFlip.setOnClickListener    { flipCamera() }
    }

    private var lensFacing = CameraSelector.LENS_FACING_BACK

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

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(executor) { proxy ->
                if (!processing.compareAndSet(false, true)) {
                    proxy.close(); return@setAnalyzer
                }
                try {
                    val bmp: Bitmap = proxy.toBitmap()
                    // Capture rotation and raw dimensions before closing proxy
                    val rotation    = proxy.imageInfo.rotationDegrees
                    val rawW        = proxy.width
                    val rawH        = proxy.height

                    val t0 = System.currentTimeMillis()
                    val rawDets = try {
                        detector.detect(bmp, config)
                    } catch (e: Exception) {
                        Log.e("Camera", "detect error", e); emptyArray()
                    } finally {
                        bmp.recycle()
                    }

                    // Rotate detections to match display orientation.
                    // nativeDetect returns coords normalized to the raw (unrotated) bitmap.
                    val dets = rotateDetections(rawDets, rotation)

                    // After rotation, the logical image dims are swapped for 90/270
                    val imgW = if (rotation == 90 || rotation == 270) rawH else rawW
                    val imgH = if (rotation == 90 || rotation == 270) rawW else rawH

                    val dt = System.currentTimeMillis() - t0
                    lastFps = if (dt > 0) 1000f / dt else lastFps

                    runOnUiThread {
                        binding.overlay.setImageAspect(imgW.toFloat() / imgH.toFloat())
                        binding.overlay.update(dets, config.classNames, lastFps)
                    }
                } catch (e: Exception) {
                    Log.e("Camera", "frame error", e)
                } finally {
                    proxy.close()
                    processing.set(false)
                }
            }

            runCatching { provider.unbindAll() }
            provider.bindToLifecycle(
                this,
                CameraSelector.Builder().requireLensFacing(lensFacing).build(),
                preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Rotate detection boxes (normalized 0..1) by [degrees] CW to match display orientation.
     * CameraX ImageProxy gives raw sensor frames; preview auto-corrects but analysis does not.
     */
    private fun rotateDetections(dets: Array<Detection>, degrees: Int): Array<Detection> {
        return when (degrees) {
            90  -> dets.map { Detection(it.y,                1f - it.x - it.w, it.h, it.w, it.label, it.confidence) }.toTypedArray()
            180 -> dets.map { Detection(1f - it.x - it.w,   1f - it.y - it.h, it.w, it.h, it.label, it.confidence) }.toTypedArray()
            270 -> dets.map { Detection(1f - it.y - it.h,   it.x,             it.h, it.w, it.label, it.confidence) }.toTypedArray()
            else -> dets
        }
    }

    private fun showSettingsSheet() {
        SettingsSheet(config) { updated ->
            config = updated
            executor.execute {
                detector.release()
                runCatching { detector.init(config) }
            }
        }.show(supportFragmentManager, "settings")
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        detector.release()
    }
}
