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
    private var lensFacing = CameraSelector.LENS_FACING_BACK

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

            val isFront = (lensFacing == CameraSelector.LENS_FACING_FRONT)

            analysis.setAnalyzer(executor) { proxy ->
                if (!processing.compareAndSet(false, true)) {
                    proxy.close(); return@setAnalyzer
                }

                val rotation = proxy.imageInfo.rotationDegrees
                val rawW     = proxy.width
                val rawH     = proxy.height

                val bmp: Bitmap = try {
                    proxy.toBitmap()
                } catch (e: Exception) {
                    Log.e("Camera", "toBitmap failed", e)
                    proxy.close(); processing.set(false)
                    return@setAnalyzer
                }

                try {
                    val t0 = System.currentTimeMillis()

                    val rawDets: Array<Detection> = try {
                        detector.detect(bmp, config)
                    } catch (e: Exception) {
                        Log.e("Camera", "detect error", e); emptyArray()
                    }

                    // Get diagnostics from native layer (tensor shape, max confidence, etc.)
                    val diag = try { detector.getDiagnostics() } catch (_: Exception) { "" }

                    var dets = rotateDetections(rawDets, rotation)
                    if (isFront) {
                        dets = dets.map {
                            Detection(1f - it.x - it.w, it.y, it.w, it.h, it.label, it.confidence)
                        }.toTypedArray()
                    }

                    val imgW = if (rotation == 90 || rotation == 270) rawH else rawW
                    val imgH = if (rotation == 90 || rotation == 270) rawW else rawH

                    val dt = System.currentTimeMillis() - t0
                    lastFps = if (dt > 0) 1000f / dt else lastFps

                    // Debug line: "rot=90 | v10|300x6|px:1|maxC:0.87|dets:3"
                    val debugLine = "rot=${rotation}° | $diag"

                    runOnUiThread {
                        binding.overlay.setImageAspect(imgW.toFloat() / imgH.toFloat())
                        binding.overlay.setDebugLine(debugLine)
                        binding.overlay.update(dets, config.classNames, lastFps)
                    }
                } catch (e: Exception) {
                    Log.e("Camera", "frame error", e)
                } finally {
                    bmp.recycle()
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
     * Rotate detection boxes (normalised 0..1) by [degrees] CW to match display orientation.
     * 90° CW: point (x,y) → (y, 1-x);  box (x,y,w,h) → (y, 1-x-w, h, w)
     */
    private fun rotateDetections(dets: Array<Detection>, degrees: Int): Array<Detection> =
        when (degrees) {
            // 90° CW:  (px,py)→(1-py,px)  ∴ box(x,y,w,h)→(1-y-h, x,   h, w)
            90  -> dets.map { Detection(1f - it.y - it.h, it.x,             it.h, it.w, it.label, it.confidence) }.toTypedArray()
            180 -> dets.map { Detection(1f - it.x - it.w, 1f - it.y - it.h, it.w, it.h, it.label, it.confidence) }.toTypedArray()
            // 270° CW: (px,py)→(py,1-px)  ∴ box(x,y,w,h)→(y,   1-x-w, h, w)
            270 -> dets.map { Detection(it.y,             1f - it.x - it.w, it.h, it.w, it.label, it.confidence) }.toTypedArray()
            else -> dets
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
