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
    private var lastFrameMs = System.currentTimeMillis()

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
                if (ok) {
                    startCamera()
                } else {
                    Toast.makeText(this, "Ошибка загрузки модели\nout0=${config.outputName0}", Toast.LENGTH_LONG).show()
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
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(executor) { proxy ->
                // Drop frame if previous still running
                if (!processing.compareAndSet(false, true)) {
                    proxy.close(); return@setAnalyzer
                }
                try {
                    val bmp: Bitmap = proxy.toBitmap()
                    val t0 = System.currentTimeMillis()

                    val dets = try {
                        detector.detect(bmp, config)
                    } catch (e: Exception) {
                        Log.e("CameraActivity", "detect error", e)
                        emptyArray()
                    }

                    val dt = System.currentTimeMillis() - t0
                    lastFps = if (dt > 0) 1000f / dt else lastFps

                    runOnUiThread {
                        binding.overlay.update(dets, config.classNames, lastFps)
                    }
                } catch (e: Exception) {
                    Log.e("CameraActivity", "frame error", e)
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
