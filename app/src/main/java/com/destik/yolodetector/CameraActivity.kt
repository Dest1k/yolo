package com.destik.yolodetector

import android.os.Bundle
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

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var config: ModelConfig
    private val detector = YoloDetector()
    private lateinit var executor: ExecutorService
    private var lastTime = System.currentTimeMillis()

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
            val ok = detector.init(config)
            runOnUiThread {
                if (ok) startCamera()
                else {
                    Toast.makeText(this, "Ошибка загрузки модели", Toast.LENGTH_LONG).show()
                    finish()
                }
            }
        }

        binding.fabSettings.setOnClickListener { showSettingsSheet() }
        binding.btnFlip.setOnClickListener { flipCamera() }
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
                val bmp = proxy.toBitmap()
                val t0 = System.currentTimeMillis()
                val dets = detector.detect(bmp, config)
                val dt = System.currentTimeMillis() - t0
                val fps = if (dt > 0) 1000f / dt else 0f
                runOnUiThread {
                    binding.overlay.update(dets, config.classNames, fps)
                }
                proxy.close()
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
            // Reinit detector with new params
            executor.execute {
                detector.release()
                detector.init(config)
            }
        }.show(supportFragmentManager, "settings")
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        detector.release()
    }
}
