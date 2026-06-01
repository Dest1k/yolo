package com.destik.yolodetector

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.destik.yolodetector.databinding.ActivityMainBinding
import com.google.gson.Gson
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var config = ModelConfig()
    private val detector = YoloDetector()
    private val executor = Executors.newSingleThreadExecutor()

    private val pickParam = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { applyFile(it, isParam = true) }
    }
    private val pickBin = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { applyFile(it, isParam = false) }
    }
    private val settingsResult = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        if (r.resultCode == RESULT_OK) {
            r.data?.getStringExtra("config")?.let { config = Gson().fromJson(it, ModelConfig::class.java) }
            refreshSummary()
        }
    }
    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) launchCamera() else toast("Нужно разрешение камеры")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        loadConfig()

        binding.btnSelectParam.setOnClickListener { pickParam.launch("*/*") }
        binding.btnSelectBin.setOnClickListener { pickBin.launch("*/*") }
        binding.btnSettings.setOnClickListener {
            settingsResult.launch(Intent(this, SettingsActivity::class.java)
                .putExtra("config", Gson().toJson(config)))
        }
        binding.btnDetectOutputs.setOnClickListener { probeModel() }
        binding.btnStartCamera.setOnClickListener {
            if (config.paramPath.isEmpty() || config.binPath.isEmpty()) {
                toast("Выберите .param и .bin файлы")
            } else {
                saveConfig(); checkCameraAndLaunch()
            }
        }
        refreshSummary()
    }

    private fun applyFile(uri: Uri, isParam: Boolean) {
        val name = if (isParam) "model.param" else "model.bin"
        val path = FileUtils.copyToCache(this, uri, name)
        if (path == null) { toast("Ошибка копирования"); return }
        if (isParam) { config.paramPath = path; binding.tvParamFile.text = FileUtils.getFileName(this, uri) }
        else         { config.binPath   = path; binding.tvBinFile.text   = FileUtils.getFileName(this, uri) }
        refreshSummary()
    }

    /** Load model, parse param, probe outputs — show dialog with results */
    private fun probeModel() {
        if (config.paramPath.isEmpty() || config.binPath.isEmpty()) {
            toast("Сначала выберите .param и .bin")
            return
        }
        toast("Анализирую модель...")
        executor.execute {
            val ok = detector.init(config)
            val text = if (!ok) {
                "Ошибка загрузки модели"
            } else {
                val names = detector.getOutputNames()
                val probe = detector.probeOutputs()
                val sb = StringBuilder()
                sb.append("Найдено выходов: ${names.size}\n\n")
                sb.append("Формы тензоров:\n$probe\n")
                if (names.isNotEmpty()) {
                    sb.append("Применить первый выход как output0?")
                    // auto-apply first output name
                    config = config.copy(outputName0 = names[0],
                        outputName1 = if (names.size > 1) names[1] else "output1",
                        outputName2 = if (names.size > 2) names[2] else "output2")
                    saveConfig()
                }
                sb.toString()
            }
            runOnUiThread {
                refreshSummary()
                AlertDialog.Builder(this)
                    .setTitle("Инфо о модели")
                    .setMessage(text)
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    private fun refreshSummary() {
        binding.tvSummary.text = buildString {
            append("YOLO v${config.yoloVersion}  |  ${config.inputSize}×${config.inputSize}  |  ${config.numClasses} классов\n")
            append("Conf: ${"%,.2f".format(config.confThreshold)}  ")
            append("NMS: ${"%,.2f".format(config.nmsThreshold)}  ")
            append("Threads: ${config.numThreads}  ")
            append(if (config.useGPU) "GPU" else "CPU")
            append("\nOutput0: ${config.outputName0}")
        }
        binding.btnStartCamera.isEnabled   = config.paramPath.isNotEmpty() && config.binPath.isNotEmpty()
        binding.btnDetectOutputs.isEnabled = config.paramPath.isNotEmpty() && config.binPath.isNotEmpty()
    }

    private fun checkCameraAndLaunch() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            launchCamera() else cameraPermission.launch(Manifest.permission.CAMERA)
    }
    private fun launchCamera() {
        startActivity(Intent(this, CameraActivity::class.java).putExtra("config", Gson().toJson(config)))
    }
    private fun loadConfig() {
        val json = getSharedPreferences("yolo", MODE_PRIVATE).getString("config", null)
        if (json != null) runCatching { config = Gson().fromJson(json, ModelConfig::class.java) }
        binding.tvParamFile.text = config.paramPath.takeIf { it.isNotEmpty() }?.substringAfterLast('/') ?: "Не выбран"
        binding.tvBinFile.text   = config.binPath.takeIf   { it.isNotEmpty() }?.substringAfterLast('/') ?: "Не выбран"
    }
    private fun saveConfig() {
        getSharedPreferences("yolo", MODE_PRIVATE).edit().putString("config", Gson().toJson(config)).apply()
    }
    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
