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
    private val libraryResult = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        if (r.resultCode == RESULT_OK) {
            r.data?.getStringExtra("config")?.let {
                config = Gson().fromJson(it, ModelConfig::class.java)
                binding.tvParamFile.text = config.paramPath.substringAfterLast('/')
                binding.tvBinFile.text   = config.binPath.substringAfterLast('/')
                saveConfig(); refreshSummary()
                toast("Модель загружена. Нажмите «Определить выходы» для авто-настройки.")
            }
        }
    }
    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) launchCamera() else toast("Нужно разрешение камеры")
    }

    // Shared detector instance — parse-only, no inference here
    private val detector = YoloDetector()
    private var detectorReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        loadConfig()

        binding.btnSelectParam.setOnClickListener { pickParam.launch("*/*") }
        binding.btnSelectBin.setOnClickListener   { pickBin.launch("*/*") }
        binding.btnSettings.setOnClickListener {
            settingsResult.launch(Intent(this, SettingsActivity::class.java)
                .putExtra("config", Gson().toJson(config)))
        }
        binding.btnModelLibrary.setOnClickListener {
            libraryResult.launch(Intent(this, ModelLibraryActivity::class.java))
        }
        binding.btnDetectOutputs.setOnClickListener { analyzeModel() }
        binding.btnStartCamera.setOnClickListener {
            if (config.paramPath.isEmpty() || config.binPath.isEmpty()) {
                toast("Выберите .param и .bin файлы")
            } else { saveConfig(); checkCameraAndLaunch() }
        }
        refreshSummary()
    }

    private fun applyFile(uri: Uri, isParam: Boolean) {
        val name = if (isParam) "model.param" else "model.bin"
        val path = FileUtils.copyToCache(this, uri, name)
        if (path == null) { toast("Ошибка копирования"); return }
        if (isParam) { config.paramPath = path; binding.tvParamFile.text = FileUtils.getFileName(this, uri) }
        else         { config.binPath   = path; binding.tvBinFile.text   = FileUtils.getFileName(this, uri) }
        detectorReady = false
        refreshSummary()
    }

    /** Load model (needed to set g_param_path), then parse .param — no inference */
    private fun analyzeModel() {
        if (config.paramPath.isEmpty() || config.binPath.isEmpty()) {
            toast("Сначала выберите .param и .bin"); return
        }
        binding.btnDetectOutputs.isEnabled = false
        binding.btnDetectOutputs.text = "Анализ..."
        executor.execute {
            // Init loads the model AND sets g_param_path for parsing
            val ok = runCatching { detector.init(config) }.getOrDefault(false)
            val names: Array<String> = if (ok) runCatching { detector.getOutputNames() }.getOrDefault(emptyArray()) else emptyArray()
            runOnUiThread {
                binding.btnDetectOutputs.isEnabled = true
                binding.btnDetectOutputs.text = "🔍 Определить выходы модели"
                if (!ok) { toast("Ошибка загрузки модели"); return@runOnUiThread }
                if (names.isEmpty()) { toast("Не удалось определить выходы (проверьте .param файл)"); return@runOnUiThread }

                // Auto-apply
                config = config.copy(
                    outputName0 = names.getOrElse(0) { "output0" },
                    outputName1 = names.getOrElse(1) { "output1" },
                    outputName2 = names.getOrElse(2) { "output2" }
                )
                detectorReady = true
                saveConfig(); refreshSummary()

                val msg = buildString {
                    append("Найдено выходов: ${names.size}\n\n")
                    names.forEachIndexed { i, n -> append("[$i] $n\n") }
                    append("\nИмена автоматически записаны в настройки.")
                    append("\n\nТеперь выберите версию YOLO в Настройках.")
                }
                AlertDialog.Builder(this)
                    .setTitle("Выходы модели")
                    .setMessage(msg)
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    private fun refreshSummary() {
        binding.tvSummary.text = buildString {
            append("YOLO v${config.yoloVersion}  |  ${config.inputSize}×${config.inputSize}  |  ${config.numClasses} классов\n")
            append("Conf: ${"%,.2f".format(config.confThreshold)}  NMS: ${"%,.2f".format(config.nmsThreshold)}  Thr: ${config.numThreads}  ${if (config.useGPU) "GPU" else "CPU"}\n")
            append("out0: ${config.outputName0}")
        }
        val hasFiles = config.paramPath.isNotEmpty() && config.binPath.isNotEmpty()
        binding.btnStartCamera.isEnabled   = hasFiles
        binding.btnDetectOutputs.isEnabled = hasFiles
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
