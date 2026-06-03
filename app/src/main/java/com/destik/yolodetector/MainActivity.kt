package com.destik.yolodetector

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.view.inputmethod.EditorInfo
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
    private val pickOnnx = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { u ->
            val path = FileUtils.copyToCache(this, u, "model.onnx")
            if (path == null) { toast("Ошибка копирования"); return@let }
            config.onnxPath = path
            binding.tvOnnxFile.text = FileUtils.getFileName(this, u)
            saveConfig(); refreshSummary()
        }
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
        if (ok) launchActivity() else toast("Нужно разрешение камеры")
    }

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
        binding.btnSelectOnnx.setOnClickListener { pickOnnx.launch("*/*") }
        binding.btnDetectOutputs.setOnClickListener { analyzeModel() }
        binding.btnToggleAdvanced.setOnClickListener {
            val show = binding.advancedSection.visibility != android.view.View.VISIBLE
            binding.advancedSection.visibility =
                if (show) android.view.View.VISIBLE else android.view.View.GONE
            binding.btnToggleAdvanced.text =
                if (show) "⚙ Скрыть расширенное" else "⚙ Расширенное (своя модель)"
        }

        // Stream URL field
        binding.etStreamUrl.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) { applyStreamUrl(); true } else false
        }
        binding.etStreamUrl.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) applyStreamUrl()
        }
        binding.btnClearStream.setOnClickListener {
            binding.etStreamUrl.setText("")
            config.streamUrl = ""
            saveConfig(); refreshSummary()
        }

        binding.btnStartCamera.setOnClickListener {
            val streamUrl = binding.etStreamUrl.text.toString().trim()
            config.streamUrl = streamUrl
            saveConfig()

            if (streamUrl.isNotEmpty()) {
                if (streamUrl.startsWith("rtsp://", ignoreCase = true)) {
                    toast("RTSP пока не поддерживается. Используйте HTTP MJPEG.")
                    return@setOnClickListener
                }
                if (!streamUrl.startsWith("http", ignoreCase = true)) {
                    toast("Неверный URL. Формат: http://IP:PORT/stream")
                    return@setOnClickListener
                }
                val hasModel = hasRequiredModelFiles()
                if (!hasModel) {
                    AlertDialog.Builder(this)
                        .setTitle("Запуск без модели")
                        .setMessage("Файлы модели не выбраны. Стрим будет отображаться без детекции объектов.\n\nПродолжить?")
                        .setPositiveButton("Да") { _, _ -> checkCameraPermAndLaunch() }
                        .setNegativeButton("Отмена", null)
                        .show()
                } else {
                    checkCameraPermAndLaunch()
                }
            } else {
                if (!hasRequiredModelFiles()) {
                    toast(if (config.engine == "onnx") "Выберите .onnx файл" else "Выберите .param и .bin файлы")
                } else {
                    checkCameraPermAndLaunch()
                }
            }
        }
        refreshSummary()
    }

    private fun applyStreamUrl() {
        val url = binding.etStreamUrl.text.toString().trim()
        if (config.streamUrl != url) {
            config.streamUrl = url
            saveConfig(); refreshSummary()
        }
    }

    private fun hasRequiredModelFiles(): Boolean {
        return if (config.engine == "onnx") config.onnxPath.isNotEmpty()
               else config.paramPath.isNotEmpty() && config.binPath.isNotEmpty()
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

    private fun analyzeModel() {
        if (config.paramPath.isEmpty() || config.binPath.isEmpty()) {
            toast("Сначала выберите .param и .bin"); return
        }
        binding.btnDetectOutputs.isEnabled = false
        binding.btnDetectOutputs.text = "Анализ..."
        executor.execute {
            // Parse the .param text file directly — no model/GPU load, so this can't
            // crash even for models whose Vulkan path is unstable on this device.
            val names: Array<String> = runCatching { detector.getOutputNames(config) }.getOrDefault(emptyArray())
            runOnUiThread {
                binding.btnDetectOutputs.isEnabled = true
                binding.btnDetectOutputs.text = "🔍 Определить выходы модели"
                if (names.isEmpty()) { toast("Не удалось определить выходы (проверьте .param файл)"); return@runOnUiThread }
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
                AlertDialog.Builder(this).setTitle("Выходы модели").setMessage(msg)
                    .setPositiveButton("OK", null).show()
            }
        }
    }

    private fun refreshSummary() {
        val streamUrl = config.streamUrl
        binding.tvSummary.text = buildString {
            append("YOLO v${config.yoloVersion}  |  ${config.inputSize}×${config.inputSize}  |  ${config.numClasses} классов\n")
            append("Conf: ${"%,.2f".format(config.confThreshold)}  NMS: ${"%,.2f".format(config.nmsThreshold)}  Thr: ${config.numThreads}  ${if (config.useGPU) "GPU" else "CPU"}\n")
            append("out0: ${config.outputName0}")
            if (streamUrl.isNotEmpty()) append("\n📡 Поток: $streamUrl")
        }
        val hasNcnn = config.paramPath.isNotEmpty() && config.binPath.isNotEmpty()
        val hasOnnx = config.onnxPath.isNotEmpty()
        val hasFiles = if (config.engine == "onnx") hasOnnx else hasNcnn
        val hasStream = streamUrl.isNotEmpty()
        binding.btnStartCamera.isEnabled   = hasFiles || hasStream
        binding.btnStartCamera.text        = if (hasStream) "Запустить стрим" else "Запустить камеру"
        binding.btnDetectOutputs.isEnabled = hasNcnn
    }

    private fun checkCameraPermAndLaunch() {
        // Camera permission needed even for stream mode (CameraX might still init)
        if (config.streamUrl.isNotEmpty()) {
            // Stream mode: no camera permission needed
            launchActivity()
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
                launchActivity() else cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchActivity() {
        startActivity(Intent(this, CameraActivity::class.java).putExtra("config", Gson().toJson(config)))
    }

    private fun loadConfig() {
        val json = getSharedPreferences("yolo", MODE_PRIVATE).getString("config", null)
        if (json != null) runCatching { config = Gson().fromJson(json, ModelConfig::class.java) }
        binding.tvParamFile.text = config.paramPath.takeIf { it.isNotEmpty() }?.substringAfterLast('/') ?: "Не выбран"
        binding.tvBinFile.text   = config.binPath.takeIf   { it.isNotEmpty() }?.substringAfterLast('/') ?: "Не выбран"
        binding.tvOnnxFile.text  = config.onnxPath.takeIf  { it.isNotEmpty() }?.substringAfterLast('/') ?: "Не выбран"
        if (config.streamUrl.isNotEmpty()) binding.etStreamUrl.setText(config.streamUrl)
    }

    private fun saveConfig() {
        getSharedPreferences("yolo", MODE_PRIVATE).edit().putString("config", Gson().toJson(config)).apply()
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    override fun onDestroy() { super.onDestroy(); executor.shutdown() }
}
