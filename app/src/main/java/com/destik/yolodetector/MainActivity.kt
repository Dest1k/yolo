package com.destik.yolodetector

import android.Manifest
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

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var config = ModelConfig()

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
        binding.btnStartCamera.setOnClickListener {
            if (config.paramPath.isEmpty() || config.binPath.isEmpty()) {
                toast("Выберите .param и .bin файлы модели")
            } else {
                saveConfig()
                checkCameraAndLaunch()
            }
        }
        refreshSummary()
    }

    private fun applyFile(uri: Uri, isParam: Boolean) {
        val name = if (isParam) "model.param" else "model.bin"
        val path = FileUtils.copyToCache(this, uri, name)
        if (path == null) { toast("Ошибка копирования файла"); return }
        val displayName = FileUtils.getFileName(this, uri)
        if (isParam) {
            config.paramPath = path
            binding.tvParamFile.text = displayName
        } else {
            config.binPath = path
            binding.tvBinFile.text = displayName
        }
        refreshSummary()
    }

    private fun refreshSummary() {
        binding.tvSummary.text = buildString {
            append("YOLO v${config.yoloVersion}  |  ")
            append("${config.inputSize}×${config.inputSize}  |  ")
            append("${config.numClasses} классов\n")
            append("Conf: ${"%,.2f".format(config.confThreshold)}  ")
            append("NMS: ${"%,.2f".format(config.nmsThreshold)}  ")
            append("Threads: ${config.numThreads}  ")
            append(if (config.useGPU) "GPU" else "CPU")
        }
        binding.btnStartCamera.isEnabled = config.paramPath.isNotEmpty() && config.binPath.isNotEmpty()
    }

    private fun checkCameraAndLaunch() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            launchCamera()
        else
            cameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun launchCamera() {
        startActivity(Intent(this, CameraActivity::class.java)
            .putExtra("config", Gson().toJson(config)))
    }

    private fun loadConfig() {
        val json = getSharedPreferences("yolo", MODE_PRIVATE).getString("config", null)
        if (json != null) runCatching { config = Gson().fromJson(json, ModelConfig::class.java) }
        config.paramPath.takeIf { it.isNotEmpty() }?.let {
            binding.tvParamFile.text = it.substringAfterLast('/')
        }
        config.binPath.takeIf { it.isNotEmpty() }?.let {
            binding.tvBinFile.text = it.substringAfterLast('/')
        }
    }

    private fun saveConfig() {
        getSharedPreferences("yolo", MODE_PRIVATE).edit()
            .putString("config", Gson().toJson(config)).apply()
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
