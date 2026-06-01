package com.destik.yolodetector

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.destik.yolodetector.databinding.ActivitySettingsBinding
import com.google.gson.Gson

class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding
    private lateinit var config: ModelConfig

    private val pickNames = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { loadClassNames(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Настройки модели"

        config = Gson().fromJson(intent.getStringExtra("config") ?: "{}", ModelConfig::class.java)
        bind()

        binding.btnLoadNames.setOnClickListener { pickNames.launch("*/*") }
        binding.btnSave.setOnClickListener { saveAndFinish() }
    }

    private fun bind() {
        binding.seekConf.progress = (config.confThreshold * 100).toInt()
        binding.tvConfVal.text = "%.2f".format(config.confThreshold)
        binding.seekConf.setOnSeekBarChangeListener(simpleSeek { binding.tvConfVal.text = "%.2f".format(it / 100f) })

        binding.seekNms.progress = (config.nmsThreshold * 100).toInt()
        binding.tvNmsVal.text = "%.2f".format(config.nmsThreshold)
        binding.seekNms.setOnSeekBarChangeListener(simpleSeek { binding.tvNmsVal.text = "%.2f".format(it / 100f) })

        binding.seekThreads.max = 7
        binding.seekThreads.progress = config.numThreads - 1
        binding.tvThreadsVal.text = config.numThreads.toString()
        binding.seekThreads.setOnSeekBarChangeListener(simpleSeek { binding.tvThreadsVal.text = (it + 1).toString() })

        binding.switchGpu.isChecked = config.useGPU
        binding.etInputSize.setText(config.inputSize.toString())
        binding.etNumClasses.setText(config.numClasses.toString())
        binding.etOut0.setText(config.outputName0)
        binding.etOut1.setText(config.outputName1)
        binding.etOut2.setText(config.outputName2)

        // YOLO version radio
        binding.rgVersion.check(
            if (config.yoloVersion >= 8) R.id.rbV8 else R.id.rbV5
        )
        binding.tvClassNames.text = if (config.classNames.isEmpty())
            "Не загружены (будет cls0, cls1...)" else "${config.classNames.size} имён загружено"
    }

    private fun simpleSeek(onChange: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(sb: SeekBar, p: Int, f: Boolean) = onChange(p)
        override fun onStartTrackingTouch(sb: SeekBar) {}
        override fun onStopTrackingTouch(sb: SeekBar) {}
    }

    private fun loadClassNames(uri: Uri) {
        val lines = contentResolver.openInputStream(uri)?.bufferedReader()?.readLines()
            ?.map { it.trim() }?.filter { it.isNotEmpty() } ?: emptyList()
        config = config.copy(classNames = lines)
        binding.tvClassNames.text = "${lines.size} имён загружено"
    }

    private fun saveAndFinish() {
        config = config.copy(
            confThreshold = binding.seekConf.progress / 100f,
            nmsThreshold  = binding.seekNms.progress / 100f,
            numThreads    = binding.seekThreads.progress + 1,
            useGPU        = binding.switchGpu.isChecked,
            inputSize     = binding.etInputSize.text.toString().toIntOrNull() ?: config.inputSize,
            numClasses    = binding.etNumClasses.text.toString().toIntOrNull() ?: config.numClasses,
            outputName0   = binding.etOut0.text.toString().trim(),
            outputName1   = binding.etOut1.text.toString().trim(),
            outputName2   = binding.etOut2.text.toString().trim(),
            yoloVersion   = if (binding.rbV8.isChecked) 8 else 5
        )
        setResult(RESULT_OK, Intent().putExtra("config", Gson().toJson(config)))
        finish()
    }

    override fun onSupportNavigateUp(): Boolean { finish(); return true }
}
