package com.destik.yolodetector

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import com.destik.yolodetector.databinding.SheetSettingsBinding
import com.google.android.material.bottomsheet.BottomSheetDialogFragment

class SettingsSheet(
    private var config: ModelConfig,
    private val onApply: (ModelConfig) -> Unit
) : BottomSheetDialogFragment() {

    private var _b: SheetSettingsBinding? = null
    private val b get() = _b!!

    override fun onCreateView(inf: LayoutInflater, vg: ViewGroup?, s: Bundle?): View {
        _b = SheetSettingsBinding.inflate(inf, vg, false)
        return b.root
    }

    override fun onViewCreated(v: View, s: Bundle?) {
        super.onViewCreated(v, s)
        bind()
        b.btnApply.setOnClickListener { collect(); onApply(config); dismiss() }
    }

    private fun bind() {
        b.seekConf.progress = (config.confThreshold * 100).toInt()
        b.tvConfVal.text = "%.2f".format(config.confThreshold)
        b.seekConf.setOnSeekBarChangeListener(seek { b.tvConfVal.text = "%.2f".format(it / 100f) })

        b.seekNms.progress = (config.nmsThreshold * 100).toInt()
        b.tvNmsVal.text = "%.2f".format(config.nmsThreshold)
        b.seekNms.setOnSeekBarChangeListener(seek { b.tvNmsVal.text = "%.2f".format(it / 100f) })

        b.switchCpuOnly.isChecked = config.cpuOnly
        b.etInputSize.setText(config.inputSize.toString())
        b.etNumClasses.setText(config.numClasses.toString())
        b.etOut0.setText(config.outputName0)
        b.etOut1.setText(config.outputName1)
        b.etOut2.setText(config.outputName2)

        when {
            config.yoloVersion >= 10 -> b.chipV10.isChecked = true
            config.yoloVersion >= 8  -> b.chipV8.isChecked  = true
            else                     -> b.chipV5.isChecked  = true
        }
    }

    private fun collect() {
        val version = when {
            b.chipV10.isChecked -> 10
            b.chipV8.isChecked  -> 8
            else                -> 5
        }
        config = config.copy(
            confThreshold = b.seekConf.progress / 100f,
            nmsThreshold  = b.seekNms.progress / 100f,
            cpuOnly       = b.switchCpuOnly.isChecked,
            inputSize     = b.etInputSize.text.toString().toIntOrNull() ?: config.inputSize,
            numClasses    = b.etNumClasses.text.toString().toIntOrNull() ?: config.numClasses,
            outputName0   = b.etOut0.text.toString().trim(),
            outputName1   = b.etOut1.text.toString().trim(),
            outputName2   = b.etOut2.text.toString().trim(),
            yoloVersion   = version
        )
    }

    private fun seek(onChange: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(sb: SeekBar, p: Int, f: Boolean) = onChange(p)
        override fun onStartTrackingTouch(sb: SeekBar) {}
        override fun onStopTrackingTouch(sb: SeekBar) {}
    }

    override fun onDestroyView() { super.onDestroyView(); _b = null }
}
