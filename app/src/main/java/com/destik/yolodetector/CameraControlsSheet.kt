package com.destik.yolodetector

import android.annotation.SuppressLint
import android.graphics.Color
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import kotlin.math.ln
import kotlin.math.exp
import kotlin.math.roundToInt
import kotlin.math.roundToLong

/**
 * Full manual camera panel, built dynamically from the *actual* capabilities of
 * the bound camera: zoom and exposure-compensation are always offered (CameraX
 * core), while ISO, shutter speed and manual focus appear only when the sensor
 * reports MANUAL_SENSOR / a real (non-fixed) lens. Everything routes through
 * Camera2 interop and is wrapped so an unsupported key degrades instead of
 * crashing on the long tail of single-board-computer / cheap-phone cameras.
 */
@OptIn(ExperimentalCamera2Interop::class)
class CameraControlsSheet(private val camera: Camera) : BottomSheetDialogFragment() {

    private val c2Control by lazy { Camera2CameraControl.from(camera.cameraControl) }
    private val c2Info    by lazy { Camera2CameraInfo.from(camera.cameraInfo) }

    // Manual-override state — null = "let the camera decide" for that axis.
    private var manualAe = false
    private var manualAf = false
    private var iso: Int? = null
    private var exposureNs: Long? = null
    private var focusDist: Float? = null

    private val accent = 0xFF00E676.toInt()
    private val sub    = 0xFFAAAAAA.toInt()

    @SuppressLint("SetTextI18n")
    override fun onCreateView(inf: android.view.LayoutInflater, vg: ViewGroup?, s: Bundle?): View {
        val dp = resources.displayMetrics.density
        fun px(v: Int) = (v * dp).roundToInt()

        val root = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF1E1E1E.toInt())
            setPadding(px(20), px(16), px(20), px(24))
        }

        root.addView(TextView(requireContext()).apply {
            text = "Управление камерой"; setTextColor(accent); textSize = 18f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, 0, 0, px(8))
        })

        // ── Zoom (always) ──────────────────────────────────────────────────────
        runCatching {
            val zs = camera.cameraInfo.zoomState.value
            val minZ = zs?.minZoomRatio ?: 1f
            val maxZ = zs?.maxZoomRatio ?: 1f
            if (maxZ > minZ + 0.01f) {
                addFloatSlider(root, "Зум", minZ, maxZ, zs?.zoomRatio ?: minZ, "×%.1f") { r ->
                    runCatching { camera.cameraControl.setZoomRatio(r) }
                }
            }
        }

        // ── Exposure compensation / EV (always if supported) ───────────────────
        runCatching {
            val es = camera.cameraInfo.exposureState
            if (es.isExposureCompensationSupported) {
                val range = es.exposureCompensationRange
                val step  = es.exposureCompensationStep.toFloat()
                addIntSlider(root, "Яркость (EV)", range.lower, range.upper, es.exposureCompensationIndex) { idx ->
                    runCatching { camera.cameraControl.setExposureCompensationIndex(idx) }
                    "%+.1f EV".format(idx * step)
                }
            }
        }

        // ── Manual exposure: ISO + shutter (only with MANUAL_SENSOR) ────────────
        val caps = runCatching {
            c2Info.getCameraCharacteristic(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        }.getOrNull()
        val hasManualSensor = caps?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
        val isoRange = runCatching {
            c2Info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        }.getOrNull()
        val expRange = runCatching {
            c2Info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        }.getOrNull()

        if (hasManualSensor && isoRange != null && expRange != null) {
            addToggle(root, "Ручная экспозиция (ISO + выдержка)") { on ->
                manualAe = on
                if (!on) { iso = null; exposureNs = null }
                else {
                    if (iso == null) iso = isoRange.lower
                    if (exposureNs == null) exposureNs = expRange.lower.coerceAtLeast(1L)
                }
                apply()
            }
            addIntSlider(root, "ISO", isoRange.lower, isoRange.upper, iso ?: isoRange.lower) { v ->
                if (manualAe) { iso = v; apply() }
                "ISO $v"
            }
            addLogSlider(root, "Выдержка", expRange.lower.coerceAtLeast(1L), expRange.upper,
                exposureNs ?: expRange.lower.coerceAtLeast(1L)) { ns ->
                if (manualAe) { exposureNs = ns; apply() }
                shutterLabel(ns)
            }
        }

        // ── Manual focus (only on lenses with a real focus range) ──────────────
        val minFocus = runCatching {
            c2Info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
        }.getOrNull() ?: 0f
        if (minFocus > 0f) {
            addToggle(root, "Ручной фокус") { on ->
                manualAf = on
                if (!on) focusDist = null else if (focusDist == null) focusDist = 0f
                apply()
            }
            // 0 diopters = infinity, minFocus = closest. Slider 0..100 → far..near.
            addFloatSlider(root, "Фокус (∞ → близко)", 0f, minFocus, focusDist ?: 0f, "%.1f дптр") { d ->
                if (manualAf) { focusDist = d; apply() }
            }
        }

        // ── Reset to full auto ─────────────────────────────────────────────────
        root.addView(Button(requireContext()).apply {
            text = "Сбросить (авто)"
            setBackgroundColor(0xFF333333.toInt()); setTextColor(Color.WHITE)
            val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            lp.topMargin = px(12); layoutParams = lp
            setOnClickListener {
                manualAe = false; manualAf = false; iso = null; exposureNs = null; focusDist = null
                runCatching { c2Control.setCaptureRequestOptions(CaptureRequestOptions.Builder().build()) }
                runCatching { camera.cameraControl.setExposureCompensationIndex(0) }
                dismiss()
            }
        })

        if (!hasManualSensor && minFocus <= 0f) {
            root.addView(TextView(requireContext()).apply {
                text = "Эта камера не поддерживает ручные ISO/выдержку/фокус — доступны зум и яркость."
                setTextColor(sub); textSize = 12f; setPadding(0, px(12), 0, 0)
            })
        }

        return ScrollView(requireContext()).apply { addView(root) }
    }

    /** Rebuild the full Camera2 request from current manual state and push it. */
    private fun apply() {
        runCatching {
            val b = CaptureRequestOptions.Builder()
            if (manualAf) {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                focusDist?.let { b.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, it) }
            } else {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
            }
            if (manualAe) {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                iso?.let { b.setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, it) }
                exposureNs?.let { b.setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, it) }
            } else {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            }
            c2Control.setCaptureRequestOptions(b.build())
        }
    }

    // ── UI builders ────────────────────────────────────────────────────────────

    private fun shutterLabel(ns: Long): String {
        val sec = ns / 1_000_000_000.0
        return if (sec >= 1.0) "%.1f s".format(sec)
        else "1/${(1.0 / sec).roundToInt()} s"
    }

    private fun label(text: String): TextView = TextView(requireContext()).apply {
        setTextColor(sub); textSize = 13f; this.text = text
        setPadding(0, (8 * resources.displayMetrics.density).roundToInt(), 0, 0)
    }

    private fun seekBar(): SeekBar = SeekBar(requireContext()).apply {
        max = 100; progressTintList = android.content.res.ColorStateList.valueOf(accent)
        thumbTintList = android.content.res.ColorStateList.valueOf(accent)
    }

    private fun simpleSeek(onProgress: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) { if (fromUser) onProgress(p) }
        override fun onStartTrackingTouch(sb: SeekBar) {}
        override fun onStopTrackingTouch(sb: SeekBar) {}
    }

    private fun addFloatSlider(
        parent: LinearLayout, name: String, lo: Float, hi: Float, cur: Float,
        fmt: String, onSet: (Float) -> Unit
    ) {
        val lbl = label("$name: " + fmt.format(cur)); parent.addView(lbl)
        val sb = seekBar(); parent.addView(sb)
        sb.progress = (((cur - lo) / (hi - lo)) * 100f).roundToInt().coerceIn(0, 100)
        sb.setOnSeekBarChangeListener(simpleSeek { p ->
            val v = lo + (hi - lo) * p / 100f
            lbl.text = "$name: " + fmt.format(v); onSet(v)
        })
    }

    private fun addIntSlider(
        parent: LinearLayout, name: String, lo: Int, hi: Int, cur: Int,
        onSet: (Int) -> String
    ) {
        val lbl = label("$name"); parent.addView(lbl)
        val sb = seekBar(); parent.addView(sb)
        val span = (hi - lo).coerceAtLeast(1)
        sb.progress = (((cur - lo).toFloat() / span) * 100f).roundToInt().coerceIn(0, 100)
        lbl.text = onSet(cur)
        sb.setOnSeekBarChangeListener(simpleSeek { p ->
            val v = (lo + span * p / 100f).roundToInt().coerceIn(lo, hi)
            lbl.text = onSet(v)
        })
    }

    /** Log-scaled slider for wide-range values (shutter time spans 1/10000s..several s). */
    private fun addLogSlider(
        parent: LinearLayout, name: String, lo: Long, hi: Long, cur: Long,
        onSet: (Long) -> String
    ) {
        val lbl = label("$name"); parent.addView(lbl)
        val sb = seekBar(); parent.addView(sb)
        val lnLo = ln(lo.toDouble()); val lnHi = ln(hi.toDouble())
        val span = (lnHi - lnLo).coerceAtLeast(1e-6)
        sb.progress = (((ln(cur.toDouble()) - lnLo) / span) * 100.0).roundToInt().coerceIn(0, 100)
        lbl.text = "$name: " + onSet(cur)
        sb.setOnSeekBarChangeListener(simpleSeek { p ->
            val v = exp(lnLo + span * p / 100.0).roundToLong().coerceIn(lo, hi)
            lbl.text = "$name: " + onSet(v)
        })
    }

    private fun addToggle(parent: LinearLayout, name: String, onChange: (Boolean) -> Unit) {
        val sw = com.google.android.material.switchmaterial.SwitchMaterial(requireContext()).apply {
            text = name; setTextColor(Color.WHITE); textSize = 14f
            val dp = resources.displayMetrics.density
            setPadding(0, (10 * dp).roundToInt(), 0, 0)
            setOnCheckedChangeListener { _, checked -> onChange(checked) }
        }
        parent.addView(sw)
    }
}
