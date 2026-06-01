package com.destik.yolodetector

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs

class DetectionOverlayView @JvmOverloads constructor(
    ctx: Context, attrs: AttributeSet? = null
) : View(ctx, attrs) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        textSize = 38f
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
        color = Color.WHITE
    }
    private val bgPaint = Paint().apply { style = Paint.Style.FILL }
    private val fpsTextPaint = Paint().apply {
        textSize = 42f
        color = Color.WHITE
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }
    private val fpsBgPaint = Paint().apply {
        color = 0xBB000000.toInt()
        style = Paint.Style.FILL
    }

    private val colors = intArrayOf(
        0xFFE74C3C.toInt(), 0xFF3498DB.toInt(), 0xFF2ECC71.toInt(),
        0xFFF39C12.toInt(), 0xFF9B59B6.toInt(), 0xFF1ABC9C.toInt(),
        0xFFE67E22.toInt(), 0xFF34495E.toInt(), 0xFFEC407A.toInt(),
        0xFF00BCD4.toInt()
    )

    private var detections: Array<Detection> = emptyArray()
    private var classNames: List<String> = emptyList()
    private var fps: Float = 0f

    fun update(dets: Array<Detection>, names: List<String>, fps: Float) {
        this.detections = dets
        this.classNames = names
        this.fps = fps
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val W = width.toFloat()
        val H = height.toFloat()
        if (W == 0f || H == 0f) return

        for (det in detections) {
            // Guard against out-of-range label indices
            val colorIdx = abs(det.label) % colors.size
            val color = colors[colorIdx]
            boxPaint.color = color
            bgPaint.color = (color and 0x00FFFFFF) or 0xCC000000.toInt()

            val x1 = (det.x * W).coerceIn(0f, W)
            val y1 = (det.y * H).coerceIn(0f, H)
            val x2 = ((det.x + det.w) * W).coerceIn(0f, W)
            val y2 = ((det.y + det.h) * H).coerceIn(0f, H)
            if (x2 <= x1 || y2 <= y1) continue

            canvas.drawRect(x1, y1, x2, y2, boxPaint)

            val label = if (det.label >= 0 && det.label < classNames.size)
                classNames[det.label] else "cls${det.label}"
            val pct = (det.confidence * 100).toInt().coerceIn(0, 100)
            val text = "$label $pct%"

            val th = textPaint.textSize
            val tw = textPaint.measureText(text)
            // Place label above box if there's room, otherwise inside top
            val ty = if (y1 >= th + 6f) y1 - 4f else y1 + th + 2f
            canvas.drawRect(x1, ty - th - 2f, x1 + tw + 8f, ty + 4f, bgPaint)
            canvas.drawText(text, x1 + 4f, ty, textPaint)
        }

        // FPS counter
        val fpsText = "%.1f FPS".format(fps)
        val fw = fpsTextPaint.measureText(fpsText)
        canvas.drawRoundRect(8f, 8f, fw + 20f, 62f, 8f, 8f, fpsBgPaint)
        canvas.drawText(fpsText, 14f, 52f, fpsTextPaint)
    }
}
