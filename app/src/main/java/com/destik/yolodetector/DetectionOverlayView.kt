package com.destik.yolodetector

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class DetectionOverlayView @JvmOverloads constructor(
    ctx: Context, attrs: AttributeSet? = null
) : View(ctx, attrs) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        textSize = 36f
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }
    private val bgPaint = Paint().apply { style = Paint.Style.FILL }

    private val colors = intArrayOf(
        0xFFE74C3C.toInt(), 0xFF3498DB.toInt(), 0xFF2ECC71.toInt(),
        0xFFF39C12.toInt(), 0xFF9B59B6.toInt(), 0xFF1ABC9C.toInt(),
        0xFFE67E22.toInt(), 0xFF34495E.toInt(), 0xFFEC407A.toInt(),
        0xFF00BCD4.toInt()
    )

    private var detections: Array<Detection> = emptyArray()
    private var classNames: List<String> = emptyList()
    private var fps: Float = 0f
    private val fpsRect = RectF()
    private val fpsTextPaint = Paint().apply {
        textSize = 40f
        color = Color.WHITE
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }
    private val fpsBgPaint = Paint().apply {
        color = 0xBB000000.toInt()
        style = Paint.Style.FILL
    }

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

        for (det in detections) {
            val color = colors[det.label % colors.size]
            boxPaint.color = color
            textPaint.color = color
            bgPaint.color = (color and 0x00FFFFFF) or 0x99000000.toInt()

            val x1 = det.x * W
            val y1 = det.y * H
            val x2 = (det.x + det.w) * W
            val y2 = (det.y + det.h) * H

            canvas.drawRect(x1, y1, x2, y2, boxPaint)

            val label = if (det.label < classNames.size) classNames[det.label]
                        else "cls${det.label}"
            val text = "$label ${"%d".format((det.confidence * 100).toInt())}%"
            val tw = textPaint.measureText(text)
            val th = textPaint.textSize
            val ty = if (y1 > th + 4) y1 else y2 + th
            canvas.drawRect(x1, ty - th - 2, x1 + tw + 4, ty + 4, bgPaint)
            canvas.drawText(text, x1 + 2, ty, textPaint)
        }

        // FPS
        val fpsText = "%.1f FPS".format(fps)
        val fw = fpsTextPaint.measureText(fpsText)
        canvas.drawRoundRect(8f, 8f, fw + 20f, 60f, 8f, 8f, fpsBgPaint)
        canvas.drawText(fpsText, 14f, 50f, fpsTextPaint)
    }
}
