package com.destik.yolodetector

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs

class DetectionOverlayView @JvmOverloads constructor(
    ctx: Context, attrs: AttributeSet? = null
) : View(ctx, attrs) {

    private val boxPaint   = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true }
    private val textPaint  = Paint().apply { textSize = 38f; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD; color = Color.WHITE }
    private val bgPaint    = Paint().apply { style = Paint.Style.FILL }
    private val fpsBgPaint = Paint().apply { color = 0xBB000000.toInt(); style = Paint.Style.FILL }
    private val fpsPaint   = Paint().apply { textSize = 38f; color = Color.WHITE; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD }
    private val diagPaint  = Paint().apply { textSize = 28f; color = Color.YELLOW; isAntiAlias = true; typeface = Typeface.MONOSPACE }

    private val colors = intArrayOf(
        0xFFE74C3C.toInt(), 0xFF3498DB.toInt(), 0xFF2ECC71.toInt(),
        0xFFF39C12.toInt(), 0xFF9B59B6.toInt(), 0xFF1ABC9C.toInt(),
        0xFFE67E22.toInt(), 0xFF607D8B.toInt(), 0xFFEC407A.toInt(),
        0xFF00BCD4.toInt()
    )

    private var detections: Array<Detection> = emptyArray()
    private var classNames: List<String> = emptyList()
    private var fps: Float = 0f
    private var imgAspect: Float = 9f / 16f   // portrait default
    private var debugLine: String = ""

    fun setImageAspect(aspect: Float) { imgAspect = aspect }
    fun setDebugLine(line: String)    { debugLine = line }

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

        // ── FILL_CENTER coordinate transform ────────────────────────────────
        val viewAspect = W / H
        val cropX: Float; val cropY: Float
        val scaleX: Float; val scaleY: Float
        if (imgAspect > viewAspect) {
            val k = imgAspect / viewAspect
            cropX = (1f - 1f/k) / 2f; cropY = 0f
            scaleX = 1f/k; scaleY = 1f
        } else {
            val k = viewAspect / imgAspect
            cropX = 0f; cropY = (1f - 1f/k) / 2f
            scaleX = 1f; scaleY = 1f/k
        }

        // ── Detection boxes ─────────────────────────────────────────────────
        for (det in detections) {
            val vx1 = ((det.x - cropX) / scaleX * W).coerceIn(0f, W)
            val vy1 = ((det.y - cropY) / scaleY * H).coerceIn(0f, H)
            val vx2 = ((det.x + det.w - cropX) / scaleX * W).coerceIn(0f, W)
            val vy2 = ((det.y + det.h - cropY) / scaleY * H).coerceIn(0f, H)
            if (vx2 <= vx1 || vy2 <= vy1) continue

            val color = colors[abs(det.label) % colors.size]
            boxPaint.color = color
            bgPaint.color  = (color and 0x00FFFFFF) or 0xCC000000.toInt()
            canvas.drawRect(vx1, vy1, vx2, vy2, boxPaint)

            val label = if (det.label >= 0 && det.label < classNames.size)
                classNames[det.label] else "cls${det.label}"
            val pct  = (det.confidence * 100).toInt().coerceIn(0, 100)
            val text = "$label $pct%"
            val th   = textPaint.textSize
            val tw   = textPaint.measureText(text)
            val ty   = if (vy1 >= th + 6f) vy1 - 4f else vy1 + th + 2f
            canvas.drawRect(vx1, ty - th - 2f, vx1 + tw + 8f, ty + 4f, bgPaint)
            canvas.drawText(text, vx1 + 4f, ty, textPaint)
        }

        // ── FPS counter (top-left) ────────────────────────────────────────────
        val ft = "%.1f FPS".format(fps)
        val fw = fpsPaint.measureText(ft)
        canvas.drawRoundRect(8f, 8f, fw + 20f, 56f, 6f, 6f, fpsBgPaint)
        canvas.drawText(ft, 14f, 46f, fpsPaint)

        // ── Diagnostic line (below FPS, yellow monospace) ─────────────────────
        if (debugLine.isNotEmpty()) {
            val dw = diagPaint.measureText(debugLine)
            canvas.drawRoundRect(8f, 62f, dw + 20f, 98f, 6f, 6f, fpsBgPaint)
            canvas.drawText(debugLine, 14f, 90f, diagPaint)
        }
    }
}
