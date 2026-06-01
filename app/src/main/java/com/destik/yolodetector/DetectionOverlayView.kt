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
    private val bgPaint  = Paint().apply { style = Paint.Style.FILL }
    private val fpsBgPaint = Paint().apply { color = 0xBB000000.toInt(); style = Paint.Style.FILL }
    private val fpsPaint   = Paint().apply { textSize = 42f; color = Color.WHITE; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD }

    private val colors = intArrayOf(
        0xFFE74C3C.toInt(), 0xFF3498DB.toInt(), 0xFF2ECC71.toInt(),
        0xFFF39C12.toInt(), 0xFF9B59B6.toInt(), 0xFF1ABC9C.toInt(),
        0xFFE67E22.toInt(), 0xFF607D8B.toInt(), 0xFFEC407A.toInt(),
        0xFF00BCD4.toInt()
    )

    private var detections: Array<Detection> = emptyArray()
    private var classNames: List<String> = emptyList()
    private var fps: Float = 0f
    // Aspect ratio of the image AFTER rotation (portrait: <1, landscape: >1)
    private var imgAspect: Float = 9f / 16f

    fun setImageAspect(aspect: Float) { imgAspect = aspect }

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

        // CameraX PreviewView uses FILL_CENTER: scales image to fill the view,
        // cropping the axis that overflows. We apply the same transform to boxes.
        val viewAspect = W / H
        val cropX: Float   // fraction of image width cropped from each side
        val cropY: Float   // fraction of image height cropped from each side
        val scaleX: Float  // (1 - 2*cropX) = visible fraction of image width
        val scaleY: Float
        if (imgAspect > viewAspect) {
            // Image wider than view: fill height, crop left+right
            val k = imgAspect / viewAspect   // k > 1
            cropX = (1f - 1f / k) / 2f
            cropY = 0f
            scaleX = 1f / k
            scaleY = 1f
        } else {
            // Image taller than view: fill width, crop top+bottom
            val k = viewAspect / imgAspect   // k > 1
            cropX = 0f
            cropY = (1f - 1f / k) / 2f
            scaleX = 1f
            scaleY = 1f / k
        }

        for (det in detections) {
            // Map from image-normalized to view-pixel space via the crop transform
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

        // FPS
        val ft = "%.1f FPS".format(fps)
        canvas.drawRoundRect(8f, 8f, fpsPaint.measureText(ft) + 20f, 62f, 8f, 8f, fpsBgPaint)
        canvas.drawText(ft, 14f, 52f, fpsPaint)
    }
}
