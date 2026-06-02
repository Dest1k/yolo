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
    private val countPaint = Paint().apply { textSize = 34f; color = Color.WHITE; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD }

    private val colors = intArrayOf(
        0xFFE74C3C.toInt(), 0xFF3498DB.toInt(), 0xFF2ECC71.toInt(),
        0xFFF39C12.toInt(), 0xFF9B59B6.toInt(), 0xFF1ABC9C.toInt(),
        0xFFE67E22.toInt(), 0xFF607D8B.toInt(), 0xFFEC407A.toInt(),
        0xFF00BCD4.toInt()
    )

    private var detections: Array<Detection> = emptyArray()
    private var classNames: List<String> = emptyList()
    private var fps: Float = 0f
    private var imgAspect: Float = 9f / 16f
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
        val W = width.toFloat(); val H = height.toFloat()
        if (W == 0f || H == 0f) return

        // ── FILL_CENTER coordinate transform ────────────────────────────────
        val viewAspect = W / H
        val cropX: Float; val cropY: Float; val scaleX: Float; val scaleY: Float
        if (imgAspect > viewAspect) {
            val k = imgAspect / viewAspect
            cropX = (1f - 1f/k) / 2f; cropY = 0f; scaleX = 1f/k; scaleY = 1f
        } else {
            val k = viewAspect / imgAspect
            cropX = 0f; cropY = (1f - 1f/k) / 2f; scaleX = 1f; scaleY = 1f/k
        }

        // ── Detection boxes ─────────────────────────────────────────────────
        drawBoxes(canvas, W, H, cropX, cropY, scaleX, scaleY)

        // ── FPS counter (top-left) ────────────────────────────────────────────
        val ft = "%.1f FPS".format(fps)
        val fw = fpsPaint.measureText(ft)
        canvas.drawRoundRect(8f, 8f, fw + 20f, 56f, 6f, 6f, fpsBgPaint)
        canvas.drawText(ft, 14f, 46f, fpsPaint)

        // ── Diagnostic line (below FPS) ────────────────────────────────────────
        if (debugLine.isNotEmpty()) {
            val dw = diagPaint.measureText(debugLine)
            canvas.drawRoundRect(8f, 62f, dw + 20f, 98f, 6f, 6f, fpsBgPaint)
            canvas.drawText(debugLine, 14f, 90f, diagPaint)
        }

        // ── Per-class counters (top-right) ────────────────────────────────────
        if (detections.isNotEmpty()) {
            val counts = detections.groupBy { it.label }
                .mapValues { it.value.size }.entries
                .sortedByDescending { it.value }
            var cy = 16f + countPaint.textSize
            for ((lbl, cnt) in counts) {
                val name = if (lbl >= 0 && lbl < classNames.size) classNames[lbl] else "cls$lbl"
                val color = colors[abs(lbl) % colors.size]
                val txt = "$name: $cnt"
                val tw = countPaint.measureText(txt)
                val rx = W - tw - 20f
                bgPaint.color = 0xBB000000.toInt()
                canvas.drawRoundRect(rx - 6f, cy - countPaint.textSize - 2f, W - 10f, cy + 6f, 6f, 6f, bgPaint)
                countPaint.color = color
                canvas.drawText(txt, rx, cy, countPaint)
                cy += countPaint.textSize + 8f
            }
        }
    }

    private fun drawBoxes(
        canvas: Canvas, W: Float, H: Float,
        cropX: Float, cropY: Float, scaleX: Float, scaleY: Float
    ) {
        for (det in detections) {
            val vx1 = ((det.x - cropX) / scaleX * W).coerceIn(0f, W)
            val vy1 = ((det.y - cropY) / scaleY * H).coerceIn(0f, H)
            val vx2 = ((det.x + det.w - cropX) / scaleX * W).coerceIn(0f, W)
            val vy2 = ((det.y + det.h - cropY) / scaleY * H).coerceIn(0f, H)
            if (vx2 <= vx1 || vy2 <= vy1) continue
            drawBox(canvas, vx1, vy1, vx2, vy2, det)
        }
    }

    private fun drawBox(canvas: Canvas, x1: Float, y1: Float, x2: Float, y2: Float, det: Detection) {
        val color = colors[abs(det.label) % colors.size]
        boxPaint.color = color
        bgPaint.color  = (color and 0x00FFFFFF) or 0xCC000000.toInt()
        canvas.drawRect(x1, y1, x2, y2, boxPaint)
        val label = if (det.label >= 0 && det.label < classNames.size) classNames[det.label] else "cls${det.label}"
        val pct  = (det.confidence * 100).toInt().coerceIn(0, 100)
        val text = "$label $pct%"
        val th   = textPaint.textSize
        val tw   = textPaint.measureText(text)
        val ty   = if (y1 >= th + 6f) y1 - 4f else y1 + th + 2f
        canvas.drawRect(x1, ty - th - 2f, x1 + tw + 8f, ty + 4f, bgPaint)
        canvas.drawText(text, x1 + 4f, ty, textPaint)
    }

    /** Draw detection boxes onto an arbitrary bitmap (full-frame, no display crop). */
    fun drawBoxesOnBitmap(bmp: Bitmap, dets: Array<Detection>, names: List<String>) {
        val canvas = Canvas(bmp)
        val W = bmp.width.toFloat(); val H = bmp.height.toFloat()
        val savedNames = classNames; classNames = names
        for (det in dets) {
            drawBox(canvas,
                (det.x * W).coerceIn(0f, W), (det.y * H).coerceIn(0f, H),
                ((det.x + det.w) * W).coerceIn(0f, W), ((det.y + det.h) * H).coerceIn(0f, H),
                det
            )
        }
        classNames = savedNames
    }
}
