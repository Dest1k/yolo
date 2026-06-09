package com.destik.yolodesktop

import java.awt.image.BufferedImage

/**
 * Lightweight single-object visual tracker — locks onto a user-drawn rectangle
 * and follows that patch across frames *independently of YOLO*. It's meant as a
 * hand-pick "in help of YOLO": you box a target with the mouse and a firmly
 * attached frame sticks to it even for objects the model never detects (or while
 * the slow CPU detector is between updates).
 *
 * Algorithm: normalised cross-correlation (NCC) template matching over a small
 * search window around the last position, evaluated at a few scales so the box
 * grows/shrinks with the object. The template is kept tiny (downsampled to a
 * fixed grid) so the whole thing costs well under a millisecond per frame on a
 * Raspberry Pi A76 — it runs on the stream thread without denting the FPS.
 *
 * The template slowly adapts to a confident match (appearance drift), and a miss
 * counter drops the lock after the object has been lost for a while. All state is
 * owned by a single caller thread (the stream loop), so no locking is needed —
 * the web panel hands in lock/clear requests via [Headless], not by calling here.
 */
class ObjectTracker(
    private val grid: Int = 32,          // template is grid×grid grayscale samples
    private val searchStep: Int = 2,     // px stride of the search scan
    private val lostNcc: Float = 0.35f,  // below this the frame counts as a miss
    private val adaptNcc: Float = 0.65f, // above this the template adapts to the match
    private val maxMiss: Int = 25         // drop the lock after this many misses (~1s @ 24fps)
) {
    data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float)

    private val tw = grid
    private val th = grid
    private var template = FloatArray(tw * th)
    private var tMean = 0f
    private var tNorm = 0f

    private var cx = 0f; private var cy = 0f      // box centre (px)
    private var bw = 0f; private var bh = 0f      // box size (px)
    private var miss = 0
    private var lastNcc = 0f

    @Volatile var locked = false; private set

    // Scratch grayscale plane of the current frame, reused across the scan.
    private var gray = IntArray(0)
    private var gw = 0; private var gh = 0

    /** Begin tracking the patch inside the given pixel rectangle of [frame]. */
    fun lock(frame: BufferedImage, x1: Float, y1: Float, x2: Float, y2: Float) {
        toGray(frame)
        val rx1 = x1.coerceIn(0f, (gw - 1).toFloat()); val ry1 = y1.coerceIn(0f, (gh - 1).toFloat())
        val rx2 = x2.coerceIn(rx1 + 4f, gw.toFloat()); val ry2 = y2.coerceIn(ry1 + 4f, gh.toFloat())
        cx = (rx1 + rx2) / 2f; cy = (ry1 + ry2) / 2f
        bw = rx2 - rx1; bh = ry2 - ry1
        sampleInto(template, rx1, ry1, rx2, ry2)
        recomputeTemplateStats()
        miss = 0; lastNcc = 1f; locked = true
    }

    /** Drop the current lock. */
    fun reset() { locked = false; miss = 0 }

    /**
     * Advance one frame. Returns the new box (still attached even through brief
     * misses) while the lock holds, or null once the target is lost / not locked.
     */
    fun update(frame: BufferedImage): Box? {
        if (!locked) return null
        toGray(frame)

        // Search a window around the last centre at a few scales; keep the best NCC.
        val radius = (maxOf(bw, bh) * 0.35f).toInt().coerceIn(6, 40)
        val scales = floatArrayOf(0.92f, 1f, 1.08f)
        var bestNcc = -1f; var bestCx = cx; var bestCy = cy; var bestW = bw; var bestH = bh
        for (s in scales) {
            val w = bw * s; val h = bh * s
            var dy = -radius
            while (dy <= radius) {
                var dx = -radius
                while (dx <= radius) {
                    val ncx = cx + dx; val ncy = cy + dy
                    val ncc = nccAt(ncx, ncy, w, h)
                    if (ncc > bestNcc) { bestNcc = ncc; bestCx = ncx; bestCy = ncy; bestW = w; bestH = h }
                    dx += searchStep
                }
                dy += searchStep
            }
        }
        lastNcc = bestNcc

        if (bestNcc >= lostNcc) {
            cx = bestCx; cy = bestCy
            // Ease the size so a single noisy scale pick doesn't make the box jitter.
            bw += (bestW - bw) * 0.5f; bh += (bestH - bh) * 0.5f
            clampBox()
            miss = 0
            if (bestNcc >= adaptNcc) adaptTemplate()
        } else {
            // Brief miss: keep the box where it was (still "stuck") until we give up.
            if (++miss > maxMiss) { locked = false; return null }
        }
        val (x1, y1, x2, y2) = boxCorners()
        return Box(x1, y1, x2, y2, bestNcc.coerceIn(0f, 1f))
    }

    // ── internals ────────────────────────────────────────────────────────────
    // (FloatArray.component1..4 come from the stdlib, used to destructure corners.)

    private fun boxCorners(): FloatArray =
        floatArrayOf(cx - bw / 2f, cy - bh / 2f, cx + bw / 2f, cy + bh / 2f)

    private fun clampBox() {
        bw = bw.coerceIn(8f, gw.toFloat()); bh = bh.coerceIn(8f, gh.toFloat())
        cx = cx.coerceIn(bw / 2f, gw - bw / 2f)
        cy = cy.coerceIn(bh / 2f, gh - bh / 2f)
    }

    /** Extract a grayscale plane of the frame into [gray] (reused buffer). */
    private fun toGray(frame: BufferedImage) {
        val w = frame.width; val h = frame.height
        if (gray.size != w * h) gray = IntArray(w * h)
        gw = w; gh = h
        val rgb = frame.getRGB(0, 0, w, h, null, 0, w)
        for (i in rgb.indices) {
            val p = rgb[i]
            gray[i] = (((p ushr 16) and 0xFF) * 77 + ((p ushr 8) and 0xFF) * 150 + (p and 0xFF) * 29) shr 8
        }
    }

    /** Sample the [tw]×[th] template grid out of a box region of the gray plane. */
    private fun sampleInto(dst: FloatArray, x1: Float, y1: Float, x2: Float, y2: Float) {
        val sx = (x2 - x1) / tw; val sy = (y2 - y1) / th
        var k = 0
        for (j in 0 until th) {
            val fy = (y1 + (j + 0.5f) * sy).toInt().coerceIn(0, gh - 1)
            val row = fy * gw
            for (i in 0 until tw) {
                val fx = (x1 + (i + 0.5f) * sx).toInt().coerceIn(0, gw - 1)
                dst[k++] = gray[row + fx].toFloat()
            }
        }
    }

    private fun recomputeTemplateStats() {
        var sum = 0f
        for (v in template) sum += v
        tMean = sum / template.size
        var sq = 0f
        for (v in template) { val d = v - tMean; sq += d * d }
        tNorm = kotlin.math.sqrt(sq).coerceAtLeast(1e-3f)
    }

    private val patch = FloatArray(tw * th)

    /** NCC between the template and a candidate box centred at (cx,cy) sized w×h. */
    private fun nccAt(ccx: Float, ccy: Float, w: Float, h: Float): Float {
        val x1 = ccx - w / 2f; val y1 = ccy - h / 2f
        sampleInto(patch, x1, y1, x1 + w, y1 + h)
        var sum = 0f
        for (v in patch) sum += v
        val pMean = sum / patch.size
        var cross = 0f; var pSq = 0f
        for (i in patch.indices) {
            val pd = patch[i] - pMean
            cross += pd * (template[i] - tMean)
            pSq += pd * pd
        }
        val pNorm = kotlin.math.sqrt(pSq)
        if (pNorm < 1e-3f) return 0f
        return cross / (pNorm * tNorm)
    }

    /** Blend the template toward the last confident match so it follows drift. */
    private fun adaptTemplate() {
        val (x1, y1, x2, y2) = boxCorners()
        sampleInto(patch, x1, y1, x2, y2)
        for (i in template.indices) template[i] = template[i] * 0.9f + patch[i] * 0.1f
        recomputeTemplateStats()
    }
}
