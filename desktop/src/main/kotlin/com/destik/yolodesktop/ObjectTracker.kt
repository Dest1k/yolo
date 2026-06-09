package com.destik.yolodesktop

import java.awt.image.BufferedImage
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Lightweight single-object visual tracker — locks onto a user-drawn rectangle
 * and follows that patch across frames *independently of YOLO*. It's meant as a
 * hand-pick "in help of YOLO": you box a target with the mouse and a firmly
 * attached frame sticks to it even for objects the model never detects (or while
 * the slow CPU detector is between updates).
 *
 * Algorithm: normalised cross-correlation (NCC) template matching. Each frame it
 * first does a cheap local search around the last position (a few scales so the
 * box grows/shrinks with the object). When that fails — typically an occlusion —
 * it freezes the box and runs a re-acquisition scan over a window that *expands*
 * every miss, so once the object reappears (even somewhere else, because it was
 * moved while hidden) the box snaps back onto it.
 *
 * Robustness tricks:
 *   • An immutable **anchor** template captured at lock time is matched alongside
 *     the slowly-adapting one (we take the better of the two). The adaptive copy
 *     tracks appearance drift; the anchor lets us recover after drift/occlusion.
 *   • The template only adapts on a *confident* match, so a hand/occluder passing
 *     in front can't poison it and steal the lock.
 *   • The box is held (not moved) through brief misses, so it never drifts onto
 *     whatever is covering the target.
 *
 * Cost is bounded: templates are tiny (grid×grid), local search is small, and the
 * expensive wide re-acquire only runs while the target is actually lost (and only
 * every other miss frame). On a Raspberry Pi A76 this stays well within frame
 * budget and runs on the stream thread without denting FPS.
 *
 * All state is owned by a single caller thread (the stream loop), so no locking is
 * needed — the web panel hands in lock/clear requests via [Headless], not here.
 */
class ObjectTracker(
    private val grid: Int = 32,          // template is grid×grid grayscale samples
    private val searchStep: Int = 2,     // px stride of the local search scan
    private val trackNcc: Float = 0.42f, // accept a local match (continue following) above this
    private val adaptNcc: Float = 0.72f, // only above this does the template adapt (avoid poisoning)
    private val reacqNcc: Float = 0.55f, // re-lock a lost target only on a strong match (avoid false locks)
    private val maxMiss: Int = 90         // keep trying to re-acquire this many frames (~3–4 s) before dropping
) {
    data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float)

    private val tw = grid
    private val th = grid
    private var template = FloatArray(tw * th)   // slowly-adapting appearance
    private var tMean = 0f; private var tNorm = 0f
    private var anchor = FloatArray(tw * th)      // immutable appearance from lock time
    private var aMean = 0f; private var aNorm = 0f

    private var cx = 0f; private var cy = 0f      // box centre (px)
    private var bw = 0f; private var bh = 0f      // box size (px)
    private var miss = 0
    private var lastNcc = 0f

    @Volatile var locked = false; private set

    // Scratch grayscale plane of the current frame, reused across the scan.
    private var gray = IntArray(0)
    private var gw = 0; private var gh = 0
    private val patch = FloatArray(tw * th)

    /** Begin tracking the patch inside the given pixel rectangle of [frame]. */
    fun lock(frame: BufferedImage, x1: Float, y1: Float, x2: Float, y2: Float) {
        toGray(frame)
        val rx1 = x1.coerceIn(0f, (gw - 1).toFloat()); val ry1 = y1.coerceIn(0f, (gh - 1).toFloat())
        val rx2 = x2.coerceIn(rx1 + 4f, gw.toFloat()); val ry2 = y2.coerceIn(ry1 + 4f, gh.toFloat())
        cx = (rx1 + rx2) / 2f; cy = (ry1 + ry2) / 2f
        bw = rx2 - rx1; bh = ry2 - ry1
        sampleInto(template, rx1, ry1, rx2, ry2)
        System.arraycopy(template, 0, anchor, 0, template.size)   // anchor = the original view
        recomputeStats(template).let { tMean = it.first; tNorm = it.second }
        aMean = tMean; aNorm = tNorm
        miss = 0; lastNcc = 1f; locked = true
    }

    /** Drop the current lock. */
    fun reset() { locked = false; miss = 0 }

    /**
     * Advance one frame. Returns the new box (still attached even through brief
     * misses / occlusions) while the lock holds, or null once the target is lost.
     */
    fun update(frame: BufferedImage): Box? {
        if (!locked) return null
        toGray(frame)

        // 1) Local search around the last position — the common, cheap case.
        val r = (max(bw, bh) * 0.35f).toInt().coerceIn(6, 40)
        val local = scan(cx, cy, r, searchStep, SCALES_LOCAL)
        lastNcc = local[0]
        if (local[0] >= trackNcc) {
            cx = local[1]; cy = local[2]
            // Ease the size so a noisy scale pick doesn't make the box jitter.
            bw += (local[3] - bw) * 0.5f; bh += (local[4] - bh) * 0.5f
            clampBox(); miss = 0
            if (local[0] >= adaptNcc) adaptTemplate()
            return box(local[0])
        }

        // 2) Miss (occlusion / lost). Freeze the box and, on alternating frames,
        //    re-acquire over a window that grows the longer we've been lost — so a
        //    target that moved while hidden is found when it reappears.
        miss++
        if (miss % 2 == 0) {
            val grow = 1f + miss * 0.4f
            val rr = (max(bw, bh) * 1.5f * grow).toInt().coerceIn(24, 250)
            val re = scan(cx, cy, rr, max(6, grid / 6), SCALES_REACQ)
            if (re[0] >= reacqNcc) {
                cx = re[1]; cy = re[2]; bw = re[3]; bh = re[4]
                clampBox(); miss = 0; lastNcc = re[0]
                return box(re[0])
            }
        }
        if (miss > maxMiss) { locked = false; return null }
        return box(local[0].coerceAtLeast(0f))   // frozen box, still attached
    }

    // ── internals ────────────────────────────────────────────────────────────

    private fun box(conf: Float): Box {
        val x1 = cx - bw / 2f; val y1 = cy - bh / 2f
        return Box(x1, y1, cx + bw / 2f, cy + bh / 2f, conf.coerceIn(0f, 1f))
    }

    private fun boxCorners(): FloatArray =
        floatArrayOf(cx - bw / 2f, cy - bh / 2f, cx + bw / 2f, cy + bh / 2f)

    private fun clampBox() {
        bw = bw.coerceIn(8f, gw.toFloat()); bh = bh.coerceIn(8f, gh.toFloat())
        cx = cx.coerceIn(bw / 2f, gw - bw / 2f)
        cy = cy.coerceIn(bh / 2f, gh - bh / 2f)
    }

    /**
     * Scan candidate boxes around (centerX,centerY) within ±radius (stride [step])
     * at the given [scales] of the current box size. Returns [ncc,cx,cy,w,h] of the
     * best match. Each candidate is scored against *both* templates (best wins).
     */
    private fun scan(centerX: Float, centerY: Float, radius: Int, step: Int, scales: FloatArray): FloatArray {
        var bN = -1f; var bX = centerX; var bY = centerY; var bW = bw; var bH = bh
        for (s in scales) {
            val w = bw * s; val h = bh * s
            var dy = -radius
            while (dy <= radius) {
                var dx = -radius
                while (dx <= radius) {
                    val n = matchAt(centerX + dx, centerY + dy, w, h)
                    if (n > bN) { bN = n; bX = centerX + dx; bY = centerY + dy; bW = w; bH = h }
                    dx += step
                }
                dy += step
            }
        }
        return floatArrayOf(bN, bX, bY, bW, bH)
    }

    /** Best NCC of a candidate box (centre cx,cy, size w×h) vs the adaptive OR anchor template. */
    private fun matchAt(ccx: Float, ccy: Float, w: Float, h: Float): Float {
        val x1 = ccx - w / 2f; val y1 = ccy - h / 2f
        sampleInto(patch, x1, y1, x1 + w, y1 + h)
        var sum = 0f
        for (v in patch) sum += v
        val pMean = sum / patch.size
        var pSq = 0f; var cT = 0f; var cA = 0f
        for (i in patch.indices) {
            val pd = patch[i] - pMean
            pSq += pd * pd
            cT += pd * (template[i] - tMean)
            cA += pd * (anchor[i] - aMean)
        }
        val pNorm = sqrt(pSq)
        if (pNorm < 1e-3f) return 0f
        return max(cT / (pNorm * tNorm), cA / (pNorm * aNorm))
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
            val rowBase = fy * gw
            for (i in 0 until tw) {
                val fx = (x1 + (i + 0.5f) * sx).toInt().coerceIn(0, gw - 1)
                dst[k++] = gray[rowBase + fx].toFloat()
            }
        }
    }

    /** Mean and L2-norm of (values - mean) for a template. */
    private fun recomputeStats(t: FloatArray): Pair<Float, Float> {
        var sum = 0f
        for (v in t) sum += v
        val mean = sum / t.size
        var sq = 0f
        for (v in t) { val d = v - mean; sq += d * d }
        return mean to sqrt(sq).coerceAtLeast(1e-3f)
    }

    /** Blend the adaptive template toward the current (confident) match. */
    private fun adaptTemplate() {
        val (x1, y1, x2, y2) = boxCorners()
        sampleInto(patch, x1, y1, x2, y2)
        for (i in template.indices) template[i] = template[i] * 0.9f + patch[i] * 0.1f
        recomputeStats(template).let { tMean = it.first; tNorm = it.second }
    }

    private companion object {
        val SCALES_LOCAL = floatArrayOf(0.9f, 1f, 1.1f)
        val SCALES_REACQ = floatArrayOf(0.7f, 0.85f, 1f, 1.2f, 1.45f)
    }
}
