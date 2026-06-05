package com.destik.yolodesktop

import kotlin.math.abs
import kotlin.math.hypot

/**
 * Visual-servoing follower: keeps a YOLO-detected target centred in frame by
 * driving the SIYI gimbal at a speed proportional to the target's offset from the
 * image centre (P-controller with a deadzone).
 *
 * Target selection favours continuity — it keeps following the detection nearest
 * the previously locked one, and only locks a fresh target (the largest box) when
 * there's no continuation. A short stability gate avoids chasing one-frame false
 * positives: the lock must persist a few ticks before any gimbal motion.
 *
 * Sign convention follows [SiyiGimbal.rotate] (yaw>0 right, pitch>0 up). If the
 * gimbal chases away from the target on your unit, flip [invertYaw]/[invertPitch].
 */
class GimbalFollower(
    private val gimbal: SiyiGimbal,
    private val maxSpeed: Int = 40,
    private val deadzone: Float = 0.05f,
    private val stableTicks: Int = 3,
    private val invertYaw: Boolean = false,
    private val invertPitch: Boolean = false
) {
    private var prev: Detection? = null
    private var lockCount = 0
    private var moving = false
    @Volatile private var pending: Pair<Float, Float>? = null   // normalised click point

    /** Manually lock the target under a click (normalised 0..1 coords). */
    fun requestPick(nx: Float, ny: Float) { pending = nx to ny }

    /** Advance one control tick; returns the currently locked target (for drawing), or null. */
    fun step(dets: List<Detection>, fw: Int, fh: Int): Detection? {
        if (fw <= 0 || fh <= 0 || dets.isEmpty()) { stop(); prev = null; lockCount = 0; return null }

        pending?.let { (nx, ny) ->
            pending = null
            pickAt(dets, nx * fw, ny * fh)?.let { prev = it; lockCount = 0 }
        }
        val t = pick(dets, prev, fw)
        prev = t
        if (t == null) { stop(); lockCount = 0; return null }

        lockCount++
        if (lockCount < stableTicks) { stop(); return t }   // show the lock, but wait for stability

        val cx = (t.x1 + t.x2) / 2f; val cy = (t.y1 + t.y2) / 2f
        val ex = cx / fw - 0.5f                              // -0.5 (left) .. 0.5 (right)
        val ey = cy / fh - 0.5f                              // -0.5 (top)  .. 0.5 (bottom)
        if (abs(ex) < deadzone && abs(ey) < deadzone) { stop(); return t }

        val gain = 2f * maxSpeed                             // full speed at frame edge
        var yawSpeed   = (ex * gain).toInt().coerceIn(-maxSpeed, maxSpeed)
        var pitchSpeed = (-ey * gain).toInt().coerceIn(-maxSpeed, maxSpeed)
        if (invertYaw) yawSpeed = -yawSpeed
        if (invertPitch) pitchSpeed = -pitchSpeed
        gimbal.rotate(yawSpeed, pitchSpeed)
        moving = true
        return t
    }

    /** Stop gimbal motion if we were moving (called when tracking is off / target lost). */
    fun stop() { if (moving) { gimbal.stopRotation(); moving = false } }

    /** Detection whose box contains the point, else the nearest by centre. */
    private fun pickAt(dets: List<Detection>, px: Float, py: Float): Detection? {
        val inside = dets.filter { px in it.x1..it.x2 && py in it.y1..it.y2 }
        val pool = inside.ifEmpty { dets }
        return pool.minByOrNull {
            val dx = (it.x1 + it.x2) / 2 - px; val dy = (it.y1 + it.y2) / 2 - py; dx * dx + dy * dy
        }
    }

    private fun pick(dets: List<Detection>, prev: Detection?, fw: Int): Detection? {
        if (prev != null) {
            val near = dets.minByOrNull { centerDist(it, prev) }
            if (near != null && centerDist(near, prev) < 0.3f * fw) return near   // continuity
        }
        return dets.maxByOrNull { (it.x2 - it.x1) * (it.y2 - it.y1) }             // else: largest
    }

    private fun centerDist(a: Detection, b: Detection) =
        hypot(((a.x1 + a.x2) - (b.x1 + b.x2)) / 2.0, ((a.y1 + a.y2) - (b.y1 + b.y2)) / 2.0).toFloat()
}
