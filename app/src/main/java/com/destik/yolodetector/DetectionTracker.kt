package com.destik.yolodetector

/**
 * Stabilises per-frame detections so boxes stop flickering when the model
 * misses an object for a frame or two — e.g. motion blur while the camera is
 * being moved, or a momentary low-confidence dip.
 *
 * Each incoming detection is matched to a recently-seen track of the same class
 * by IoU (intersection over union). Matched tracks snap to the fresh box;
 * unmatched detections start new tracks; tracks that go unseen keep drawing at
 * their last position until [holdMs] elapses, then are dropped. The result: a
 * box survives brief detection gaps but still disappears shortly after the
 * object genuinely leaves the frame — independent of the model's confidence.
 *
 * All coordinates are normalised 0..1, matching [Detection]. Inference runs on a
 * single executor thread, but [reset] may be called from the UI thread, so the
 * public methods are synchronized.
 */
class DetectionTracker(
    private val holdMs: Long = 600L,
    private val iouThreshold: Float = 0.3f
) {
    private class Track(var det: Detection, var lastSeen: Long)

    private val tracks = ArrayList<Track>()

    /** Folds [dets] into the tracked set and returns the stabilised detections. */
    @Synchronized
    fun update(dets: Array<Detection>, nowMs: Long): Array<Detection> {
        val matched = BooleanArray(tracks.size)
        for (d in dets) {
            var best = -1
            var bestIou = iouThreshold
            for (i in tracks.indices) {
                if (matched[i] || tracks[i].det.label != d.label) continue
                val iou = iou(tracks[i].det, d)
                if (iou >= bestIou) { bestIou = iou; best = i }
            }
            if (best >= 0) {
                tracks[best].det = d
                tracks[best].lastSeen = nowMs
                matched[best] = true
            } else {
                tracks.add(Track(d, nowMs))
            }
        }
        tracks.removeAll { nowMs - it.lastSeen > holdMs }
        return Array(tracks.size) { tracks[it].det }
    }

    /** Clears all tracks — call when the source changes (camera flip, restart). */
    @Synchronized
    fun reset() = tracks.clear()

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = maxOf(a.x, b.x); val iy1 = maxOf(a.y, b.y)
        val ix2 = minOf(a.x + a.w, b.x + b.w); val iy2 = minOf(a.y + a.h, b.y + b.h)
        val iw = ix2 - ix1; val ih = iy2 - iy1
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val union = a.w * a.h + b.w * b.h - inter
        return if (union <= 0f) 0f else inter / union
    }
}
