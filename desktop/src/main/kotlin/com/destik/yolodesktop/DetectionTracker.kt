package com.destik.yolodesktop

/**
 * Stabilises detections across frames so boxes don't flicker or disappear on a
 * momentary miss, and so they persist smoothly between the (slower) inference
 * updates while the stream runs at full camera FPS.
 *
 * This is the runtime equivalent of Ultralytics' `model.track()` — that API only
 * exists in the Python package, not in an exported ONNX/NCNN/RKNN model — so we
 * track here: each detection is matched to a recent track of the same class by
 * IoU, matched tracks snap to the fresh box, unseen tracks keep drawing until
 * [holdMs] elapses, then drop. Coordinates are pixel-space, matching [Detection].
 */
class DetectionTracker(
    private val holdMs: Long = 800L,
    private val iouThreshold: Float = 0.3f
) {
    private class Track(var det: Detection, var lastSeen: Long)

    private val tracks = ArrayList<Track>()

    @Synchronized
    fun update(dets: List<Detection>, nowMs: Long): List<Detection> {
        // Match only against tracks that existed at the start: tracks.add() below
        // grows the list within this call, so bound the inner loop by the original
        // count (and size `matched` to it) to avoid index-out-of-bounds.
        val n = tracks.size
        val matched = BooleanArray(n)
        for (d in dets) {
            var best = -1
            var bestIou = iouThreshold
            for (i in 0 until n) {
                if (matched[i] || tracks[i].det.cls != d.cls) continue
                val v = iou(tracks[i].det, d)
                if (v >= bestIou) { bestIou = v; best = i }
            }
            if (best >= 0) {
                tracks[best].det = d; tracks[best].lastSeen = nowMs; matched[best] = true
            } else {
                tracks.add(Track(d, nowMs))
            }
        }
        tracks.removeAll { nowMs - it.lastSeen > holdMs }
        return tracks.map { it.det }
    }

    @Synchronized
    fun reset() = tracks.clear()

    private fun iou(p: Detection, q: Detection): Float {
        val ix1 = maxOf(p.x1, q.x1); val iy1 = maxOf(p.y1, q.y1)
        val ix2 = minOf(p.x2, q.x2); val iy2 = minOf(p.y2, q.y2)
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        val union = (p.x2 - p.x1) * (p.y2 - p.y1) + (q.x2 - q.x1) * (q.y2 - q.y1) - inter
        return if (union <= 0f) 0f else inter / union
    }
}
