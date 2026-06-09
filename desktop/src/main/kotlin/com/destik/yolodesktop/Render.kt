package com.destik.yolodesktop

import java.awt.image.BufferedImage

/** Shared detection rendering + class labels, used by both the GUI and headless runners. */
object Render {

    val cocoLabels = arrayOf(
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    )

    fun labelFor(cls: Int, labels: List<String>? = null): String =
        labels?.getOrNull(cls) ?: cocoLabels.getOrNull(cls) ?: "cls$cls"

    private val palette = arrayOf(
        java.awt.Color(255, 80, 80),  java.awt.Color(80, 200, 80),
        java.awt.Color(80, 120, 255), java.awt.Color(255, 200, 0),
        java.awt.Color(200, 0, 200),  java.awt.Color(0, 200, 200)
    )

    /** Returns a copy of [src] with detection boxes + labels drawn on it.
     *  Optional [hud] text is drawn in the bottom-left corner (e.g. an FPS meter).
     *  [labels] overrides the built-in COCO names (for custom models).
     *  When [tracking] is on, a centre crosshair + TRACKING badge are drawn and the
     *  locked [target] is highlighted. */
    fun draw(src: BufferedImage, dets: List<Detection>, hud: String? = null,
             labels: List<String>? = null, target: Detection? = null,
             tracking: Boolean = false, manual: ObjectTracker.Box? = null): BufferedImage {
        val out = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_RGB)
        val g   = out.createGraphics()
        g.drawImage(src, 0, 0, null)
        g.stroke = java.awt.BasicStroke(2f)
        for (d in dets) {
            g.color = palette[d.cls % palette.size]
            g.drawRect(d.x1.toInt(), d.y1.toInt(), (d.x2 - d.x1).toInt(), (d.y2 - d.y1).toInt())
            val label = "${labelFor(d.cls, labels)} ${"%.2f".format(d.conf)}"
            val fm = g.fontMetrics
            val tw = fm.stringWidth(label)
            val th = fm.height
            g.fillRect(d.x1.toInt(), d.y1.toInt() - th, tw + 4, th)
            g.color = java.awt.Color.BLACK
            g.drawString(label, d.x1.toInt() + 2, d.y1.toInt() - 2)
        }
        if (tracking) {
            val cx = out.width / 2; val cy = out.height / 2
            g.color = java.awt.Color(255, 60, 60)
            g.stroke = java.awt.BasicStroke(2f)
            g.drawLine(cx - 16, cy, cx + 16, cy); g.drawLine(cx, cy - 16, cx, cy + 16)
            g.drawOval(cx - 6, cy - 6, 12, 12)
            if (target != null) {
                g.color = java.awt.Color(255, 230, 0)
                g.stroke = java.awt.BasicStroke(3f)
                g.drawRect(target.x1.toInt(), target.y1.toInt(),
                    (target.x2 - target.x1).toInt(), (target.y2 - target.y1).toInt())
                // line from frame centre to the target centre
                g.color = java.awt.Color(255, 60, 60)
                g.stroke = java.awt.BasicStroke(1f)
                g.drawLine(cx, cy, ((target.x1 + target.x2) / 2).toInt(), ((target.y1 + target.y2) / 2).toInt())
            }
            g.font = g.font.deriveFont(java.awt.Font.BOLD, 16f)
            g.color = java.awt.Color(255, 60, 60)
            g.drawString("TRACKING", out.width / 2 - 38, 22)
        }
        // Manually locked target (drawn by the mouse) — a firmly attached cyan box
        // with corner brackets so it reads as a hard lock, independent of YOLO.
        if (manual != null) {
            val mx = manual.x1.toInt(); val my = manual.y1.toInt()
            val mw = (manual.x2 - manual.x1).toInt(); val mh = (manual.y2 - manual.y1).toInt()
            val cyan = java.awt.Color(0, 230, 230)
            g.color = cyan
            g.stroke = java.awt.BasicStroke(2f)
            g.drawRect(mx, my, mw, mh)
            val c = minOf(mw, mh) / 4
            g.stroke = java.awt.BasicStroke(4f)
            g.drawLine(mx, my, mx + c, my);            g.drawLine(mx, my, mx, my + c)
            g.drawLine(mx + mw, my, mx + mw - c, my);  g.drawLine(mx + mw, my, mx + mw, my + c)
            g.drawLine(mx, my + mh, mx + c, my + mh);  g.drawLine(mx, my + mh, mx, my + mh - c)
            g.drawLine(mx + mw, my + mh, mx + mw - c, my + mh); g.drawLine(mx + mw, my + mh, mx + mw, my + mh - c)
            g.font = g.font.deriveFont(java.awt.Font.BOLD, 14f)
            g.drawString("LOCK ${"%.2f".format(manual.conf)}", mx + 2, (my - 4).coerceAtLeast(12))
        }
        if (hud != null) {
            g.font = g.font.deriveFont(java.awt.Font.BOLD, 16f)
            val fm = g.fontMetrics
            val tw = fm.stringWidth(hud)
            val th = fm.height
            val x = 6
            val y = out.height - 6
            g.color = java.awt.Color(0, 0, 0, 160)
            g.fillRect(x - 4, y - th + 2, tw + 8, th + 2)
            g.color = java.awt.Color(0, 230, 118)
            g.drawString(hud, x, y - 4)
        }
        g.dispose()
        return out
    }
}
