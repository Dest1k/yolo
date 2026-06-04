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

    fun labelFor(cls: Int): String = cocoLabels.getOrNull(cls) ?: "cls$cls"

    private val palette = arrayOf(
        java.awt.Color(255, 80, 80),  java.awt.Color(80, 200, 80),
        java.awt.Color(80, 120, 255), java.awt.Color(255, 200, 0),
        java.awt.Color(200, 0, 200),  java.awt.Color(0, 200, 200)
    )

    /** Returns a copy of [src] with detection boxes + labels drawn on it. */
    fun draw(src: BufferedImage, dets: List<Detection>): BufferedImage {
        val out = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_RGB)
        val g   = out.createGraphics()
        g.drawImage(src, 0, 0, null)
        g.stroke = java.awt.BasicStroke(2f)
        for (d in dets) {
            g.color = palette[d.cls % palette.size]
            g.drawRect(d.x1.toInt(), d.y1.toInt(), (d.x2 - d.x1).toInt(), (d.y2 - d.y1).toInt())
            val label = "${labelFor(d.cls)} ${"%.2f".format(d.conf)}"
            val fm = g.fontMetrics
            val tw = fm.stringWidth(label)
            val th = fm.height
            g.fillRect(d.x1.toInt(), d.y1.toInt() - th, tw + 4, th)
            g.color = java.awt.Color.BLACK
            g.drawString(label, d.x1.toInt() + 2, d.y1.toInt() - 2)
        }
        g.dispose()
        return out
    }
}
