package com.destik.yolodetector

data class Detection(
    val x: Float,          // normalized 0..1
    val y: Float,
    val w: Float,
    val h: Float,
    val label: Int,
    val confidence: Float
)
