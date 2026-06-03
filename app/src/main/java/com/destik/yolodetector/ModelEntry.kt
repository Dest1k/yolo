package com.destik.yolodetector

data class ModelEntry(
    val id: String,
    val name: String,
    val description: String,
    val yoloVersion: Int,
    val numClasses: Int,
    val inputSize: Int,
    val outputName0: String,
    val outputName1: String = "",
    val outputName2: String = "",
    val paramUrl: String,
    val binUrl: String,
    val approxMb: Int,
    // Per-model tuned defaults applied automatically on selection, so a freshly
    // chosen model "just works" without the user touching the settings sheet.
    val confThreshold: Float = 0.25f,
    val nmsThreshold: Float = 0.45f
) {
    companion object {
        // Curated to the detection models that nihui/ncnn-assets actually ships in a
        // clean, decodable NCNN layout. yolov6n / yolov7-tiny were dropped: their
        // exports use non-standard permuted / objectness blobs that don't decode
        // reliably, which is the opposite of "guaranteed best result".
        //
        // All URLs use raw.githubusercontent.com so LFS pointer detection works.
        private const val RAW = "https://raw.githubusercontent.com/nihui/ncnn-assets/master/models"

        val CATALOG = listOf(
            ModelEntry(
                id = "yolo11n",
                name = "YOLO11n",
                description = "Новейшая · COCO 80 классов · 640 · лучший баланс скорость/точность",
                yoloVersion = 8, numClasses = 80, inputSize = 640,
                outputName0 = "out0",
                paramUrl = "$RAW/yolo11n.ncnn.param",
                binUrl   = "$RAW/yolo11n.ncnn.bin",
                approxMb = 5,
                confThreshold = 0.25f
            ),
            ModelEntry(
                id = "yolov8n",
                name = "YOLOv8n",
                description = "Популярная · COCO 80 классов · 640 · быстрая и точная",
                yoloVersion = 8, numClasses = 80, inputSize = 640,
                outputName0 = "out0",
                paramUrl = "$RAW/yolov8n.ncnn.param",
                binUrl   = "$RAW/yolov8n.ncnn.bin",
                approxMb = 6,
                confThreshold = 0.25f
            ),
            ModelEntry(
                id = "yolov5s",
                name = "YOLOv5s",
                description = "Классическая · COCO 80 классов · 640 · стабильная на слабом железе",
                yoloVersion = 5, numClasses = 80, inputSize = 640,
                outputName0 = "out0", outputName1 = "out1", outputName2 = "out2",
                paramUrl = "$RAW/yolov5s.ncnn.param",
                binUrl   = "$RAW/yolov5s.ncnn.bin",
                approxMb = 7,
                confThreshold = 0.25f
            )
        )
    }
}
