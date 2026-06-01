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
    val approxMb: Int
) {
    companion object {
        // All detection models (COCO 80 classes) from nihui/ncnn-assets
        private const val RAW  = "https://raw.githubusercontent.com/nihui/ncnn-assets/master/models"
        private const val MEDIA = "https://media.githubusercontent.com/media/nihui/ncnn-assets/master/models"

        val CATALOG = listOf(
            ModelEntry(
                id = "yolov8n",
                name = "YOLOv8n",
                description = "COCO · 80 классов · 640×640 · ~6 MB",
                yoloVersion = 8, numClasses = 80, inputSize = 640,
                outputName0 = "output0",
                paramUrl  = "$RAW/yolov8n.ncnn.param",
                binUrl    = "$MEDIA/yolov8n.ncnn.bin",
                approxMb  = 6
            ),
            ModelEntry(
                id = "yolo11n",
                name = "YOLO11n",
                description = "COCO · 80 классов · 640×640 · ~5 MB",
                yoloVersion = 8, numClasses = 80, inputSize = 640,
                outputName0 = "output0",
                paramUrl  = "$RAW/yolo11n.ncnn.param",
                binUrl    = "$MEDIA/yolo11n.ncnn.bin",
                approxMb  = 5
            ),
            ModelEntry(
                id = "yolov5s",
                name = "YOLOv5s",
                description = "COCO · 80 классов · 640×640 · ~7 MB",
                yoloVersion = 5, numClasses = 80, inputSize = 640,
                outputName0 = "output", outputName1 = "output1", outputName2 = "output2",
                paramUrl  = "$RAW/yolov5s.ncnn.param",
                binUrl    = "$MEDIA/yolov5s.ncnn.bin",
                approxMb  = 7
            ),
            ModelEntry(
                id = "yolov6n",
                name = "YOLOv6n",
                description = "COCO · 80 классов · 640×640 · ~4 MB",
                yoloVersion = 8, numClasses = 80, inputSize = 640,
                outputName0 = "output",
                paramUrl  = "$RAW/yolov6n.param",
                binUrl    = "$MEDIA/yolov6n.bin",
                approxMb  = 4
            ),
            ModelEntry(
                id = "yolov7tiny",
                name = "YOLOv7-tiny",
                description = "COCO · 80 классов · 640×640 · ~6 MB",
                yoloVersion = 7, numClasses = 80, inputSize = 640,
                outputName0 = "output", outputName1 = "output1", outputName2 = "output2",
                paramUrl  = "$RAW/yolov7-tiny.param",
                binUrl    = "$MEDIA/yolov7-tiny.bin",
                approxMb  = 6
            )
        )
    }
}
