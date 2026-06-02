package com.destik.yolodetector

data class ModelConfig(
    var paramPath: String = "",
    var binPath: String = "",
    var yoloVersion: Int = 10,
    var inputSize: Int = 640,
    var numClasses: Int = 80,
    var confThreshold: Float = 0.15f,
    var nmsThreshold: Float = 0.45f,
    var numThreads: Int = 8,
    var useGPU: Boolean = true,
    var outputName0: String = "output0",
    var outputName1: String = "output1",
    var outputName2: String = "output2",
    var classNames: List<String> = emptyList(),
    var onnxPath: String = "",
    var engine: String = "ncnn"   // "ncnn" or "onnx"
)
