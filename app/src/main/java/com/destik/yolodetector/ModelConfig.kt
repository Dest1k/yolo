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
    // Default OFF: ncnn-Vulkan crashes on some models (e.g. YOLOv11) on certain
    // devices, and for end-to-end / NMS-free ONNX models NNAPI is slower than
    // CPU+XNNPACK. Users can opt back into GPU in settings.
    var useGPU: Boolean = false,
    var outputName0: String = "output0",
    var outputName1: String = "output1",
    var outputName2: String = "output2",
    // YOLO-FastestV2 anchors, pixels at the training input, in the trainer's
    // `anchors=` order: [stride16 na pairs][stride32 na pairs]. na is derived from
    // the model; paste your training config's value here if you regenerated anchors.
    var yfAnchors: String = "12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87",
    var classNames: List<String> = emptyList(),
    var onnxPath: String = "",
    var engine: String = "ncnn",   // "ncnn" or "onnx"
    var streamUrl: String = "",    // if non-empty, use network stream instead of camera
    // Hold boxes through brief detection gaps (IoU tracking) so they stop
    // flickering when the camera moves. On by default.
    var stabilizeBoxes: Boolean = true,
    // Rotate the frame upright before inference so portrait orientation detects
    // as well as landscape. Costs one bitmap rotation per inference frame, so
    // it's off by default.
    var uprightInference: Boolean = false
)
