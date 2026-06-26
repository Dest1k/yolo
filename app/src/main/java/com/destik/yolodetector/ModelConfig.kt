package com.destik.yolodetector

data class ModelConfig(
    var paramPath: String = "",
    var binPath: String = "",
    // По умолчанию NanoDet-Plus (нативный декодер: strides 8/16/32/64, reg_max 7).
    var yoloVersion: Int = 1,
    var inputSize: Int = 416,
    var numClasses: Int = 80,
    var confThreshold: Float = 0.35f,
    var nmsThreshold: Float = 0.6f,
    var numThreads: Int = 8,
    // Default OFF: ncnn-Vulkan crashes on some models (e.g. YOLOv11) on certain
    // devices, and for end-to-end / NMS-free ONNX models NNAPI is slower than
    // CPU+XNNPACK. Users can opt back into GPU in settings.
    var useGPU: Boolean = false,
    var outputName0: String = "output",
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
    // Держим рамки сквозь короткие пропуски детекции (IoU-трекинг) — чтобы не мигали
    // при движении камеры. Включено для максимальной цепкости рамок.
    var stabilizeBoxes: Boolean = true,
    // Поворачиваем кадр «как надо» перед инференсом, чтобы портрет ловился так же
    // хорошо, как ландшафт. Включено по умолчанию (цепкость в обеих ориентациях).
    var uprightInference: Boolean = true
)
