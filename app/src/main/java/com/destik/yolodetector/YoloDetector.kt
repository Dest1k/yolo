package com.destik.yolodetector

import android.graphics.Bitmap

class YoloDetector {
    external fun nativeInit(
        paramPath: String, binPath: String,
        version: Int, inputSize: Int, numClasses: Int, useGPU: Boolean,
        out0: String, out1: String, out2: String
    ): Boolean
    external fun nativeDetect(bitmap: Bitmap, confThreshold: Float, nmsThreshold: Float, numThreads: Int): Array<Detection>
    external fun nativeGetOutputNames(): Array<String>
    external fun nativeGetDiagnostics(): String
    external fun nativeRelease()

    fun init(config: ModelConfig): Boolean =
        nativeInit(config.paramPath, config.binPath,
            config.yoloVersion, config.inputSize, config.numClasses, !config.cpuOnly,
            config.outputName0, config.outputName1, config.outputName2)

    fun detect(bitmap: Bitmap, config: ModelConfig): Array<Detection> =
        nativeDetect(bitmap, config.confThreshold, config.nmsThreshold,
            Runtime.getRuntime().availableProcessors())

    fun getOutputNames(): Array<String> = nativeGetOutputNames()
    fun getDiagnostics(): String = nativeGetDiagnostics()
    fun release() = nativeRelease()

    companion object { init { System.loadLibrary("yolo_ncnn") } }
}
