package com.destik.yolodetector

import android.graphics.Bitmap

class YoloDetector {
    external fun nativeInit(
        paramPath: String, binPath: String,
        version: Int, inputSize: Int, numClasses: Int, useGPU: Boolean,
        out0: String, out1: String, out2: String, yfAnchors: String
    ): Boolean
    external fun nativeDetect(bitmap: Bitmap, confThreshold: Float, nmsThreshold: Float, numThreads: Int): Array<Detection>
    external fun nativeGetOutputNames(paramPath: String): Array<String>
    external fun nativeGetDiagnostics(): String
    external fun nativeRelease()

    fun init(config: ModelConfig): Boolean =
        nativeInit(config.paramPath, config.binPath,
            config.yoloVersion, config.inputSize, config.numClasses, config.useGPU,
            config.outputName0, config.outputName1, config.outputName2, config.yfAnchors)

    fun detect(bitmap: Bitmap, config: ModelConfig): Array<Detection> =
        nativeDetect(bitmap, config.confThreshold, config.nmsThreshold, config.numThreads)

    fun getOutputNames(config: ModelConfig): Array<String> = nativeGetOutputNames(config.paramPath)
    fun getDiagnostics(): String = nativeGetDiagnostics()
    fun release() = nativeRelease()

    companion object { init { System.loadLibrary("yolo_ncnn") } }
}
