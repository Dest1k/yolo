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
        // 4 готовые модели NanoDet-Plus (COCO 80 классов), как «Готовые модели» в обучалке:
        // экспортируются get_model.py одним кликом (--все), выкладываются в Releases репозитория
        // и здесь скачиваются/выбираются. yoloVersion=1 → нативный декодер NanoDet-Plus
        // (strides 8/16/32/64, reg_max 7). outputName0="output" — реальное имя выходного блоба
        // нативный код определяет сам, менять не нужно.
        //
        // КУДА ЗАЛИВАТЬ: собери `python get_model.py --все`, затем загрузи 8 файлов
        // (.param/.bin каждого варианта) в релиз репозитория с тегом nanodet-ncnn.
        private const val RELEASES =
            "https://github.com/dest1k/yolo/releases/download/nanodet-ncnn"

        private fun nd(id: String, name: String, desc: String, input: Int, mb: Int) = ModelEntry(
            id = id, name = name, description = desc,
            yoloVersion = 1, numClasses = 80, inputSize = input,
            outputName0 = "output",
            paramUrl = "$RELEASES/$id.param",
            binUrl   = "$RELEASES/$id.bin",
            approxMb = mb,
            // NanoDet-Plus: порог 0.35 / NMS 0.6 — сразу рабочие значения.
            confThreshold = 0.35f, nmsThreshold = 0.6f
        )

        val CATALOG = listOf(
            nd("nanodet-m-416", "NanoDet-Plus m · 416",
               "Баланс скорость/точность · COCO 80 · вход 416 (рекомендую)", 416, 4),
            nd("nanodet-m-320", "NanoDet-Plus m · 320",
               "Самая БЫСТРАЯ · COCO 80 · вход 320", 320, 4),
            nd("nanodet-m-1.5x-416", "NanoDet-Plus m-1.5x · 416",
               "Самая ТОЧНАЯ · COCO 80 · вход 416 · тяжелее", 416, 8),
            nd("nanodet-m-1.5x-320", "NanoDet-Plus m-1.5x · 320",
               "Точнее обычной · COCO 80 · вход 320", 320, 8)
        )
    }
}
