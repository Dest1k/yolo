package com.destik.yolodesktop

/**
 * Probes the system `ffmpeg` for the hardware codecs the board actually exposes,
 * so the pipeline can offload encode/decode to silicon when it's there and fall
 * back to software when it isn't — the whole point on weak single-board computers
 * (Raspberry Pi, Orange Pi, Rock Pi, x86 mini-PCs with an iGPU…).
 *
 * Detection is by name from `ffmpeg -hide_banner -encoders/-decoders`, in board
 * preference order. Nothing here forces a codec onto a board that lacks it: the
 * caller picks the first *present* hardware option, else a software codec.
 *
 * Results are cached — the probe spawns ffmpeg once and is reused for the run.
 */
object HwAccel {

    /** Hardware H.264 encoders we know how to drive, best-first by platform. */
    private val H264_ENCODERS = listOf(
        "h264_v4l2m2m",   // Raspberry Pi / generic V4L2 stateful encoder (the SBC case)
        "h264_nvenc",     // NVIDIA (Jetson / desktop GPU)
        "h264_vaapi",     // Intel / AMD via VA-API (x86 mini-PCs)
        "h264_qsv",       // Intel QuickSync
        "h264_omx"        // legacy Pi (older firmware / 32-bit)
    )

    private val encoders: Set<String> by lazy { listCodecs("-encoders") }
    private val decoders: Set<String> by lazy { listCodecs("-decoders") }

    val ffmpegAvailable: Boolean by lazy { runCatching { run("-version").isNotEmpty() }.getOrDefault(false) }

    /** First available hardware H.264 encoder, or "libx264" (software) as fallback. */
    fun h264Encoder(): String = H264_ENCODERS.firstOrNull { it in encoders } ?: "libx264"

    /** Whether the chosen encoder is a real hardware path (vs. software libx264). */
    fun isHardware(encoder: String): Boolean = encoder != "libx264" && encoder != "mpeg4"

    /**
     * Extra `-vf`/`-c:v` setup some hardware encoders need before the encoder, e.g.
     * VA-API wants the frame uploaded to a hw surface in NV12. Returned as the list
     * of ffmpeg args to insert right before `-c:v <encoder>`.
     */
    fun encoderPreFilter(encoder: String): List<String> = when (encoder) {
        "h264_vaapi" -> listOf(
            "-vaapi_device", System.getenv("YOLO_VAAPI_DEVICE")?.trim()?.ifEmpty { null } ?: "/dev/dri/renderD128",
            "-vf", "format=nv12,hwupload"
        )
        else -> emptyList()
    }

    private fun listCodecs(flag: String): Set<String> = runCatching {
        // Lines look like: " V....D h264_v4l2m2m  V4L2 mem2mem H.264 encoder"
        run(flag).lineSequence()
            .mapNotNull { line ->
                val t = line.trim()
                if (t.length < 8 || !t[0].isLetter() && t[0] != '.') return@mapNotNull null
                t.split(Regex("\\s+")).getOrNull(1)
            }
            .toSet()
    }.getOrDefault(emptySet())

    private fun run(vararg args: String): String {
        val p = ProcessBuilder(listOf("ffmpeg", "-hide_banner") + args)
            .redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText()
        p.waitFor()
        return out
    }
}
