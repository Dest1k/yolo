import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("jvm")
    id("org.jetbrains.compose")
}

val osName: String = System.getProperty("os.name").lowercase()
val isMac     = osName.contains("mac")
val isWindows = osName.contains("windows")
val isLinux   = osName.contains("linux")

dependencies {
    implementation(compose.desktop.currentOs)
    implementation(compose.material3)
    implementation(compose.materialIconsExtended)

    // ── ONNX Runtime ───────────────────────────────────────────────────────────
    // GPU variant on Linux/Windows: enables CUDA EP + DirectML EP at runtime.
    // Falls back to CPU automatically when no compatible GPU/driver is present.
    if (isMac) {
        implementation("com.microsoft.onnxruntime:onnxruntime:1.18.0")
    } else {
        implementation("com.microsoft.onnxruntime:onnxruntime-gpu:1.18.0")
    }

    // ── DJL PyTorch (.pt / TorchScript models) ─────────────────────────────────
    implementation("ai.djl:api:0.27.0")
    implementation("ai.djl.pytorch:pytorch-engine:0.27.0")

    // CPU-only native for all platforms (always available fallback)
    when {
        isLinux   -> runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:2.1.1:linux-x86_64")
        isWindows -> runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:2.1.1:win-x86_64")
        isMac     -> runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:2.1.1:osx-x86_64")
    }
    // CUDA 12.1 native (NVIDIA GPU — loaded automatically when CUDA is present)
    when {
        isLinux   -> runtimeOnly("ai.djl.pytorch:pytorch-native-cu121:2.1.1:linux-x86_64")
        isWindows -> runtimeOnly("ai.djl.pytorch:pytorch-native-cu121:2.1.1:win-x86_64")
    }

    // ── JavaCV for webcam (OpenCV grabber) ─────────────────────────────────────
    implementation("org.bytedeco:javacv:1.5.10") {
        exclude(group = "org.bytedeco", module = "flycapture")
        exclude(group = "org.bytedeco", module = "libdc1394")
        exclude(group = "org.bytedeco", module = "libfreenect")
        exclude(group = "org.bytedeco", module = "videoinput")
        exclude(group = "org.bytedeco", module = "artoolkitplus")
        exclude(group = "org.bytedeco", module = "flandmark")
        exclude(group = "org.bytedeco", module = "leptonica")
        exclude(group = "org.bytedeco", module = "tesseract")
    }
    when {
        isLinux   -> {
            runtimeOnly("org.bytedeco:opencv:4.9.0-1.5.10:linux-x86_64")
            runtimeOnly("org.bytedeco:ffmpeg:6.1.1-1.5.10:linux-x86_64")
        }
        isWindows -> {
            runtimeOnly("org.bytedeco:opencv:4.9.0-1.5.10:windows-x86_64")
            runtimeOnly("org.bytedeco:ffmpeg:6.1.1-1.5.10:windows-x86_64")
        }
        isMac     -> {
            runtimeOnly("org.bytedeco:opencv:4.9.0-1.5.10:macosx-arm64")
            runtimeOnly("org.bytedeco:ffmpeg:6.1.1-1.5.10:macosx-arm64")
        }
    }

    implementation("com.google.code.gson:gson:2.10.1")
    // Provides Dispatchers.Main backed by Swing EDT (required for UI state mutations)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.7.3")
}

kotlin {
    jvmToolchain(17)
}

compose.desktop {
    application {
        mainClass = "com.destik.yolodesktop.MainKt"

        // Ensure bundled JVM is large enough for ML workloads
        jvmArgs("-Xmx4g", "-Xms512m")

        nativeDistributions {
            targetFormats(TargetFormat.Deb, TargetFormat.Msi, TargetFormat.Dmg)
            packageName    = "YoloDetector"
            packageVersion = "1.0.0"
            description    = "Real-time YOLO object detection"
            vendor         = "destik"

            linux {
                iconFile.set(project.file("src/main/resources/icon.png"))
            }
            windows {
                menuGroup   = "YoloDetector"
                upgradeUuid = "4e3a2b1c-0f5d-4e7a-9b8c-2d3e4f5a6b7c"
            }
        }

        buildTypes.release.proguard {
            isEnabled.set(false)
        }
    }
}
