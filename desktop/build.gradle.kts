import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("jvm")
    id("org.jetbrains.compose")
}

val osName: String = System.getProperty("os.name").lowercase()
val isMac     = osName.contains("mac")
val isWindows = osName.contains("windows")
val isLinux   = osName.contains("linux")

// Architecture matters for single-board computers: Raspberry Pi / Orange Pi /
// Rock Pi run aarch64 Linux, where the x86_64 OpenCV / ffmpeg / PyTorch natives
// won't load at all (no webcam, no .pt). Resolve the right native classifier from
// the build host so `:desktop:run` / packaging on an ARM board pulls ARM binaries.
val osArch: String = System.getProperty("os.arch").lowercase()
val isArm64   = osArch.contains("aarch64") || osArch.contains("arm64")
val isArm32   = !isArm64 && (osArch.contains("arm") || osArch.startsWith("aarch"))

// JavaCV / bytedeco platform classifier (opencv, ffmpeg).
val bytedecoPlatform: String = when {
    isLinux && isArm64   -> "linux-arm64"
    isLinux && isArm32   -> "linux-armhf"
    isLinux              -> "linux-x86_64"
    isWindows            -> "windows-x86_64"
    isMac && isArm64     -> "macosx-arm64"
    isMac                -> "macosx-x86_64"
    else                 -> "linux-x86_64"
}
// DJL PyTorch CPU native dependency for the host platform. NOTE: on aarch64 Linux
// DJL ships PyTorch only under the `-precxx11` module (the plain pytorch-native-cpu
// has no linux-aarch64 classifier), so single-board boards need that artifact —
// using the wrong one fails dependency resolution before the build even starts.
val djlTorchDep: String = when {
    isLinux && isArm64 -> "ai.djl.pytorch:pytorch-native-cpu-precxx11:2.1.1:linux-aarch64"
    isLinux            -> "ai.djl.pytorch:pytorch-native-cpu:2.1.1:linux-x86_64"
    isWindows          -> "ai.djl.pytorch:pytorch-native-cpu:2.1.1:win-x86_64"
    isMac && isArm64   -> "ai.djl.pytorch:pytorch-native-cpu:2.1.1:osx-aarch64"
    isMac              -> "ai.djl.pytorch:pytorch-native-cpu:2.1.1:osx-x86_64"
    else               -> "ai.djl.pytorch:pytorch-native-cpu:2.1.1:linux-x86_64"
}
// CUDA only exists on x86_64 Linux/Windows — never on ARM single-board computers.
val cudaSupported = (isLinux || isWindows) && !isArm64 && !isArm32

dependencies {
    implementation(compose.desktop.currentOs)
    implementation(compose.material3)
    implementation(compose.materialIconsExtended)

    // ── ONNX Runtime ───────────────────────────────────────────────────────────
    // The standard jar already includes CUDA EP + DirectML EP bindings.
    // GPU providers activate at runtime when system CUDA/DirectML libs are present;
    // fall back to CPU automatically when they are not. No separate -gpu artifact exists.
    implementation("com.microsoft.onnxruntime:onnxruntime:1.18.0")

    // ── DJL PyTorch (.pt / TorchScript models) ─────────────────────────────────
    implementation("ai.djl:api:0.27.0")
    implementation("ai.djl.pytorch:pytorch-engine:0.27.0")

    // CPU-only native for the host architecture (always-available fallback,
    // including aarch64 boards — see djlTorchDep for the precxx11 caveat).
    runtimeOnly(djlTorchDep)
    // CUDA 12.1 native (NVIDIA GPU — x86_64 only; skipped on ARM boards).
    if (cudaSupported) {
        val cudaPlatform = if (isWindows) "win-x86_64" else "linux-x86_64"
        runtimeOnly("ai.djl.pytorch:pytorch-native-cu121:2.1.1:$cudaPlatform")
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
    runtimeOnly("org.bytedeco:opencv:4.9.0-1.5.10:$bytedecoPlatform")
    runtimeOnly("org.bytedeco:ffmpeg:6.1.1-1.5.10:$bytedecoPlatform")

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

            // No custom icon shipped — jpackage uses its platform default.
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

// ── Headless runner ─────────────────────────────────────────────────────────
// For single-board computers (Raspberry Pi, etc.) driven over SSH with no
// desktop: runs detection and broadcasts the annotated MJPEG stream on the LAN,
// no GUI window. Configure via YOLO_* environment variables (see Headless.kt).
//   ./gradlew :desktop:runHeadless
tasks.register<JavaExec>("runHeadless") {
    group = "application"
    description = "Run the detector headless (MJPEG broadcast only, no GUI)"
    dependsOn("classes")
    mainClass.set("com.destik.yolodesktop.HeadlessKt")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("-Xmx2g", "-Djava.awt.headless=true")
}
