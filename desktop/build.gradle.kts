import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("jvm")
    id("org.jetbrains.compose")
    id("org.jetbrains.kotlin.plugin.compose")
}

val osName: String = System.getProperty("os.name").lowercase()

dependencies {
    implementation(compose.desktop.currentOs)
    implementation(compose.material3)
    implementation(compose.materialIconsExtended)

    implementation("com.microsoft.onnxruntime:onnxruntime:1.18.0")
    implementation("com.google.code.gson:gson:2.10.1")

    // JavaCV for webcam (OpenCV grabber)
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
        osName.contains("linux") -> {
            implementation("org.bytedeco:opencv:4.9.0-1.5.10:linux-x86_64")
            implementation("org.bytedeco:ffmpeg:6.1.1-1.5.10:linux-x86_64")
        }
        osName.contains("windows") -> {
            implementation("org.bytedeco:opencv:4.9.0-1.5.10:windows-x86_64")
            implementation("org.bytedeco:ffmpeg:6.1.1-1.5.10:windows-x86_64")
        }
        osName.contains("mac") -> {
            implementation("org.bytedeco:opencv:4.9.0-1.5.10:macosx-arm64")
            implementation("org.bytedeco:ffmpeg:6.1.1-1.5.10:macosx-arm64")
        }
    }
}

kotlin {
    jvmToolchain(17)
}

compose.desktop {
    application {
        mainClass = "com.destik.yolodesktop.MainKt"
        nativeDistributions {
            targetFormats(TargetFormat.Deb, TargetFormat.Msi, TargetFormat.Dmg)
            packageName = "YoloDetector"
            packageVersion = "1.0.0"
            description = "Real-time YOLO object detection"
            vendor = "destik"

            linux {
                iconFile.set(project.file("src/main/resources/icon.png"))
            }
            windows {
                menuGroup = "YoloDetector"
                upgradeUuid = "4e3a2b1c-0f5d-4e7a-9b8c-2d3e4f5a6b7c"
            }
        }
        buildTypes.release.proguard {
            isEnabled.set(false)
        }
    }
}
