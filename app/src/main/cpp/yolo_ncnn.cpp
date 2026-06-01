#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "net.h"

#define TAG "YoloNCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct Object {
    float x, y, w, h;
    int label;
    float prob;
};

static ncnn::Net g_net;
static bool g_initialized = false;
static int g_num_classes = 80;
static int g_input_size = 640;
static int g_yolo_version = 8;
// Configurable output layer names
static std::string g_out0 = "output0";
static std::string g_out1 = "output1";
static std::string g_out2 = "output2";

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float iou_calc(const Object& a, const Object& b) {
    float ix1 = std::max(a.x, b.x);
    float iy1 = std::max(a.y, b.y);
    float ix2 = std::min(a.x + a.w, b.x + b.w);
    float iy2 = std::min(a.y + a.h, b.y + b.h);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f);
}

static void nms(std::vector<Object>& objs, float thresh) {
    std::sort(objs.begin(), objs.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });
    std::vector<bool> sup(objs.size(), false);
    for (size_t i = 0; i < objs.size(); i++) {
        if (sup[i]) continue;
        for (size_t j = i + 1; j < objs.size(); j++) {
            if (sup[j]) continue;
            if (objs[i].label == objs[j].label && iou_calc(objs[i], objs[j]) > thresh)
                sup[j] = true;
        }
    }
    std::vector<Object> res;
    for (size_t i = 0; i < objs.size(); i++)
        if (!sup[i]) res.push_back(objs[i]);
    objs = res;
}

// YOLOv8/v9/v10/v11 anchor-free: output shape [4+nc, num_boxes]
static void detect_v8(const ncnn::Mat& in, std::vector<Object>& objects,
                      float conf_thresh, float nms_thresh) {
    ncnn::Extractor ex = g_net.create_extractor();
    ex.input("images", in);

    ncnn::Mat out;
    if (ex.extract(g_out0.c_str(), out) != 0) {
        LOGE("extract %s failed", g_out0.c_str());
        return;
    }
    LOGD("v8 out: w=%d h=%d c=%d", out.w, out.h, out.c);

    int num_boxes = out.w;
    int num_attrs = out.h;

    for (int i = 0; i < num_boxes; i++) {
        float max_score = conf_thresh;
        int max_cls = -1;
        for (int c = 0; c < g_num_classes && (4 + c) < num_attrs; c++) {
            float s = out.row(4 + c)[i];
            if (s > max_score) { max_score = s; max_cls = c; }
        }
        if (max_cls < 0) continue;

        float cx = out.row(0)[i];
        float cy = out.row(1)[i];
        float bw = out.row(2)[i];
        float bh = out.row(3)[i];

        Object obj;
        obj.x = (cx - bw * 0.5f) / g_input_size;
        obj.y = (cy - bh * 0.5f) / g_input_size;
        obj.w = bw / g_input_size;
        obj.h = bh / g_input_size;
        obj.label = max_cls;
        obj.prob = max_score;
        objects.push_back(obj);
    }
    nms(objects, nms_thresh);
}

// YOLOv5/v6/v7 anchor-based
static const float ANCHORS[3][6] = {
    {10.f,13.f, 16.f,30.f, 33.f,23.f},
    {30.f,61.f, 62.f,45.f, 59.f,119.f},
    {116.f,90.f, 156.f,198.f, 373.f,326.f}
};

static void decode_v5_feat(const ncnn::Mat& feat, int stride, int scale_idx,
                           float conf_thresh, std::vector<Object>& objs) {
    int grid_h = feat.h;
    int grid_w = feat.w;
    const int na = 3;
    const int step = 5 + g_num_classes;

    for (int a = 0; a < na; a++) {
        float aw = ANCHORS[scale_idx][a * 2];
        float ah = ANCHORS[scale_idx][a * 2 + 1];
        for (int gy = 0; gy < grid_h; gy++) {
            for (int gx = 0; gx < grid_w; gx++) {
                float obj = sigmoid_f(feat.channel(a * step + 4).row(gy)[gx]);
                if (obj < conf_thresh) continue;

                float max_cls = 0;
                int max_c = 0;
                for (int c = 0; c < g_num_classes; c++) {
                    float cs = sigmoid_f(feat.channel(a * step + 5 + c).row(gy)[gx]) * obj;
                    if (cs > max_cls) { max_cls = cs; max_c = c; }
                }
                if (max_cls < conf_thresh) continue;

                float tx = feat.channel(a * step + 0).row(gy)[gx];
                float ty = feat.channel(a * step + 1).row(gy)[gx];
                float tw = feat.channel(a * step + 2).row(gy)[gx];
                float th = feat.channel(a * step + 3).row(gy)[gx];

                float bx = (sigmoid_f(tx) * 2.f - 0.5f + gx) * stride;
                float by = (sigmoid_f(ty) * 2.f - 0.5f + gy) * stride;
                float bw = powf(sigmoid_f(tw) * 2.f, 2.f) * aw;
                float bh = powf(sigmoid_f(th) * 2.f, 2.f) * ah;

                Object obj_det;
                obj_det.x = (bx - bw * 0.5f) / g_input_size;
                obj_det.y = (by - bh * 0.5f) / g_input_size;
                obj_det.w = bw / g_input_size;
                obj_det.h = bh / g_input_size;
                obj_det.label = max_c;
                obj_det.prob = max_cls;
                objs.push_back(obj_det);
            }
        }
    }
}

static void detect_v5(const ncnn::Mat& in, std::vector<Object>& objects,
                      float conf_thresh, float nms_thresh) {
    ncnn::Extractor ex = g_net.create_extractor();
    ex.input("images", in);

    const char* names[] = {g_out0.c_str(), g_out1.c_str(), g_out2.c_str()};
    const int strides[] = {8, 16, 32};
    for (int s = 0; s < 3; s++) {
        ncnn::Mat feat;
        if (ex.extract(names[s], feat) == 0) {
            LOGD("v5 out%d: w=%d h=%d c=%d", s, feat.w, feat.h, feat.c);
            decode_v5_feat(feat, strides[s], s, conf_thresh, objects);
        }
    }
    nms(objects, nms_thresh);
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeInit(
        JNIEnv* env, jobject,
        jstring param_path, jstring bin_path,
        jint version, jint input_size, jint num_classes,
        jboolean use_gpu,
        jstring out0, jstring out1, jstring out2) {

    if (g_initialized) { g_net.clear(); g_initialized = false; }

    g_yolo_version = (int)version;
    g_input_size = (int)input_size;
    g_num_classes = (int)num_classes;

    const char* p0 = env->GetStringUTFChars(out0, nullptr);
    const char* p1 = env->GetStringUTFChars(out1, nullptr);
    const char* p2 = env->GetStringUTFChars(out2, nullptr);
    g_out0 = p0; g_out1 = p1; g_out2 = p2;
    env->ReleaseStringUTFChars(out0, p0);
    env->ReleaseStringUTFChars(out1, p1);
    env->ReleaseStringUTFChars(out2, p2);

    g_net.opt.use_vulkan_compute = (bool)use_gpu;
    g_net.opt.use_fp16_packed = true;
    g_net.opt.use_fp16_storage = true;
    g_net.opt.use_fp16_arithmetic = true;

    const char* param = env->GetStringUTFChars(param_path, nullptr);
    const char* bin   = env->GetStringUTFChars(bin_path,   nullptr);

    int r = g_net.load_param(param);
    if (r != 0) { LOGE("load_param failed: %s", param); goto fail; }
    r = g_net.load_model(bin);
    if (r != 0) { LOGE("load_model failed: %s", bin); goto fail; }

    env->ReleaseStringUTFChars(param_path, param);
    env->ReleaseStringUTFChars(bin_path, bin);
    g_initialized = true;
    LOGD("Init OK: yolov%d input=%d classes=%d gpu=%d", version, input_size, num_classes, (int)use_gpu);
    return JNI_TRUE;

fail:
    env->ReleaseStringUTFChars(param_path, param);
    env->ReleaseStringUTFChars(bin_path, bin);
    return JNI_FALSE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeDetect(
        JNIEnv* env, jobject,
        jobject bitmap, jfloat conf_thresh, jfloat nms_thresh, jint num_threads) {

    jclass det_cls = env->FindClass("com/destik/yolodetector/Detection");
    jmethodID ctor = env->GetMethodID(det_cls, "<init>", "(FFFFIF)V");

    if (!g_initialized) return env->NewObjectArray(0, det_cls, nullptr);

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    void* pixels;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    int pixel_type = (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
        ? ncnn::Mat::PIXEL_RGBA2RGB
        : ncnn::Mat::PIXEL_RGB;

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        (const unsigned char*)pixels, pixel_type,
        info.width, info.height,
        g_input_size, g_input_size);

    AndroidBitmap_unlockPixels(env, bitmap);

    const float mean_vals[] = {0.f, 0.f, 0.f};
    const float norm_vals[] = {1/255.f, 1/255.f, 1/255.f};
    input.substract_mean_normalize(mean_vals, norm_vals);

    g_net.opt.num_threads = (int)num_threads;

    std::vector<Object> objects;
    if (g_yolo_version >= 8)
        detect_v8(input, objects, conf_thresh, nms_thresh);
    else
        detect_v5(input, objects, conf_thresh, nms_thresh);

    jobjectArray result = env->NewObjectArray((jsize)objects.size(), det_cls, nullptr);
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& o = objects[i];
        jobject det = env->NewObject(det_cls, ctor, o.x, o.y, o.w, o.h, o.label, o.prob);
        env->SetObjectArrayElement(result, (jsize)i, det);
        env->DeleteLocalRef(det);
    }
    return result;
}

JNIEXPORT void JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeRelease(JNIEnv*, jobject) {
    if (g_initialized) { g_net.clear(); g_initialized = false; }
}

} // extern "C"
