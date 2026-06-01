#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

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
static bool      g_initialized  = false;
static int       g_num_classes  = 80;
static int       g_input_size   = 640;
static int       g_yolo_version = 8;
static std::string g_param_path;
static std::string g_out0 = "output0";
static std::string g_out1 = "output1";
static std::string g_out2 = "output2";

// ── helpers ─────────────────────────────────────────────────────────────────
static inline float sigmoid_f(float x) { return 1.f/(1.f+expf(-x)); }

static float iou_calc(const Object& a,const Object& b){
    float ix1=std::max(a.x,b.x),iy1=std::max(a.y,b.y);
    float ix2=std::min(a.x+a.w,b.x+b.w),iy2=std::min(a.y+a.h,b.y+b.h);
    float iw=std::max(0.f,ix2-ix1),ih=std::max(0.f,iy2-iy1);
    float inter=iw*ih;
    return inter/(a.w*a.h+b.w*b.h-inter+1e-6f);
}
static void nms(std::vector<Object>& objs,float thresh){
    std::sort(objs.begin(),objs.end(),[](const Object&a,const Object&b){return a.prob>b.prob;});
    std::vector<bool> sup(objs.size(),false);
    for(size_t i=0;i<objs.size();i++){
        if(sup[i])continue;
        for(size_t j=i+1;j<objs.size();j++){
            if(!sup[j]&&objs[i].label==objs[j].label&&iou_calc(objs[i],objs[j])>thresh) sup[j]=true;
        }
    }
    std::vector<Object> res;
    for(size_t i=0;i<objs.size();i++) if(!sup[i]) res.push_back(objs[i]);
    objs=res;
}

// ── parse .param → output blob names ────────────────────────────────────────
// Returns blobs that are produced but never consumed (= network outputs)
static std::vector<std::string> parse_output_names(const std::string& param_path){
    std::vector<std::string> result;
    FILE* f=fopen(param_path.c_str(),"r");
    if(!f) return result;

    int magic=0; fscanf(f,"%d",&magic);
    if(magic!=7767517){fclose(f);return result;}

    int num_layers=0,num_blobs=0;
    fscanf(f,"%d %d",&num_layers,&num_blobs);

    // ordered list of tops (preserves encounter order for output ordering)
    std::vector<std::string> top_order;
    std::set<std::string> all_tops,all_bottoms;

    for(int i=0;i<num_layers;i++){
        char ltype[256]={},lname[256]={};
        int nb=0,nt=0;
        if(fscanf(f,"%255s %255s %d %d",ltype,lname,&nb,&nt)!=4) break;
        for(int j=0;j<nb;j++){char b[256]={};fscanf(f,"%255s",b);all_bottoms.insert(b);}
        for(int j=0;j<nt;j++){char t[256]={};fscanf(f,"%255s",t);all_tops.insert(t);top_order.push_back(t);}
        // skip rest of line (params)
        char line[8192]={}; fgets(line,sizeof(line),f);
    }
    fclose(f);

    // outputs = tops not consumed as bottoms, in encounter order
    std::set<std::string> seen;
    for(auto& t:top_order){
        if(seen.count(t)) continue; seen.insert(t);
        if(!all_bottoms.count(t)) result.push_back(t);
    }
    return result;
}

// ── probe: run dummy inference, return "name:w×h×c" for each output ──────────
static std::string probe_outputs(const std::vector<std::string>& names){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.set_light_mode(true);
    // dummy 1×1 black image
    ncnn::Mat dummy(g_input_size,g_input_size,3);
    dummy.fill(0.f);
    ex.input("images",dummy);

    std::string info;
    for(auto& name:names){
        ncnn::Mat out;
        if(ex.extract(name.c_str(),out)==0){
            char buf[256];
            snprintf(buf,sizeof(buf),"%s: w=%d h=%d c=%d\n",name.c_str(),out.w,out.h,out.c);
            info+=buf;
            LOGD("probe %s: w=%d h=%d c=%d",name.c_str(),out.w,out.h,out.c);
        } else {
            info+=name+": extract failed\n";
        }
    }
    return info;
}

// ── YOLOv8/v9 anchor-free ────────────────────────────────────────────────────
static void detect_v8(const ncnn::Mat& in,std::vector<Object>& objects,
                      float conf_thresh,float nms_thresh){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input("images",in);
    ncnn::Mat out;
    if(ex.extract(g_out0.c_str(),out)!=0){LOGE("v8: extract '%s' failed",g_out0.c_str());return;}
    LOGD("v8 out: w=%d h=%d c=%d",out.w,out.h,out.c);

    // auto-detect orientation: h==4+nc → rows are attributes
    bool t=(out.h==4+g_num_classes);
    int num_boxes=t?out.w:out.h, num_attrs=t?out.h:out.w;

    for(int i=0;i<num_boxes;i++){
        float max_score=conf_thresh; int max_cls=-1;
        for(int c=0;c<g_num_classes&&(4+c)<num_attrs;c++){
            float s=t?out.row(4+c)[i]:out.row(i)[4+c];
            if(s>max_score){max_score=s;max_cls=c;}
        }
        if(max_cls<0) continue;
        float cx=t?out.row(0)[i]:out.row(i)[0];
        float cy=t?out.row(1)[i]:out.row(i)[1];
        float bw=t?out.row(2)[i]:out.row(i)[2];
        float bh=t?out.row(3)[i]:out.row(i)[3];
        Object o;
        o.x=(cx-bw*.5f)/g_input_size; o.y=(cy-bh*.5f)/g_input_size;
        o.w=bw/g_input_size;          o.h=bh/g_input_size;
        o.label=max_cls; o.prob=max_score;
        objects.push_back(o);
    }
    nms(objects,nms_thresh);
}

// ── YOLOv10/v11 NMS-free ─────────────────────────────────────────────────────
// Expected shapes (NCNN): [num_dets, 6]  or  [6, num_dets]
// Each detection: [x1,y1,x2,y2,score,class_id]  (pixel or 0-1 coords auto-detected)
static void detect_v10(const ncnn::Mat& in,std::vector<Object>& objects,float conf_thresh){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input("images",in);
    ncnn::Mat out;
    if(ex.extract(g_out0.c_str(),out)!=0){LOGE("v10: extract '%s' failed",g_out0.c_str());return;}
    LOGD("v10 out: w=%d h=%d c=%d",out.w,out.h,out.c);

    // orientation: if h==6 → transposed [6,N], else [N,6]
    bool t=(out.h==6);
    int num_dets=t?out.w:out.h;

    // peek first row to detect normalised vs pixel coords
    bool pixel_coords=true;
    if(num_dets>0){
        float x2=t?out.row(2)[0]:out.row(0)[2];
        float y2=t?out.row(3)[0]:out.row(0)[3];
        if(x2<=1.01f&&y2<=1.01f) pixel_coords=false; // already 0-1
    }
    float scale=pixel_coords?(1.f/g_input_size):1.f;
    LOGD("v10 pixel_coords=%d scale=%f",pixel_coords,scale);

    for(int i=0;i<num_dets;i++){
        float x1,y1,x2,y2,score,cls_id;
        if(t){x1=out.row(0)[i];y1=out.row(1)[i];x2=out.row(2)[i];y2=out.row(3)[i];score=out.row(4)[i];cls_id=out.row(5)[i];}
        else {x1=out.row(i)[0];y1=out.row(i)[1];x2=out.row(i)[2];y2=out.row(i)[3];score=out.row(i)[4];cls_id=out.row(i)[5];}
        if(score<conf_thresh||x2<=x1||y2<=y1) continue;
        Object o;
        o.x=x1*scale; o.y=y1*scale;
        o.w=(x2-x1)*scale; o.h=(y2-y1)*scale;
        o.label=(int)cls_id; o.prob=score;
        objects.push_back(o);
    }
}

// ── YOLOv5/v6/v7 anchor-based ────────────────────────────────────────────────
static const float ANCHORS[3][6]={{10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326}};
static void decode_v5_feat(const ncnn::Mat& feat,int stride,int si,float ct,std::vector<Object>& objs){
    int gh=feat.h,gw=feat.w; const int na=3,step=5+g_num_classes;
    for(int a=0;a<na;a++){
        float aw=ANCHORS[si][a*2],ah=ANCHORS[si][a*2+1];
        for(int gy=0;gy<gh;gy++) for(int gx=0;gx<gw;gx++){
            float obj=sigmoid_f(feat.channel(a*step+4).row(gy)[gx]);
            if(obj<ct) continue;
            float mc=0; int mi=0;
            for(int c=0;c<g_num_classes;c++){float cs=sigmoid_f(feat.channel(a*step+5+c).row(gy)[gx])*obj;if(cs>mc){mc=cs;mi=c;}}
            if(mc<ct) continue;
            float bx=(sigmoid_f(feat.channel(a*step+0).row(gy)[gx])*2-.5f+gx)*stride;
            float by=(sigmoid_f(feat.channel(a*step+1).row(gy)[gx])*2-.5f+gy)*stride;
            float bw=powf(sigmoid_f(feat.channel(a*step+2).row(gy)[gx])*2,2)*aw;
            float bh=powf(sigmoid_f(feat.channel(a*step+3).row(gy)[gx])*2,2)*ah;
            Object o;o.x=(bx-bw*.5f)/g_input_size;o.y=(by-bh*.5f)/g_input_size;
            o.w=bw/g_input_size;o.h=bh/g_input_size;o.label=mi;o.prob=mc;objs.push_back(o);
        }
    }
}
static void detect_v5(const ncnn::Mat& in,std::vector<Object>& objs,float ct,float nt){
    ncnn::Extractor ex=g_net.create_extractor(); ex.input("images",in);
    const char* n[]={g_out0.c_str(),g_out1.c_str(),g_out2.c_str()};
    const int st[]={8,16,32};
    for(int s=0;s<3;s++){ncnn::Mat f;if(ex.extract(n[s],f)==0) decode_v5_feat(f,st[s],s,ct,objs);}
    nms(objs,nt);
}

// ── JNI ──────────────────────────────────────────────────────────────────────
extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeInit(
        JNIEnv* env,jobject,
        jstring param_path,jstring bin_path,
        jint version,jint input_size,jint num_classes,
        jboolean use_gpu,
        jstring out0,jstring out1,jstring out2){

    if(g_initialized){g_net.clear();g_initialized=false;}
    g_yolo_version=(int)version; g_input_size=(int)input_size; g_num_classes=(int)num_classes;

    const char* p0=env->GetStringUTFChars(out0,0); const char* p1=env->GetStringUTFChars(out1,0); const char* p2=env->GetStringUTFChars(out2,0);
    g_out0=p0; g_out1=p1; g_out2=p2;
    env->ReleaseStringUTFChars(out0,p0); env->ReleaseStringUTFChars(out1,p1); env->ReleaseStringUTFChars(out2,p2);

    const char* param=env->GetStringUTFChars(param_path,0);
    const char* bin  =env->GetStringUTFChars(bin_path,0);
    g_param_path=param;

    g_net.opt.use_vulkan_compute=(bool)use_gpu;
    g_net.opt.use_fp16_packed=g_net.opt.use_fp16_storage=g_net.opt.use_fp16_arithmetic=true;

    int r=g_net.load_param(param); if(r!=0){LOGE("load_param failed"); goto fail;}
    r=g_net.load_model(bin);        if(r!=0){LOGE("load_model failed");  goto fail;}

    env->ReleaseStringUTFChars(param_path,param); env->ReleaseStringUTFChars(bin_path,bin);
    g_initialized=true;
    LOGD("Init OK: yolov%d input=%d classes=%d gpu=%d out0=%s",version,input_size,num_classes,(int)use_gpu,g_out0.c_str());
    return JNI_TRUE;
fail:
    env->ReleaseStringUTFChars(param_path,param); env->ReleaseStringUTFChars(bin_path,bin);
    return JNI_FALSE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeGetOutputNames(JNIEnv* env,jobject){
    auto names=parse_output_names(g_param_path);
    // also run probe to log shapes
    if(g_initialized&&!names.empty()) probe_outputs(names);
    jclass sc=env->FindClass("java/lang/String");
    jobjectArray arr=env->NewObjectArray((jsize)names.size(),sc,nullptr);
    for(size_t i=0;i<names.size();i++) env->SetObjectArrayElement(arr,(jsize)i,env->NewStringUTF(names[i].c_str()));
    return arr;
}

JNIEXPORT jstring JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeProbeOutputs(JNIEnv* env,jobject){
    if(!g_initialized) return env->NewStringUTF("Model not loaded");
    auto names=parse_output_names(g_param_path);
    std::string info=probe_outputs(names);
    if(info.empty()) info="No outputs detected";
    return env->NewStringUTF(info.c_str());
}

JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeDetect(
        JNIEnv* env,jobject,
        jobject bitmap,jfloat conf_thresh,jfloat nms_thresh,jint num_threads){

    jclass dc=env->FindClass("com/destik/yolodetector/Detection");
    jmethodID ctor=env->GetMethodID(dc,"<init>","(FFFFIF)V");
    if(!g_initialized) return env->NewObjectArray(0,dc,nullptr);

    AndroidBitmapInfo info; AndroidBitmap_getInfo(env,bitmap,&info);
    void* pixels; AndroidBitmap_lockPixels(env,bitmap,&pixels);
    int pt=(info.format==ANDROID_BITMAP_FORMAT_RGBA_8888)?ncnn::Mat::PIXEL_RGBA2RGB:ncnn::Mat::PIXEL_RGB;
    ncnn::Mat input=ncnn::Mat::from_pixels_resize((const unsigned char*)pixels,pt,info.width,info.height,g_input_size,g_input_size);
    AndroidBitmap_unlockPixels(env,bitmap);

    const float mv[]={0,0,0}, nv[]={1/255.f,1/255.f,1/255.f};
    input.substract_mean_normalize(mv,nv);
    g_net.opt.num_threads=(int)num_threads;

    std::vector<Object> objects;
    if     (g_yolo_version>=10) detect_v10(input,objects,conf_thresh);
    else if(g_yolo_version>= 8) detect_v8 (input,objects,conf_thresh,nms_thresh);
    else                        detect_v5 (input,objects,conf_thresh,nms_thresh);

    jobjectArray res=env->NewObjectArray((jsize)objects.size(),dc,nullptr);
    for(size_t i=0;i<objects.size();i++){
        const Object& o=objects[i];
        jobject det=env->NewObject(dc,ctor,o.x,o.y,o.w,o.h,o.label,o.prob);
        env->SetObjectArrayElement(res,(jsize)i,det); env->DeleteLocalRef(det);
    }
    return res;
}

JNIEXPORT void JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeRelease(JNIEnv*,jobject){
    if(g_initialized){g_net.clear();g_initialized=false;}
}

} // extern "C"
