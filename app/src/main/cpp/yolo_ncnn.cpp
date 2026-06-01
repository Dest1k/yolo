#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "net.h"

#define TAG "YoloNCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct Object { float x,y,w,h; int label; float prob; };

static ncnn::Net  g_net;
static bool       g_initialized  = false;
static int        g_num_classes  = 80;
static int        g_input_size   = 640;
static int        g_yolo_version = 10;
static std::string g_param_path;
static std::string g_out0 = "out0";
static std::string g_out1 = "output1";
static std::string g_out2 = "output2";

static inline float sigmoid_f(float x){return 1.f/(1.f+expf(-x));}

static float iou_calc(const Object& a,const Object& b){
    float ix1=std::max(a.x,b.x),iy1=std::max(a.y,b.y);
    float ix2=std::min(a.x+a.w,b.x+b.w),iy2=std::min(a.y+a.h,b.y+b.h);
    float iw=std::max(0.f,ix2-ix1),ih=std::max(0.f,iy2-iy1),inter=iw*ih;
    return inter/(a.w*a.h+b.w*b.h-inter+1e-6f);
}
static void nms(std::vector<Object>& o,float t){
    std::sort(o.begin(),o.end(),[](const Object&a,const Object&b){return a.prob>b.prob;});
    std::vector<bool> s(o.size(),false);
    for(size_t i=0;i<o.size();i++){if(s[i])continue;
        for(size_t j=i+1;j<o.size();j++) if(!s[j]&&o[i].label==o[j].label&&iou_calc(o[i],o[j])>t) s[j]=true;}
    std::vector<Object> r; for(size_t i=0;i<o.size();i++) if(!s[i]) r.push_back(o[i]); o=r;
}

static std::vector<std::string> parse_output_names(const std::string& path){
    std::vector<std::string> result;
    if(path.empty()) return result;
    FILE* f=fopen(path.c_str(),"r");
    if(!f){LOGE("Cannot open param: %s",path.c_str());return result;}
    int magic=0; if(fscanf(f,"%d",&magic)!=1||magic!=7767517){fclose(f);return result;}
    int nl=0,nb=0; fscanf(f,"%d %d",&nl,&nb);
    std::vector<std::string> top_order;
    std::set<std::string> all_tops,all_bottoms;
    char ltype[256],lname[256]; int bc,tc;
    for(int i=0;i<nl;i++){
        if(fscanf(f,"%255s %255s %d %d",ltype,lname,&bc,&tc)!=4) break;
        for(int j=0;j<bc;j++){char b[256]={};if(fscanf(f,"%255s",b)==1) all_bottoms.insert(b);}
        for(int j=0;j<tc;j++){char t[256]={};if(fscanf(f,"%255s",t)==1){all_tops.insert(t);top_order.push_back(t);}}
        char line[4096]={}; fgets(line,sizeof(line),f);
    }
    fclose(f);
    std::set<std::string> seen;
    for(auto& t:top_order){
        if(seen.count(t)) continue; seen.insert(t);
        if(!all_bottoms.count(t)) result.push_back(t);
    }
    for(auto& n:result) LOGD("output blob: %s",n.c_str());
    return result;
}

// Safe accessor — returns 0 if out of bounds
static inline float safe_get(const ncnn::Mat& m, int row, int col){
    if(row<0||row>=m.h||col<0||col>=m.w) return 0.f;
    return m.row(row)[col];
}

// ── YOLOv10/v11 NMS-free ───────────────────────────────────────────────────────
static void detect_v10(const ncnn::Mat& in,std::vector<Object>& objects,float ct){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input("images",in);
    ncnn::Mat out;
    if(ex.extract(g_out0.c_str(),out)!=0){
        LOGE("v10: extract '%s' failed",g_out0.c_str()); return;
    }
    LOGD("v10 out: w=%d h=%d c=%d",out.w,out.h,out.c);
    if(out.w==0||out.h==0){ LOGE("v10: empty output"); return; }

    // Determine layout:
    // Case A: [N, 6]  → h=N w=6  (or h=N w>6 for some exports)
    // Case B: [6, N]  → h=6 w=N  (transposed)
    // Case C: [1, N, 6] → c=1 h=N w=6 (batched)
    bool tr = (out.h == 6 && out.w > 6); // transposed [6, N]
    int nd = tr ? out.w : out.h;
    int na = tr ? out.h : out.w; // should be >=6
    LOGD("v10 transposed=%d nd=%d na=%d",tr,nd,na);
    if(na < 6){ LOGE("v10: unexpected attr count %d",na); return; }

    // detect coord scale: pixel (0..input_size) or normalised (0..1)
    bool pixel = true;
    for(int i=0;i<std::min(nd,20);i++){
        float score = tr ? safe_get(out,4,i) : safe_get(out,i,4);
        if(score < ct) continue;
        float x2 = tr ? safe_get(out,2,i) : safe_get(out,i,2);
        if(x2 > 0.f && x2 <= 1.5f){ pixel=false; }
        break;
    }
    float sc = pixel ? (1.f/g_input_size) : 1.f;
    LOGD("v10 pixel=%d sc=%f",pixel,sc);

    for(int i=0;i<nd;i++){
        float x1    = tr?safe_get(out,0,i):safe_get(out,i,0);
        float y1    = tr?safe_get(out,1,i):safe_get(out,i,1);
        float x2    = tr?safe_get(out,2,i):safe_get(out,i,2);
        float y2    = tr?safe_get(out,3,i):safe_get(out,i,3);
        float score = tr?safe_get(out,4,i):safe_get(out,i,4);
        float cid   = tr?safe_get(out,5,i):safe_get(out,i,5);
        if(score<ct||x2<=x1||y2<=y1) continue;
        Object o;
        o.x=x1*sc; o.y=y1*sc; o.w=(x2-x1)*sc; o.h=(y2-y1)*sc;
        o.label=(int)cid; o.prob=score;
        objects.push_back(o);
    }
    LOGD("v10 detected %d objects",(int)objects.size());
}

// ── YOLOv8/v9 anchor-free ──────────────────────────────────────────────────────
static void detect_v8(const ncnn::Mat& in,std::vector<Object>& objects,float ct,float nt){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input("images",in);
    ncnn::Mat out;
    if(ex.extract(g_out0.c_str(),out)!=0){LOGE("v8: extract '%s' failed",g_out0.c_str());return;}
    LOGD("v8 out: w=%d h=%d c=%d",out.w,out.h,out.c);
    if(out.w==0||out.h==0) return;
    bool t=(out.h==4+g_num_classes);
    int nb=t?out.w:out.h, na=t?out.h:out.w;
    for(int i=0;i<nb;i++){
        float ms=ct; int mc=-1;
        for(int c=0;c<g_num_classes&&(4+c)<na;c++){
            float s=t?safe_get(out,4+c,i):safe_get(out,i,4+c);
            if(s>ms){ms=s;mc=c;}
        }
        if(mc<0) continue;
        float cx=t?safe_get(out,0,i):safe_get(out,i,0);
        float cy=t?safe_get(out,1,i):safe_get(out,i,1);
        float bw=t?safe_get(out,2,i):safe_get(out,i,2);
        float bh=t?safe_get(out,3,i):safe_get(out,i,3);
        Object o; o.x=(cx-bw*.5f)/g_input_size; o.y=(cy-bh*.5f)/g_input_size;
        o.w=bw/g_input_size; o.h=bh/g_input_size; o.label=mc; o.prob=ms;
        objects.push_back(o);
    }
    nms(objects,nt);
}

// ── YOLOv5/v6/v7 anchor-based ───────────────────────────────────────────────
static const float ANCHORS[3][6]={{10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326}};
static void decode_v5(const ncnn::Mat& f,int st,int si,float ct,std::vector<Object>& o){
    int gh=f.h,gw=f.w; if(gh==0||gw==0) return;
    const int na=3,step=5+g_num_classes;
    if(f.c < na*step){ LOGE("v5: too few channels %d",f.c); return; }
    for(int a=0;a<na;a++){
        float aw=ANCHORS[si][a*2],ah=ANCHORS[si][a*2+1];
        for(int gy=0;gy<gh;gy++) for(int gx=0;gx<gw;gx++){
            float obj=sigmoid_f(f.channel(a*step+4).row(gy)[gx]); if(obj<ct) continue;
            float mc=0; int mi=0;
            for(int c=0;c<g_num_classes;c++){float s=sigmoid_f(f.channel(a*step+5+c).row(gy)[gx])*obj;if(s>mc){mc=s;mi=c;}}
            if(mc<ct) continue;
            float bx=(sigmoid_f(f.channel(a*step).row(gy)[gx])*2-.5f+gx)*st;
            float by=(sigmoid_f(f.channel(a*step+1).row(gy)[gx])*2-.5f+gy)*st;
            float bw=powf(sigmoid_f(f.channel(a*step+2).row(gy)[gx])*2,2)*aw;
            float bh=powf(sigmoid_f(f.channel(a*step+3).row(gy)[gx])*2,2)*ah;
            Object ob; ob.x=(bx-bw*.5f)/g_input_size; ob.y=(by-bh*.5f)/g_input_size;
            ob.w=bw/g_input_size; ob.h=bh/g_input_size; ob.label=mi; ob.prob=mc; o.push_back(ob);
        }
    }
}
static void detect_v5(const ncnn::Mat& in,std::vector<Object>& o,float ct,float nt){
    ncnn::Extractor ex=g_net.create_extractor(); ex.input("images",in);
    const char* n[]={g_out0.c_str(),g_out1.c_str(),g_out2.c_str()}; const int st[]={8,16,32};
    for(int s=0;s<3;s++){ncnn::Mat f;if(ex.extract(n[s],f)==0) decode_v5(f,st[s],s,ct,o);}
    nms(o,nt);
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeInit(
        JNIEnv* env,jobject,
        jstring pp,jstring bp,jint ver,jint isz,jint nc,jboolean gpu,
        jstring o0,jstring o1,jstring o2){
    if(g_initialized){g_net.clear();g_initialized=false;}
    g_yolo_version=(int)ver; g_input_size=(int)isz; g_num_classes=(int)nc;
    const char* c0=env->GetStringUTFChars(o0,0); const char* c1=env->GetStringUTFChars(o1,0); const char* c2=env->GetStringUTFChars(o2,0);
    g_out0=c0; g_out1=c1; g_out2=c2;
    env->ReleaseStringUTFChars(o0,c0); env->ReleaseStringUTFChars(o1,c1); env->ReleaseStringUTFChars(o2,c2);
    const char* param=env->GetStringUTFChars(pp,0); const char* bin=env->GetStringUTFChars(bp,0);
    g_param_path=param;
    g_net.opt.use_vulkan_compute=(bool)gpu;
    g_net.opt.use_fp16_packed=g_net.opt.use_fp16_storage=g_net.opt.use_fp16_arithmetic=true;
    int r=g_net.load_param(param);
    if(r!=0){LOGE("load_param failed");env->ReleaseStringUTFChars(pp,param);env->ReleaseStringUTFChars(bp,bin);return JNI_FALSE;}
    r=g_net.load_model(bin);
    if(r!=0){LOGE("load_model failed"); env->ReleaseStringUTFChars(pp,param);env->ReleaseStringUTFChars(bp,bin);return JNI_FALSE;}
    env->ReleaseStringUTFChars(pp,param); env->ReleaseStringUTFChars(bp,bin);
    g_initialized=true;
    LOGD("Init OK yolov%d input=%d nc=%d gpu=%d out0=%s",ver,isz,nc,(int)gpu,g_out0.c_str());
    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeGetOutputNames(JNIEnv* env,jobject){
    auto names=parse_output_names(g_param_path);
    jclass sc=env->FindClass("java/lang/String");
    jobjectArray arr=env->NewObjectArray((jsize)names.size(),sc,nullptr);
    for(size_t i=0;i<names.size();i++) env->SetObjectArrayElement(arr,(jsize)i,env->NewStringUTF(names[i].c_str()));
    return arr;
}

JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeDetect(
        JNIEnv* env,jobject,jobject bitmap,jfloat ct,jfloat nt,jint nth){
    jclass dc=env->FindClass("com/destik/yolodetector/Detection");
    jmethodID ctor=env->GetMethodID(dc,"<init>","(FFFFIF)V");
    if(!g_initialized) return env->NewObjectArray(0,dc,nullptr);

    AndroidBitmapInfo info;
    if(AndroidBitmap_getInfo(env,bitmap,&info)!=ANDROID_BITMAP_RESULT_SUCCESS)
        return env->NewObjectArray(0,dc,nullptr);
    void* px=nullptr;
    if(AndroidBitmap_lockPixels(env,bitmap,&px)!=ANDROID_BITMAP_RESULT_SUCCESS)
        return env->NewObjectArray(0,dc,nullptr);

    ncnn::Mat in;
    if(info.format==ANDROID_BITMAP_FORMAT_RGBA_8888){
        in=ncnn::Mat::from_pixels_resize((const unsigned char*)px,ncnn::Mat::PIXEL_RGBA2RGB,
            (int)info.width,(int)info.height,g_input_size,g_input_size);
    } else if(info.format==ANDROID_BITMAP_FORMAT_RGB_565){
        in=ncnn::Mat::from_pixels_resize((const unsigned char*)px,ncnn::Mat::PIXEL_RGB565toRGB,
            (int)info.width,(int)info.height,g_input_size,g_input_size);
    } else {
        LOGE("unsupported bitmap format %d",info.format);
        AndroidBitmap_unlockPixels(env,bitmap);
        return env->NewObjectArray(0,dc,nullptr);
    }
    AndroidBitmap_unlockPixels(env,bitmap);

    if(in.empty()){LOGE("from_pixels_resize returned empty mat");return env->NewObjectArray(0,dc,nullptr);}

    const float mv[]={0,0,0},nv[]={1/255.f,1/255.f,1/255.f};
    in.substract_mean_normalize(mv,nv);
    g_net.opt.num_threads=(int)nth;

    std::vector<Object> objs;
    if     (g_yolo_version>=10) detect_v10(in,objs,ct);
    else if(g_yolo_version>= 8) detect_v8 (in,objs,ct,nt);
    else                        detect_v5 (in,objs,ct,nt);

    jobjectArray res=env->NewObjectArray((jsize)objs.size(),dc,nullptr);
    for(size_t i=0;i<objs.size();i++){
        const Object& o=objs[i];
        jobject d=env->NewObject(dc,ctor,o.x,o.y,o.w,o.h,o.label,o.prob);
        env->SetObjectArrayElement(res,(jsize)i,d); env->DeleteLocalRef(d);
    }
    return res;
}

JNIEXPORT void JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeRelease(JNIEnv*,jobject){
    if(g_initialized){g_net.clear();g_initialized=false;}
}

} // extern "C"
