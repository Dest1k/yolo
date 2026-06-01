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
static std::string g_input_name = "images";
static std::string g_out0 = "out0";
static std::string g_out1 = "output1";
static std::string g_out2 = "output2";

static inline float sigmoid_f(float x){ return 1.f/(1.f+expf(-x)); }
static inline bool  is_bad(float v)   { return std::isnan(v)||std::isinf(v); }

static float iou_calc(const Object& a, const Object& b){
    float ix1=std::max(a.x,b.x), iy1=std::max(a.y,b.y);
    float ix2=std::min(a.x+a.w,b.x+b.w), iy2=std::min(a.y+a.h,b.y+b.h);
    float iw=std::max(0.f,ix2-ix1), ih=std::max(0.f,iy2-iy1), inter=iw*ih;
    return inter/(a.w*a.h+b.w*b.h-inter+1e-6f);
}
static void nms(std::vector<Object>& o, float t){
    std::sort(o.begin(),o.end(),[](const Object&a,const Object&b){return a.prob>b.prob;});
    std::vector<bool> s(o.size(),false);
    for(size_t i=0;i<o.size();i++){
        if(s[i]) continue;
        for(size_t j=i+1;j<o.size();j++)
            if(!s[j]&&o[i].label==o[j].label&&iou_calc(o[i],o[j])>t) s[j]=true;
    }
    std::vector<Object> r;
    for(size_t i=0;i<o.size();i++) if(!s[i]) r.push_back(o[i]);
    o=r;
}

struct ParamInfo {
    std::string input_name;
    std::vector<std::string> output_names;
};

static ParamInfo parse_param(const std::string& path){
    ParamInfo info;
    info.input_name="images";
    if(path.empty()) return info;
    FILE* f=fopen(path.c_str(),"r");
    if(!f){ LOGE("Cannot open param: %s",path.c_str()); return info; }
    int magic=0;
    if(fscanf(f,"%d",&magic)!=1||magic!=7767517){ fclose(f); return info; }
    int nl=0,nb=0; fscanf(f,"%d %d",&nl,&nb);
    std::vector<std::string> top_order;
    std::set<std::string> all_tops, all_bottoms;
    char ltype[256], lname[256]; int bc, tc;
    for(int i=0;i<nl;i++){
        if(fscanf(f,"%255s %255s %d %d",ltype,lname,&bc,&tc)!=4) break;
        std::vector<std::string> tops;
        for(int j=0;j<bc;j++){ char b[256]={}; if(fscanf(f,"%255s",b)==1) all_bottoms.insert(b); }
        for(int j=0;j<tc;j++){ char t[256]={}; if(fscanf(f,"%255s",t)==1){ tops.push_back(t); all_tops.insert(t); top_order.push_back(t); } }
        if(strcasecmp(ltype,"Input")==0&&!tops.empty()) info.input_name=tops[0];
        char line[4096]={}; fgets(line,sizeof(line),f);
    }
    fclose(f);
    std::set<std::string> seen;
    for(auto& t:top_order){
        if(seen.count(t)) continue; seen.insert(t);
        if(!all_bottoms.count(t)) info.output_names.push_back(t);
    }
    LOGD("input=%s outputs=%d",info.input_name.c_str(),(int)info.output_names.size());
    return info;
}

// Flatten a 3-D NCNN Mat (c,h,w) into a 2-D (c*h, w) view so row(i) works uniformly.
// For 2-D mats (c==1) this is a no-op.
static ncnn::Mat squeeze2d(const ncnn::Mat& m){
    if(m.c<=1) return m;
    // Reinterpret: rows = c*h, cols = w
    ncnn::Mat out(m.w, m.c*m.h);
    for(int ci=0;ci<m.c;ci++)
        memcpy((float*)out+ci*m.h*m.w, m.channel(ci), m.h*m.w*sizeof(float));
    return out;
}

static inline float safe_get(const ncnn::Mat& m, int row, int col){
    if(row<0||row>=m.h||col<0||col>=m.w) return 0.f;
    return m.row(row)[col];
}

// ── YOLOv10/v11 NMS-free ──────────────────────────────────────────────────────
static void detect_v10(const ncnn::Mat& in, std::vector<Object>& objects, float ct){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input(g_input_name.c_str(),in);
    ncnn::Mat raw;
    if(ex.extract(g_out0.c_str(),raw)!=0){ LOGE("v10: extract '%s' failed",g_out0.c_str()); return; }
    ncnn::Mat out=squeeze2d(raw);
    LOGD("v10 out: w=%d h=%d c=%d",out.w,out.h,out.c);
    if(out.w==0||out.h==0){ LOGE("v10: empty output"); return; }

    // Layout: [N,6] h=N w=6  OR transposed [6,N] h=6 w=N.
    // Use w==6 as the definitive check (more robust than h==6 for small N).
    bool tr=(out.w==6&&out.h>6) ? false   // normal [N,6]
           :(out.h==6&&out.w>6) ? true    // transposed [6,N]
           :(out.w>=6)          ? false   // w>=6 → treat as [N, w]
           :true;                         // fallback
    int nd=tr?out.w:out.h;
    int na=tr?out.h:out.w;
    LOGD("v10 tr=%d nd=%d na=%d",tr,nd,na);
    if(na<6){ LOGE("v10: bad attr count %d (need >=6)",na); return; }

    // Determine pixel vs normalised coords by scanning ALL rows
    bool pixel=false;
    for(int i=0;i<nd;i++){
        float x2=tr?safe_get(out,2,i):safe_get(out,i,2);
        if(!is_bad(x2)&&x2>1.5f){ pixel=true; break; }
    }
    float sc=pixel?(1.f/g_input_size):1.f;
    LOGD("v10 pixel=%d sc=%f",pixel,sc);

    for(int i=0;i<nd;i++){
        float x1   =tr?safe_get(out,0,i):safe_get(out,i,0);
        float y1   =tr?safe_get(out,1,i):safe_get(out,i,1);
        float x2   =tr?safe_get(out,2,i):safe_get(out,i,2);
        float y2   =tr?safe_get(out,3,i):safe_get(out,i,3);
        float score=tr?safe_get(out,4,i):safe_get(out,i,4);
        float cid  =tr?safe_get(out,5,i):safe_get(out,i,5);
        // Skip NaN/Inf, low-confidence, degenerate boxes
        if(is_bad(score)||is_bad(x1)||is_bad(y1)||is_bad(x2)||is_bad(y2)) continue;
        if(score<ct||x2<=x1||y2<=y1) continue;
        int label=(cid>=0.f&&cid<65536.f)?(int)cid:0;
        Object o;
        o.x=x1*sc; o.y=y1*sc; o.w=(x2-x1)*sc; o.h=(y2-y1)*sc;
        o.label=label; o.prob=score;
        objects.push_back(o);
    }
    LOGD("v10 detected %d",(int)objects.size());
}

// ── YOLOv8/v9 anchor-free ─────────────────────────────────────────────────────
static void detect_v8(const ncnn::Mat& in, std::vector<Object>& objects, float ct, float nt){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input(g_input_name.c_str(),in);
    ncnn::Mat raw;
    if(ex.extract(g_out0.c_str(),raw)!=0){ LOGE("v8: extract '%s' failed",g_out0.c_str()); return; }
    ncnn::Mat out=squeeze2d(raw);
    LOGD("v8 out: w=%d h=%d",out.w,out.h);
    if(out.w==0||out.h==0) return;
    // Layout: [4+nc, N] or [N, 4+nc]
    bool t=(out.h==4+g_num_classes);
    int nb=t?out.w:out.h, na=t?out.h:out.w;
    for(int i=0;i<nb;i++){
        float ms=ct; int mc=-1;
        for(int c=0;c<g_num_classes&&(4+c)<na;c++){
            float s=t?safe_get(out,4+c,i):safe_get(out,i,4+c);
            if(!is_bad(s)&&s>ms){ ms=s; mc=c; }
        }
        if(mc<0) continue;
        float cx=t?safe_get(out,0,i):safe_get(out,i,0);
        float cy=t?safe_get(out,1,i):safe_get(out,i,1);
        float bw=t?safe_get(out,2,i):safe_get(out,i,2);
        float bh=t?safe_get(out,3,i):safe_get(out,i,3);
        if(is_bad(cx)||is_bad(cy)||is_bad(bw)||is_bad(bh)||bw<=0||bh<=0) continue;
        Object o;
        o.x=(cx-bw*.5f)/g_input_size; o.y=(cy-bh*.5f)/g_input_size;
        o.w=bw/g_input_size; o.h=bh/g_input_size; o.label=mc; o.prob=ms;
        objects.push_back(o);
    }
    nms(objects,nt);
}

// ── YOLOv5/v6/v7 anchor-based ─────────────────────────────────────────────────
static const float ANCHORS[3][6]={{10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326}};
static void decode_v5(const ncnn::Mat& f, int st, int si, float ct, std::vector<Object>& o){
    int gh=f.h, gw=f.w; if(gh==0||gw==0) return;
    const int na=3, step=5+g_num_classes;
    if(f.c<na*step){ LOGE("v5: too few channels %d",f.c); return; }
    for(int a=0;a<na;a++){
        float aw=ANCHORS[si][a*2], ah=ANCHORS[si][a*2+1];
        for(int gy=0;gy<gh;gy++) for(int gx=0;gx<gw;gx++){
            float obj=sigmoid_f(f.channel(a*step+4).row(gy)[gx]);
            if(is_bad(obj)||obj<ct) continue;
            float mc=0; int mi=0;
            for(int c=0;c<g_num_classes;c++){
                float s=sigmoid_f(f.channel(a*step+5+c).row(gy)[gx])*obj;
                if(!is_bad(s)&&s>mc){ mc=s; mi=c; }
            }
            if(mc<ct) continue;
            float tx=f.channel(a*step  ).row(gy)[gx];
            float ty=f.channel(a*step+1).row(gy)[gx];
            float tw=f.channel(a*step+2).row(gy)[gx];
            float th=f.channel(a*step+3).row(gy)[gx];
            if(is_bad(tx)||is_bad(ty)||is_bad(tw)||is_bad(th)) continue;
            float bx=(sigmoid_f(tx)*2-.5f+gx)*st;
            float by=(sigmoid_f(ty)*2-.5f+gy)*st;
            float bw=powf(sigmoid_f(tw)*2,2)*aw;
            float bh=powf(sigmoid_f(th)*2,2)*ah;
            Object ob;
            ob.x=(bx-bw*.5f)/g_input_size; ob.y=(by-bh*.5f)/g_input_size;
            ob.w=bw/g_input_size; ob.h=bh/g_input_size; ob.label=mi; ob.prob=mc;
            o.push_back(ob);
        }
    }
}
static void detect_v5(const ncnn::Mat& in, std::vector<Object>& o, float ct, float nt){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input(g_input_name.c_str(),in);
    const char* n[]={g_out0.c_str(),g_out1.c_str(),g_out2.c_str()};
    const int st[]={8,16,32};
    for(int s=0;s<3;s++){ ncnn::Mat f; if(ex.extract(n[s],f)==0) decode_v5(f,st[s],s,ct,o); }
    nms(o,nt);
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeInit(
        JNIEnv* env, jobject,
        jstring pp, jstring bp, jint ver, jint isz, jint nc, jboolean gpu,
        jstring o0, jstring o1, jstring o2){
    if(g_initialized){ g_net.clear(); g_initialized=false; }
    g_yolo_version=(int)ver; g_input_size=(int)isz; g_num_classes=(int)nc;
    auto gc=[&](jstring s)->std::string{ const char* c=env->GetStringUTFChars(s,0); std::string r(c); env->ReleaseStringUTFChars(s,c); return r; };
    g_out0=gc(o0); g_out1=gc(o1); g_out2=gc(o2);
    std::string param=gc(pp), bin=gc(bp);
    g_param_path=param;
    g_net.opt.use_vulkan_compute=(bool)gpu;
    g_net.opt.use_fp16_packed    =false;
    g_net.opt.use_fp16_storage   =false;
    g_net.opt.use_fp16_arithmetic=true;
    if(g_net.load_param(param.c_str())!=0){ LOGE("load_param failed"); return JNI_FALSE; }
    if(g_net.load_model(bin.c_str())  !=0){ LOGE("load_model failed");  return JNI_FALSE; }
    ParamInfo pi=parse_param(g_param_path);
    g_input_name=pi.input_name;
    g_initialized=true;
    LOGD("Init OK yolov%d input=%d nc=%d gpu=%d blob_in=%s out0=%s",
         ver,isz,nc,(int)gpu,g_input_name.c_str(),g_out0.c_str());
    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeGetOutputNames(JNIEnv* env, jobject){
    ParamInfo pi=parse_param(g_param_path);
    jclass sc=env->FindClass("java/lang/String");
    jobjectArray arr=env->NewObjectArray((jsize)pi.output_names.size(),sc,nullptr);
    for(size_t i=0;i<pi.output_names.size();i++)
        env->SetObjectArrayElement(arr,(jsize)i,env->NewStringUTF(pi.output_names[i].c_str()));
    return arr;
}

// Returns detections normalised to the full raw bitmap (0..1 per axis).
// Rotation and front-camera mirroring are handled on the Java side.
JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeDetect(
        JNIEnv* env, jobject, jobject bitmap, jfloat ct, jfloat nt, jint nth){
    jclass dc=env->FindClass("com/destik/yolodetector/Detection");
    jmethodID ctor=env->GetMethodID(dc,"<init>","(FFFFIF)V");
    if(!g_initialized) return env->NewObjectArray(0,dc,nullptr);

    AndroidBitmapInfo info;
    if(AndroidBitmap_getInfo(env,bitmap,&info)!=ANDROID_BITMAP_RESULT_SUCCESS)
        return env->NewObjectArray(0,dc,nullptr);
    if(info.format!=ANDROID_BITMAP_FORMAT_RGBA_8888){
        LOGE("unsupported bitmap format %d",info.format);
        return env->NewObjectArray(0,dc,nullptr);
    }
    void* px=nullptr;
    if(AndroidBitmap_lockPixels(env,bitmap,&px)!=ANDROID_BITMAP_RESULT_SUCCESS)
        return env->NewObjectArray(0,dc,nullptr);

    // Center-crop to square — avoids squishing and preserves aspect ratio
    int w=(int)info.width, h=(int)info.height;
    int sq=std::min(w,h);
    int ox=(w-sq)/2, oy=(h-sq)/2;
    const unsigned char* src=(const unsigned char*)px + oy*(int)info.stride + ox*4;

    ncnn::Mat in=ncnn::Mat::from_pixels_resize(
        src, ncnn::Mat::PIXEL_RGBA2RGB,
        sq, sq, (int)info.stride,
        g_input_size, g_input_size);

    AndroidBitmap_unlockPixels(env,bitmap);
    if(in.empty()){ LOGE("from_pixels_resize empty"); return env->NewObjectArray(0,dc,nullptr); }

    const float mv[]={0,0,0}, nv[]={1/255.f,1/255.f,1/255.f};
    in.substract_mean_normalize(mv,nv);
    g_net.opt.num_threads=(int)nth;

    std::vector<Object> objs;
    if     (g_yolo_version>=10) detect_v10(in,objs,ct);
    else if(g_yolo_version>= 8) detect_v8 (in,objs,ct,nt);
    else                        detect_v5 (in,objs,ct,nt);

    // Map from crop-space [0..1] back to full-bitmap-normalised [0..1]
    const float fw=(float)w, fh=(float)h, fsq=(float)sq;
    for(auto& o:objs){
        o.x=(o.x*fsq+ox)/fw;
        o.y=(o.y*fsq+oy)/fh;
        o.w=o.w*fsq/fw;
        o.h=o.h*fsq/fh;
    }

    jobjectArray res=env->NewObjectArray((jsize)objs.size(),dc,nullptr);
    for(size_t i=0;i<objs.size();i++){
        const Object& o=objs[i];
        jobject d=env->NewObject(dc,ctor,o.x,o.y,o.w,o.h,o.label,o.prob);
        env->SetObjectArrayElement(res,(jsize)i,d);
        env->DeleteLocalRef(d);
    }
    return res;
}

JNIEXPORT void JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeRelease(JNIEnv*, jobject){
    if(g_initialized){ g_net.clear(); g_initialized=false; }
}

} // extern "C"
