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
static std::string g_out0 = "output0";
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

// Parse .param file → return output blob names (blobs produced but never consumed)
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
        char line[4096]={}; fgets(line,sizeof(line),f); // skip params
    }
    fclose(f);
    std::set<std::string> seen;
    for(auto& t:top_order){
        if(seen.count(t)) continue; seen.insert(t);
        if(!all_bottoms.count(t)) result.push_back(t);
    }
    LOGD("param parse: found %d outputs",(int)result.size());
    for(auto& n:result) LOGD("  output: %s",n.c_str());
    return result;
}

// ── YOLOv8/v9 anchor-free ───────────────────────────────────────────────────
static void detect_v8(const ncnn::Mat& in,std::vector<Object>& objects,float ct,float nt){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input("images",in);
    ncnn::Mat out;
    if(ex.extract(g_out0.c_str(),out)!=0){LOGE("v8: extract '%s' failed",g_out0.c_str());return;}
    LOGD("v8 out w=%d h=%d c=%d",out.w,out.h,out.c);
    bool t=(out.h==4+g_num_classes);
    int nb2=t?out.w:out.h, na=t?out.h:out.w;
    for(int i=0;i<nb2;i++){
        float ms=ct; int mc=-1;
        for(int c=0;c<g_num_classes&&(4+c)<na;c++){
            float s=t?out.row(4+c)[i]:out.row(i)[4+c];
            if(s>ms){ms=s;mc=c;}
        }
        if(mc<0) continue;
        float cx=t?out.row(0)[i]:out.row(i)[0], cy=t?out.row(1)[i]:out.row(i)[1];
        float bw=t?out.row(2)[i]:out.row(i)[2], bh=t?out.row(3)[i]:out.row(i)[3];
        Object o; o.x=(cx-bw*.5f)/g_input_size; o.y=(cy-bh*.5f)/g_input_size;
        o.w=bw/g_input_size; o.h=bh/g_input_size; o.label=mc; o.prob=ms;
        objects.push_back(o);
    }
    nms(objects,nt);
}

// ── YOLOv10/v11 NMS-free [N,6] or [6,N]: x1,y1,x2,y2,score,cls ───────────────────
static void detect_v10(const ncnn::Mat& in,std::vector<Object>& objects,float ct){
    ncnn::Extractor ex=g_net.create_extractor();
    ex.input("images",in);
    ncnn::Mat out;
    if(ex.extract(g_out0.c_str(),out)!=0){LOGE("v10: extract '%s' failed",g_out0.c_str());return;}
    LOGD("v10 out w=%d h=%d c=%d",out.w,out.h,out.c);
    bool tr=(out.h==6); // [6,N] transposed
    int nd=tr?out.w:out.h;
    // auto-detect pixel vs normalised from first valid box
    bool pixel=true;
    for(int i=0;i<nd&&i<10;i++){
        float s=tr?out.row(4)[i]:out.row(i)[4];
        if(s<ct) continue;
        float x2=tr?out.row(2)[i]:out.row(i)[2];
        if(x2<=1.5f){pixel=false;} break;
    }
    float sc=pixel?(1.f/g_input_size):1.f;
    LOGD("v10 nd=%d pixel=%d sc=%f",nd,pixel,sc);
    for(int i=0;i<nd;i++){
        float x1,y1,x2,y2,score,cid;
        if(tr){x1=out.row(0)[i];y1=out.row(1)[i];x2=out.row(2)[i];y2=out.row(3)[i];score=out.row(4)[i];cid=out.row(5)[i];}
        else  {x1=out.row(i)[0];y1=out.row(i)[1];x2=out.row(i)[2];y2=out.row(i)[3];score=out.row(i)[4];cid=out.row(i)[5];}
        if(score<ct||x2<=x1||y2<=y1) continue;
        Object o; o.x=x1*sc; o.y=y1*sc; o.w=(x2-x1)*sc; o.h=(y2-y1)*sc;
        o.label=(int)cid; o.prob=score; objects.push_back(o);
    }
}

// ── YOLOv5/v6/v7 anchor-based ────────────────────────────────────────────────
static const float ANCHORS[3][6]={{10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326}};
static void decode_v5(const ncnn::Mat& f,int st,int si,float ct,std::vector<Object>& o){
    int gh=f.h,gw=f.w; const int na=3,step=5+g_num_classes;
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
    int r=g_net.load_param(param); if(r!=0){LOGE("load_param failed");env->ReleaseStringUTFChars(pp,param);env->ReleaseStringUTFChars(bp,bin);return JNI_FALSE;}
    r=g_net.load_model(bin);        if(r!=0){LOGE("load_model failed"); env->ReleaseStringUTFChars(pp,param);env->ReleaseStringUTFChars(bp,bin);return JNI_FALSE;}
    env->ReleaseStringUTFChars(pp,param); env->ReleaseStringUTFChars(bp,bin);
    g_initialized=true;
    LOGD("Init OK yolov%d input=%d nc=%d gpu=%d out0=%s",ver,isz,nc,(int)gpu,g_out0.c_str());
    return JNI_TRUE;
}

// Parse-only — no inference, no crash risk
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
    AndroidBitmapInfo info; AndroidBitmap_getInfo(env,bitmap,&info);
    void* px; AndroidBitmap_lockPixels(env,bitmap,&px);
    int pt=(info.format==ANDROID_BITMAP_FORMAT_RGBA_8888)?ncnn::Mat::PIXEL_RGBA2RGB:ncnn::Mat::PIXEL_RGB;
    ncnn::Mat in=ncnn::Mat::from_pixels_resize((const unsigned char*)px,pt,info.width,info.height,g_input_size,g_input_size);
    AndroidBitmap_unlockPixels(env,bitmap);
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
