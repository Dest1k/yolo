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
#include <sstream>
#include <iomanip>
#include <signal.h>
#include <setjmp.h>

#include "net.h"

#define TAG "YoloNCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct Object { float x,y,w,h; int label; float prob; };

// ── Native crash diagnostics ─────────────────────────────────────────────────
// A native (SIGSEGV/SIGABRT/…) crash produces no Java stack trace, so we track
// the current native stage and print it from a signal handler. With
// `adb logcat -s YoloNCNN` the line "NATIVE CRASH … stage=…" pinpoints where it
// died (e.g. init:load_model vs detect:extract) for otherwise-undebuggable
// models like YOLOv11.
static const char* volatile g_stage = "idle";
static struct sigaction g_old_sa[64];
// When a guarded region (model load / inference) is active, a fatal signal is
// recovered via siglongjmp instead of killing the process — so an incompatible
// model (e.g. the library YOLOv11 that null-derefs inside ncnn's loader) fails
// gracefully instead of crashing the whole app.
static sigjmp_buf       g_jmp;
static volatile sig_atomic_t g_guarded = 0;
static void crash_handler(int sig, siginfo_t* si, void* uc){
    LOGE("NATIVE CRASH: signal=%d stage=%s", sig, g_stage ? g_stage : "?");
    if(g_guarded){ g_guarded = 0; siglongjmp(g_jmp, sig); }   // recover
    if(sig>=0 && sig<64 && g_old_sa[sig].sa_sigaction)
        g_old_sa[sig].sa_sigaction(sig, si, uc);   // chain → system tombstone/backtrace
    else { signal(sig, SIG_DFL); raise(sig); }
}
static void install_crash_handler(){
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = crash_handler; sa.sa_flags = SA_SIGINFO; sigemptyset(&sa.sa_mask);
    int sigs[] = {SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE};
    for(int s : sigs) sigaction(s, &sa, &g_old_sa[s]);
}

static ncnn::Net* g_net = new ncnn::Net();
static bool       g_initialized  = false;
static int        g_num_classes  = 80;
static int        g_input_size   = 640;
static int        g_yolo_version = 10;
static std::string g_param_path;
static std::string g_input_name = "images";
static std::string g_out0 = "out0";
static std::string g_out1 = "output1";
static std::string g_out2 = "output2";

// Updated every nativeDetect call; read by nativeGetDiagnostics
static std::string g_diag;

// YOLO-FastestV2 activation auto-detect, cached after the first decode and reset
// on each nativeInit (a re-exported model may bake activations in or not).
static bool g_yf_act_done = false;
static bool g_yf_act_obj  = true;   // obj logits → sigmoid
static bool g_yf_act_cls  = true;   // cls logits → softmax
static bool g_yf_act_reg  = true;   // box offsets → sigmoid

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
    return info;
}

// Flatten a 3-D NCNN Mat (c,h,w) into a 2-D (c*h, w) view so row(i) works uniformly.
static ncnn::Mat squeeze2d(const ncnn::Mat& m){
    if(m.c<=1) return m;
    ncnn::Mat out(m.w, m.c*m.h);
    for(int ci=0;ci<m.c;ci++)
        memcpy((float*)out+ci*m.h*m.w, m.channel(ci), m.h*m.w*sizeof(float));
    return out;
}

static inline float safe_get(const ncnn::Mat& m, int row, int col){
    if(row<0||row>=m.h||col<0||col>=m.w) return 0.f;
    return m.row(row)[col];
}

// ── NMS-free parser (YOLOv10 / end-to-end exports): [N,6] = x1,y1,x2,y2,score,cls ──
static void parse_nms_free(const ncnn::Mat& out, std::vector<Object>& objects, float ct){
    // Attributes (6) live on the smaller axis; detections on the larger one.
    bool tr = (out.h < out.w);
    int  nd = tr ? out.w : out.h;

    // Determine pixel vs normalised by scanning rows
    bool pixel=false;
    for(int i=0;i<nd;i++){
        float x2=tr?safe_get(out,2,i):safe_get(out,i,2);
        if(!is_bad(x2)&&x2>1.5f){ pixel=true; break; }
    }
    float sc=pixel?(1.f/g_input_size):1.f;

    float max_conf=0.f;
    for(int i=0;i<nd;i++){
        float x1   =tr?safe_get(out,0,i):safe_get(out,i,0);
        float y1   =tr?safe_get(out,1,i):safe_get(out,i,1);
        float x2   =tr?safe_get(out,2,i):safe_get(out,i,2);
        float y2   =tr?safe_get(out,3,i):safe_get(out,i,3);
        float score=tr?safe_get(out,4,i):safe_get(out,i,4);
        float cid  =tr?safe_get(out,5,i):safe_get(out,i,5);
        if(is_bad(score)||is_bad(x1)||is_bad(y1)||is_bad(x2)||is_bad(y2)) continue;
        if(score>max_conf) max_conf=score;
        if(score<ct||x2<=x1||y2<=y1) continue;
        int label=(cid>=0.f&&cid<65536.f)?(int)cid:0;
        Object o;
        o.x=x1*sc; o.y=y1*sc; o.w=(x2-x1)*sc; o.h=(y2-y1)*sc;
        o.label=label; o.prob=score;
        objects.push_back(o);
    }
    char buf[256];
    snprintf(buf,sizeof(buf),"nms-free|%dx%d|px:%d|maxC:%.2f|dets:%d",
             out.h,out.w,pixel,max_conf,(int)objects.size());
    g_diag=buf;
}

// ── Raw DFL head parser (YOLOv8 / v9 / v11 ncnn pnnx export): [64+nc, N] ──────
// The nihui/ncnn-assets v8/v11 models output the *undecoded* detect head:
// 64 box channels (4 sides × 16 DFL bins) + nc class logits, across the canonical
// 3-level anchor grid (strides 8/16/32). We must DFL-decode the box distances and
// sigmoid the class logits ourselves — the old parser treated all 144 channels as
// "4 box + 140 classes" → garbage coords → every box clipped away ("no boxes").
static void parse_dfl(const ncnn::Mat& out, std::vector<Object>& objects, float ct, float nt, int nc){
    const bool t  = (out.h < out.w);
    const int  nb = t ? out.w : out.h;   // anchors
    const int  reg = 64;                 // 4 sides × 16 DFL bins

    // Build the canonical 3-level grid (strides 8/16/32) for the input size.
    // If it doesn't match the anchor count, derive the size from nb so a custom
    // input size still decodes: nb = (1 + 1/4 + 1/16)·(isz/8)² = 21·isz²/1024.
    const int strides[3] = {8, 16, 32};
    int isz = g_input_size;
    int total = 0; for(int s : strides){ int g = isz / s; total += g * g; }
    if(total != nb){
        int d = (int)lroundf(sqrtf((float)nb * 1024.f / 21.f));
        d = (d / 32) * 32; if(d < 32) d = 32;
        isz = d; total = 0; for(int s : strides){ int g = isz / s; total += g * g; }
    }
    const float norm = (float)isz;

    // Class scores are raw logits → compare in logit space, sigmoid only winners.
    float ctc = ct; if(ctc < 1e-4f) ctc = 1e-4f; if(ctc > 0.9999f) ctc = 0.9999f;
    const float cmp = logf(ctc / (1.f - ctc));

    float max_conf = 0.f;
    int anc = 0;
    for(int lvl = 0; lvl < 3 && anc < nb; lvl++){
        const int st = strides[lvl];
        const int g  = isz / st;
        for(int gy = 0; gy < g && anc < nb; gy++)
        for(int gx = 0; gx < g && anc < nb; gx++){
            const int i = anc++;
            // Grab the contiguous attribute row once per anchor for the common
            // (non-transposed) layout — avoids a per-element row()/bounds-check in
            // the hottest loop, which matters on weak single-board-computer CPUs.
            const float* row = (!t && i < out.h) ? out.row(i) : nullptr;
            auto AT = [&](int attr) -> float { return row ? row[attr] : safe_get(out, attr, i); };
            // ── class gate (cheap; runs for every anchor) ──
            float best = cmp; int bc = -1;
            for(int c = 0; c < nc; c++){
                float s = AT(reg + c);
                if(s > best){ best = s; bc = c; }
            }
            if(bc < 0) continue;                       // nothing over threshold here
            // ── DFL decode (expensive; only for survivors) ──
            float d[4];
            for(int side = 0; side < 4; side++){
                float mx = -1e30f;
                for(int b = 0; b < 16; b++){ float v = AT(side*16 + b); if(v > mx) mx = v; }
                float sum = 0.f, acc = 0.f;
                for(int b = 0; b < 16; b++){
                    float e = expf(AT(side*16 + b) - mx); sum += e; acc += e * b;
                }
                d[side] = sum > 0.f ? acc / sum : 0.f;
            }
            float cx = gx + 0.5f, cy = gy + 0.5f;
            float x1 = (cx - d[0]) * st, y1 = (cy - d[1]) * st;
            float x2 = (cx + d[2]) * st, y2 = (cy + d[3]) * st;
            if(is_bad(x1)||is_bad(y1)||is_bad(x2)||is_bad(y2)||x2<=x1||y2<=y1) continue;
            float conf = sigmoid_f(best);
            if(conf > max_conf) max_conf = conf;
            Object o;
            o.x = x1 / norm; o.y = y1 / norm; o.w = (x2 - x1) / norm; o.h = (y2 - y1) / norm;
            o.label = bc; o.prob = conf;
            objects.push_back(o);
        }
    }
    nms(objects, nt);
    char buf[256];
    snprintf(buf, sizeof(buf), "v8-dfl|%dx%d|nc=%d|isz=%d|maxC:%.2f|dets:%d",
             out.h, out.w, nc, isz, max_conf, (int)objects.size());
    g_diag = buf;
}

// ── Anchor-free parser (already-decoded boxes): [4(+1)+nc, N] ─────────────────
// Handles models whose export *already* decoded the boxes to pixel xywh, with an
// optional objectness channel (YOLOv6-style 4+1+nc). `nc`/`has_obj` come from the
// dispatcher (driven by the configured class count); negative nc means "infer".
static void parse_anchor(const ncnn::Mat& out, std::vector<Object>& objects, float ct, float nt,
                         int nc_in = -1, bool has_obj = false){
    // Attributes live on the smaller axis; anchors on the larger one.
    bool t  = (out.h < out.w);
    int  nb = t ? out.w : out.h;   // anchors
    int  na = t ? out.h : out.w;   // attributes
    int  off = has_obj ? 5 : 4;    // first class channel
    int  nc = nc_in > 0 ? nc_in : (na - off); if(nc < 1) nc = 1;

    // Detect whether box coords are in pixels (0..input_size) or already
    // normalized (0..1). Scan the width attribute across anchors: any value > 2
    // means pixel space. Without this, a model emitting normalized coords gets
    // its boxes divided by input_size → microscopic → clipped away ("no boxes").
    bool pixel=false;
    for(int i=0;i<nb;i++){
        float w=t?safe_get(out,2,i):safe_get(out,i,2);
        if(!is_bad(w)&&w>2.f){ pixel=true; break; }
    }
    float cs = pixel ? (1.f/g_input_size) : 1.f;

    // Detect raw logits: a properly exported head emits sigmoid-activated class
    // scores in 0..1. If any sampled score is >1 or <0, the sigmoid was stripped
    // during export → apply it ourselves, otherwise everything clears the
    // threshold and we get a flood of garbage boxes (maxC like 19.26).
    bool logits=false;
    int sample = nb < 256 ? nb : 256;
    for(int i=0;i<sample && !logits;i++)
        for(int c=0;c<nc;c++){
            float s=t?safe_get(out,off+c,i):safe_get(out,i,off+c);
            if(!is_bad(s)&&(s>1.f||s<0.f)){ logits=true; break; }
        }
    // Compare in logit space to avoid a sigmoid (expf) per score across all
    // anchors×classes; only the kept winner is converted back for display.
    float ctc = ct; if(ctc<1e-4f) ctc=1e-4f; if(ctc>0.9999f) ctc=0.9999f;
    float cmp = logits ? logf(ctc/(1.f-ctc)) : ct;

    float max_raw=-1e30f;
    for(int i=0;i<nb;i++){
        float ms=cmp; int mc=-1;
        for(int c=0;c<nc;c++){
            float s=t?safe_get(out,off+c,i):safe_get(out,i,off+c);
            if(is_bad(s)) continue;
            if(s>max_raw) max_raw=s;
            if(s>ms){ ms=s; mc=c; }
        }
        if(mc<0) continue;
        float prob = logits ? sigmoid_f(ms) : ms;
        if(has_obj){
            // Objectness scales every class equally, so it doesn't change the argmax;
            // apply it only to the kept winner and re-threshold in probability space.
            float obj=t?safe_get(out,4,i):safe_get(out,i,4);
            if(is_bad(obj)) continue;
            prob *= (logits ? sigmoid_f(obj) : obj);
            if(prob < ct) continue;
        }
        float cx=t?safe_get(out,0,i):safe_get(out,i,0);
        float cy=t?safe_get(out,1,i):safe_get(out,i,1);
        float bw=t?safe_get(out,2,i):safe_get(out,i,2);
        float bh=t?safe_get(out,3,i):safe_get(out,i,3);
        if(is_bad(cx)||is_bad(cy)||is_bad(bw)||is_bad(bh)||bw<=0||bh<=0) continue;
        Object o;
        o.x=(cx-bw*.5f)*cs; o.y=(cy-bh*.5f)*cs;
        o.w=bw*cs; o.h=bh*cs; o.label=mc;
        o.prob = prob;
        objects.push_back(o);
    }
    nms(objects,nt);
    float max_conf = logits ? sigmoid_f(max_raw) : max_raw;
    char buf[256];
    snprintf(buf,sizeof(buf),"v8-dec|%dx%d|nc=%d|obj:%d|px:%d|sig:%d|maxC:%.2f|dets:%d",
             out.h,out.w,nc,(int)has_obj,pixel,logits,max_conf,(int)objects.size());
    g_diag=buf;
}

// ── Modern dispatch (v8/v9/v10/v11): pick parser from the actual output shape ──
// This makes detection robust to a mismatched YOLO-version selection: a NMS-free
// model has 6 attributes with a small detection count, while an anchor-free model
// has 4+nc attributes across thousands of anchors.
static void detect_modern(const ncnn::Mat& in, std::vector<Object>& objects, float ct, float nt){
    ncnn::Extractor ex=g_net->create_extractor();
    ex.input(g_input_name.c_str(),in);
    ncnn::Mat raw;
    if(ex.extract(g_out0.c_str(),raw)!=0){
        g_diag="extract FAILED '"+g_out0+"'";
        LOGE("extract '%s' failed",g_out0.c_str()); return;
    }
    ncnn::Mat out=squeeze2d(raw);
    LOGD("out: w=%d h=%d c=%d",out.w,out.h,out.c);
    if(out.w==0||out.h==0){ g_diag="empty output"; return; }

    int amin=std::min(out.w,out.h), amax=std::max(out.w,out.h);
    if(amin<6){ char b[64]; snprintf(b,sizeof(b),"bad shape %dx%d",out.h,out.w); g_diag=b; return; }

    // amin = attributes per anchor, amax = anchor count. Pick the parser from how
    // the attribute count relates to the configured class count:
    //   nc+6/nc... small  → NMS-free [N,6]   (YOLOv10 end-to-end)
    //   nc+64             → raw DFL head      (YOLOv8 / v9 / v11 ncnn export)  ← fixes "no boxes"
    //   nc+5              → decoded + objectness (YOLOv6-style)
    //   nc+4              → decoded anchor-free
    // Falls back to structure when the class count is mis-set.
    const int na = amin;
    const int nc = g_num_classes;
    if(amin==6 && amax<=2000)        { parse_nms_free(out,objects,ct); return; }
    if(nc>0 && na==nc+64)            { parse_dfl   (out,objects,ct,nt,nc);           return; }
    if(nc>0 && na==nc+5)             { parse_anchor(out,objects,ct,nt,nc,true);      return; }
    if(nc>0 && na==nc+4)             { parse_anchor(out,objects,ct,nt,nc,false);     return; }
    // class count unknown / mismatched: infer from the attribute count.
    if(na>=65)                       { parse_dfl   (out,objects,ct,nt,na-64);        return; }
    parse_anchor(out,objects,ct,nt,na-4,false);
}

// ── YOLO-FastestV2 (anchor-based, decoupled head) ────────────────────────────
// dog-qiuqiu/Yolo-FastestV2: ShuffleNetV2 backbone, 2 detection scales (strides
// 16/32). Per-cell channels are DECOUPLED and the class scores are SHARED across
// the na anchors:  [ reg(4*na) | obj(na) | cls(nc) ]. This matches neither the v5
// (na·(5+nc)) nor the v8 single-tensor layout, so it needs its own decoder. Box
// uses the YOLOv5 formula, score = sqrt(obj · cls). Mirrors the working python
// sidecar (tools/yolo-fastestv2-sidecar). Activations are auto-detected so it
// decodes whether or not the export already baked sigmoid/softmax in.
static const int   YF_NA          = 3;   // default anchors-per-cell (overridden per model)
static const int   YF_STRIDES[2]  = {16, 32};
// Repo COCO anchors, pixels at the training input, per stride. Used as a fallback
// when no per-model anchors were supplied. NOT scaled by input size — YOLO-FastestV2
// uses absolute pixel anchors tied to the training config (see train_yolofastest.py).
static const float YF_ANCHORS[2][6] = {
    {12.64f, 19.39f, 37.88f,  51.48f,  55.71f, 138.31f},   // stride 16
    {126.91f,78.23f, 131.57f, 214.55f, 279.92f,258.87f}    // stride 32
};
// Per-model anchors parsed from nativeInit, flattened in stride order
// [stride16 na pairs][stride32 na pairs] — same format as the trainer's `anchors=`.
static std::vector<float> g_yf_anchors;

// Parse a "w,h, w,h, …" anchor string (commas and/or spaces) into floats.
static std::vector<float> parse_anchors(const std::string& s){
    std::vector<float> v;
    const char* p = s.c_str();
    while(*p){
        char* end = nullptr;
        float f = strtof(p, &end);
        if(end == p) { p++; continue; }   // skip a separator/garbage char
        v.push_back(f); p = end;
    }
    return v;
}

// Decode one detection scale of a YOLO-FastestV2 head into `objects`.
// `raw` is the untouched ncnn output (C,H,W); C = 5·na + nc, H=W=grid. `na` (anchors
// per cell) and `anc` (na w,h pairs for this stride) come from the caller.
static void decode_yolofastest(const ncnn::Mat& raw, int stride, const float* anc, int na,
                               std::vector<Object>& objects, float ct, float& max_conf){
    const int C  = (raw.dims >= 3) ? raw.c : raw.h;
    const int W  = (raw.dims >= 3) ? raw.w : (int)lroundf(sqrtf((float)raw.w));
    const int HW = (raw.dims >= 3) ? raw.h * raw.w : raw.w;
    const int nc = C - 5 * na;
    if(na < 1 || nc <= 0 || HW <= 0 || W <= 0) return;
    // Cache the base pointer of every channel once (avoids a temp Mat + refcount per
    // pixel access in the hot loop). Works for (C,H,W) and a flat (C, H·W) layout.
    std::vector<const float*> base(C);
    for(int ch = 0; ch < C; ch++)
        base[ch] = (raw.dims >= 3) ? (const float*)raw.channel(ch) : raw.row(ch);
    auto CH = [&](int ch, int cell) -> float { return base[ch][cell]; };

    // Auto-detect (once) whether obj/cls/reg are raw logits or already activated:
    // any value outside [−0.01, 1.01] means the activation wasn't baked in.
    if(!g_yf_act_done){
        auto out_of_unit = [&](int c0, int c1) -> bool {
            for(int ch = c0; ch < c1; ch++)
                for(int cell = 0; cell < HW; cell++){ float v = CH(ch, cell); if(v < -0.01f || v > 1.01f) return true; }
            return false;
        };
        g_yf_act_reg = out_of_unit(0, 4 * na);
        g_yf_act_obj = out_of_unit(4 * na, 5 * na);
        g_yf_act_cls = out_of_unit(5 * na, 5 * na + nc);
        g_yf_act_done = true;
    }

    for(int cell = 0; cell < HW; cell++){
        const int gx = cell % W, gy = cell / W;
        // Shared class score for this cell: argmax + (softmax prob of winner | prob).
        int   bc = 0; float bl = -1e30f;
        for(int c = 0; c < nc; c++){ float s = CH(5 * na + c, cell); if(s > bl){ bl = s; bc = c; } }
        float cls_p;
        if(g_yf_act_cls){                       // logits → softmax prob of the winner
            float sum = 0.f;
            for(int c = 0; c < nc; c++) sum += expf(CH(5 * na + c, cell) - bl);
            cls_p = sum > 0.f ? 1.f / sum : 0.f;
        } else cls_p = bl;                      // already a probability

        for(int a = 0; a < na; a++){
            float obj_raw = CH(4 * na + a, cell);
            float obj   = g_yf_act_obj ? sigmoid_f(obj_raw) : obj_raw;
            float score = sqrtf(obj * cls_p);
            if(is_bad(score)) continue;
            if(score > max_conf) max_conf = score;
            if(score < ct) continue;

            float tx = CH(a * 4 + 0, cell), ty = CH(a * 4 + 1, cell);
            float tw = CH(a * 4 + 2, cell), th = CH(a * 4 + 3, cell);
            if(is_bad(tx)||is_bad(ty)||is_bad(tw)||is_bad(th)) continue;
            // YOLOv5 box formula; skip the inner sigmoid if reg is already activated.
            float sx = g_yf_act_reg ? sigmoid_f(tx) : tx;
            float sy = g_yf_act_reg ? sigmoid_f(ty) : ty;
            float sw = g_yf_act_reg ? sigmoid_f(tw) : tw;
            float sh = g_yf_act_reg ? sigmoid_f(th) : th;
            float bcx = (sx * 2.f - 0.5f + gx) * stride;
            float bcy = (sy * 2.f - 0.5f + gy) * stride;
            float bw  = (sw * 2.f) * (sw * 2.f) * anc[a * 2];
            float bh  = (sh * 2.f) * (sh * 2.f) * anc[a * 2 + 1];
            float x1 = bcx - bw * 0.5f, y1 = bcy - bh * 0.5f;
            float x2 = bcx + bw * 0.5f, y2 = bcy + bh * 0.5f;
            if(is_bad(x1)||is_bad(y1)||is_bad(x2)||is_bad(y2)||x2<=x1||y2<=y1) continue;
            Object o;
            o.x = x1 / g_input_size; o.y = y1 / g_input_size;
            o.w = (x2 - x1) / g_input_size; o.h = (y2 - y1) / g_input_size;
            o.label = bc; o.prob = score;
            objects.push_back(o);
        }
    }
}

static void detect_yolofastest(const ncnn::Mat& in, std::vector<Object>& objects, float ct, float nt){
    ncnn::Extractor ex = g_net->create_extractor();
    ex.input(g_input_name.c_str(), in);
    const char* blobs[3] = { g_out0.c_str(), g_out1.c_str(), g_out2.c_str() };
    float max_conf = 0.f;
    int scales = 0;
    for(int bi = 0; bi < 3; bi++){
        if(blobs[bi] == nullptr || blobs[bi][0] == '\0') continue;
        ncnn::Mat raw;
        if(ex.extract(blobs[bi], raw) != 0) continue;
        int C  = (raw.dims >= 3) ? raw.c : raw.h;
        int W  = (raw.dims >= 3) ? raw.w : (int)lroundf(sqrtf((float)raw.w));
        if(W <= 0 || C <= 0) continue;
        // anchors-per-cell from the channel count: C = 5·na + nc. Fall back to the
        // default na=3 (and infer nc) if the configured class count doesn't fit.
        int na = (C - g_num_classes) / 5;
        if(na < 1 || 5 * na + g_num_classes != C) na = YF_NA;
        // Derive the stride from the grid (order-independent + self-correcting). The
        // two scales are 16/32; the smaller one takes the first na anchor pairs.
        int stride = (int)lroundf((float)g_input_size / (float)W);
        bool small = stride <= (YF_STRIDES[0] + YF_STRIDES[1]) / 2;
        // Prefer per-model anchors (g_yf_anchors, [stride16 na pairs][stride32 na pairs]);
        // fall back to the built-in defaults when none/too few were supplied.
        const float* anc;
        int off = small ? 0 : na * 2;
        if((int)g_yf_anchors.size() >= off + na * 2) anc = &g_yf_anchors[off];
        else                                          anc = small ? YF_ANCHORS[0] : YF_ANCHORS[1];
        decode_yolofastest(raw, stride, anc, na, objects, ct, max_conf);
        scales++;
    }
    nms(objects, nt);
    char buf[160];
    snprintf(buf, sizeof(buf), "yolo-fastestv2|scales=%d|isz=%d|act(o=%d,c=%d,r=%d)|maxC:%.2f|dets:%d",
             scales, g_input_size, (int)g_yf_act_obj, (int)g_yf_act_cls, (int)g_yf_act_reg,
             max_conf, (int)objects.size());
    g_diag = buf;
}

// ── YOLOv5/v6/v7 anchor-based ────────────────────────────────────────────────
static const float ANCHORS[3][6]={{10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326}};
static void decode_v5(const ncnn::Mat& f, int st, int si, float ct, std::vector<Object>& o, float& max_conf){
    int gh=f.h, gw=f.w; if(gh==0||gw==0) return;
    const int na=3, step=5+g_num_classes;
    if(f.c<na*step){ LOGE("v5: too few channels %d",f.c); return; }
    for(int a=0;a<na;a++){
        float aw=ANCHORS[si][a*2], ah=ANCHORS[si][a*2+1];
        for(int gy=0;gy<gh;gy++) for(int gx=0;gx<gw;gx++){
            float obj=sigmoid_f(f.channel(a*step+4).row(gy)[gx]);
            if(is_bad(obj)) continue;
            if(obj>max_conf) max_conf=obj;
            if(obj<ct) continue;
            float mc=0; int mi=0;
            for(int c=0;c<g_num_classes;c++){
                float s=sigmoid_f(f.channel(a*step+5+c).row(gy)[gx])*obj;
                if(!is_bad(s)&&s>mc){ mc=s; mi=c; }
            }
            if(mc<ct) continue;
            float tx=f.channel(a*step  ).row(gy)[gx], ty=f.channel(a*step+1).row(gy)[gx];
            float tw=f.channel(a*step+2).row(gy)[gx], th=f.channel(a*step+3).row(gy)[gx];
            if(is_bad(tx)||is_bad(ty)||is_bad(tw)||is_bad(th)) continue;
            float bx=(sigmoid_f(tx)*2-.5f+gx)*st, by=(sigmoid_f(ty)*2-.5f+gy)*st;
            float bw=powf(sigmoid_f(tw)*2,2)*aw,   bh=powf(sigmoid_f(th)*2,2)*ah;
            Object ob;
            ob.x=(bx-bw*.5f)/g_input_size; ob.y=(by-bh*.5f)/g_input_size;
            ob.w=bw/g_input_size; ob.h=bh/g_input_size; ob.label=mi; ob.prob=mc;
            o.push_back(ob);
        }
    }
}
static void detect_v5(const ncnn::Mat& in, std::vector<Object>& o, float ct, float nt){
    ncnn::Extractor ex=g_net->create_extractor();
    ex.input(g_input_name.c_str(),in);
    const char* n[]={g_out0.c_str(),g_out1.c_str(),g_out2.c_str()};
    const int st[]={8,16,32};
    float max_conf=0.f;
    for(int s=0;s<3;s++){ ncnn::Mat f; if(ex.extract(n[s],f)==0) decode_v5(f,st[s],s,ct,o,max_conf); }
    nms(o,nt);
    char buf[128]; snprintf(buf,sizeof(buf),"v5|maxC:%.2f|dets:%d",max_conf,(int)o.size());
    g_diag=buf;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeInit(
        JNIEnv* env, jobject,
        jstring pp, jstring bp, jint ver, jint isz, jint nc, jboolean gpu,
        jstring o0, jstring o1, jstring o2, jstring yfAnchors){
    if(g_initialized){ g_net->clear(); g_initialized=false; }
    g_yf_act_done=false;   // re-detect YOLO-FastestV2 activations for the new model
    g_yolo_version=(int)ver; g_input_size=(int)isz; g_num_classes=(int)nc;
    auto gc=[&](jstring s)->std::string{
        if(!s) return std::string();
        const char* c=env->GetStringUTFChars(s,0); std::string r(c?c:"");
        if(c) env->ReleaseStringUTFChars(s,c); return r;
    };
    g_out0=gc(o0); g_out1=gc(o1); g_out2=gc(o2);
    g_yf_anchors = parse_anchors(gc(yfAnchors));   // per-model YOLO-FastestV2 anchors
    std::string param=gc(pp), bin=gc(bp);
    g_param_path=param;

    // Only enable Vulkan when a usable GPU is actually present — requesting it on
    // a device/driver without one leads to a null-device crash at extract time.
    bool use_gpu=false;
#if NCNN_VULKAN
    use_gpu=((bool)gpu && ncnn::get_gpu_count()>0);
#endif
    g_net->opt.use_vulkan_compute =use_gpu;
    // fp16 on CPU: storage halves memory bandwidth, arithmetic ~doubles conv
    // throughput on ARMv8.2+/ARMv9 (Oryon / Snapdragon). ncnn auto-falls back to
    // fp32 on CPUs without FEAT_FP16, so it's a free win where supported.
    g_net->opt.use_fp16_packed    =true;
    g_net->opt.use_fp16_storage   =true;
    g_net->opt.use_fp16_arithmetic=true;
    g_net->opt.use_packing_layout =true;
    // Run INT8-quantized models (ncnn2int8) natively — uses Oryon i8mm/dotprod
    // for another ~1.5-2x over fp16 on CPU. Harmless for fp32 models.
    g_net->opt.use_int8_inference =true;

    static bool s_handler=false;
    if(!s_handler){ install_crash_handler(); s_handler=true; }

    // Guard the loader: some model files null-deref inside ncnn (e.g. the library
    // YOLOv11). If that happens, the signal handler siglongjmp's back here; we
    // abandon the half-built (possibly corrupt) net and report failure instead of
    // taking the whole app down.
    if(sigsetjmp(g_jmp, 1) != 0){
        g_guarded = 0;
        g_net = new ncnn::Net();            // leak the corrupt one; fresh net for next time
        g_initialized = false;
        g_diag = std::string("load CRASHED @ ") + (g_stage ? g_stage : "?");
        LOGE("recovered from native crash during model load");
        g_stage = "idle";
        return JNI_FALSE;
    }
    g_guarded = 1;
    g_stage="init:load_param";
    if(g_net->load_param(param.c_str())!=0){ g_guarded=0; LOGE("load_param failed"); g_stage="idle"; return JNI_FALSE; }
    g_stage="init:load_model";
    if(g_net->load_model(bin.c_str())  !=0){ g_guarded=0; LOGE("load_model failed");  g_stage="idle"; return JNI_FALSE; }
    g_guarded = 0;
    g_stage="init:parse";
    ParamInfo pi=parse_param(g_param_path);
    g_input_name=pi.input_name;

    // Auto-correct the output blob name for single-output models (v8/v9/v10/v11):
    // catalog/UI defaults like "output0" often don't match the real blob ("out0").
    if(g_yolo_version>=8 && !pi.output_names.empty()){
        bool present=false;
        for(const auto& n:pi.output_names) if(n==g_out0){ present=true; break; }
        if(!present){
            g_out0=pi.output_names.back();
            LOGD("auto output blob → %s",g_out0.c_str());
        }
    } else if(g_yolo_version<8 && !pi.output_names.empty()){
        // Anchor-based v5/v6/v7 have up to 3 outputs (strides 8/16/32). Catalog/UI
        // defaults like "output"/"output1"/"output2" often don't match the real
        // blob names (e.g. nihui yolov5s uses out0/out1/out2). Remap by file order
        // — otherwise every extract() fails and no boxes are ever produced.
        bool present=false;
        for(const auto& n:pi.output_names) if(n==g_out0){ present=true; break; }
        if(!present){
            const auto& on=pi.output_names;
            if(on.size()>0) g_out0=on[0];
            if(on.size()>1) g_out1=on[1];
            if(on.size()>2) g_out2=on[2];
            LOGD("auto v5 output blobs → %s %s %s",g_out0.c_str(),g_out1.c_str(),g_out2.c_str());
        }
    }
    g_initialized=true;
    g_stage="idle";
    g_diag="init OK";
    LOGD("Init OK yolov%d input=%d nc=%d gpu=%d blob_in=%s out0=%s",
         ver,isz,nc,(int)use_gpu,g_input_name.c_str(),g_out0.c_str());
    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeGetOutputNames(JNIEnv* env, jobject, jstring pp){
    const char* c=env->GetStringUTFChars(pp,0);
    std::string path(c?c:"");
    env->ReleaseStringUTFChars(pp,c);
    // Pure text parse of the .param file — no model/GPU load, so it cannot crash
    // even for models whose Vulkan path is unstable on this device.
    ParamInfo pi=parse_param(path.empty()?g_param_path:path);
    jclass sc=env->FindClass("java/lang/String");
    jobjectArray arr=env->NewObjectArray((jsize)pi.output_names.size(),sc,nullptr);
    for(size_t i=0;i<pi.output_names.size();i++)
        env->SetObjectArrayElement(arr,(jsize)i,env->NewStringUTF(pi.output_names[i].c_str()));
    return arr;
}

JNIEXPORT jstring JNICALL
Java_com_destik_yolodetector_YoloDetector_nativeGetDiagnostics(JNIEnv* env, jobject){
    return env->NewStringUTF(g_diag.c_str());
}

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

    int w=(int)info.width, h=(int)info.height;

    // Preprocessing must match how the model was trained. YOLO-FastestV2 trains on a
    // plain (aspect-squishing) resize to input×input, so we stretch for it; every
    // other family trains with aspect-preserving letterbox (pad with gray 114).
    const bool stretch = (g_yolo_version == 2);
    float scale_x, scale_y;     // model-px per original-px, per axis
    int   pad_x, pad_y;         // letterbox padding (0 for stretch)
    ncnn::Mat in;
    if(stretch){
        in = ncnn::Mat::from_pixels_resize(
            (const unsigned char*)px, ncnn::Mat::PIXEL_RGBA2RGB,
            w, h, (int)info.stride, g_input_size, g_input_size);
        AndroidBitmap_unlockPixels(env,bitmap);
        if(in.empty()){ LOGE("from_pixels_resize empty"); return env->NewObjectArray(0,dc,nullptr); }
        scale_x = (float)g_input_size / w; scale_y = (float)g_input_size / h;
        pad_x = 0; pad_y = 0;
    } else {
        float scale = std::min((float)g_input_size / w, (float)g_input_size / h);
        int   nw    = (int)(w * scale + .5f);
        int   nh    = (int)(h * scale + .5f);
        pad_x = (g_input_size - nw) / 2;
        pad_y = (g_input_size - nh) / 2;
        ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
            (const unsigned char*)px, ncnn::Mat::PIXEL_RGBA2RGB,
            w, h, (int)info.stride, nw, nh);
        AndroidBitmap_unlockPixels(env,bitmap);
        if(resized.empty()){ LOGE("from_pixels_resize empty"); return env->NewObjectArray(0,dc,nullptr); }
        ncnn::copy_make_border(resized, in,
            pad_y, g_input_size - nh - pad_y,
            pad_x, g_input_size - nw - pad_x,
            ncnn::BORDER_CONSTANT, 114.f);
        scale_x = scale; scale_y = scale;
    }

    const float mv[]={0,0,0}, nv[]={1/255.f,1/255.f,1/255.f};
    in.substract_mean_normalize(mv,nv);
    g_net->opt.num_threads=(int)nth;

    std::vector<Object> objs;
    g_stage="detect:extract";
    if(g_yolo_version==2)      detect_yolofastest(in,objs,ct,nt);  // YOLO-FastestV2 (decoupled head)
    else if(g_yolo_version>=8) detect_modern    (in,objs,ct,nt);  // v8/v9/v10/v11 (auto layout)
    else                       detect_v5        (in,objs,ct,nt);  // v5/v6/v7 anchor-based
    g_stage="idle";

    // Reverse the preprocessing: model-norm coords → original-frame norm coords.
    // Works for both paths via per-axis scale (letterbox: scale_x==scale_y, pad>0;
    // stretch: pad==0, scale_x/scale_y differ). scale_*·{w,h} is the content size in
    // model px (== g_input_size for stretch, == nw/nh for letterbox).
    const float fw=(float)w, fh=(float)h;
    const float fpx=(float)pad_x, fpy=(float)pad_y;
    const float fis=(float)g_input_size;
    for(auto& o:objs){
        float x1 = o.x       * fis;
        float y1 = o.y       * fis;
        float x2 = (o.x+o.w) * fis;
        float y2 = (o.y+o.h) * fis;
        o.x = (x1 - fpx) / (scale_x * fw);
        o.y = (y1 - fpy) / (scale_y * fh);
        o.w = (x2 - x1)  / (scale_x * fw);
        o.h = (y2 - y1)  / (scale_y * fh);
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
    if(g_initialized){ g_net->clear(); g_initialized=false; }
}

} // extern "C"
