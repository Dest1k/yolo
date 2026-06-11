package com.destik.yolodesktop

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress
import java.net.URI

/**
 * Tiny HTTP control API + web panel for a [SiyiGimbal], so the gimbal can be
 * driven from a browser or scripts over the LAN in the headless deployment.
 *
 * Endpoints (GET, query params):
 *   /              → HTML control panel
 *   /status        → JSON {yaw,pitch,roll,recording,mode,firmware,hardwareId}
 *   /rotate?yaw=&pitch=   speeds -100..100   /stop
 *   /angle?yaw=&pitch=    absolute degrees
 *   /center
 *   /zoom?dir=in|out|stop  |  /zoom?x=4.5 (absolute)
 *   /focus?dir=far|near|stop  |  /autofocus
 *   /photo  /record  /hdr
 *   /mode?m=lock|follow|fpv
 *   /attitude /config /version   (re-request from the gimbal)
 */
class SiyiControlServer(
    private val gimbal: SiyiGimbal? = null,
    private val port: Int = 8081,
    private val streamPort: Int = 8080,
    private val tracking: java.util.concurrent.atomic.AtomicBoolean? = null,
    private val follower: GimbalFollower? = null,
    // Manual target capture: the panel reports a drawn rectangle (normalised
    // x1,y1,x2,y2) here, and a clear request when the lock is reset. Both are
    // optional so the server works with or without a manual tracker wired up.
    private val onLock: ((FloatArray) -> Unit)? = null,
    private val onClearLock: (() -> Unit)? = null
) {

    private var server: HttpServer? = null

    fun start() {
        val s = HttpServer.create(InetSocketAddress(port), 0)
        s.createContext("/") { ex -> handle(ex) }
        s.executor = null
        s.start()
        server = s
    }

    fun stop() { server?.stop(0); server = null }

    private fun handle(ex: HttpExchange) {
        try {
            val uri = ex.requestURI
            val path = uri.path
            val q = parseQuery(uri)
            fun f(k: String, d: Float = 0f) = q[k]?.toFloatOrNull() ?: d
            fun i(k: String, d: Int = 0) = q[k]?.toIntOrNull() ?: d

            when (path) {
                "/" -> { sendHtml(ex, panelHtml()); return }
                "/status", "/attitude" -> {
                    if (path == "/attitude") gimbal?.requestAttitude()
                    sendJson(ex, status()); return
                }
                "/rotate"   -> gimbal?.rotate(i("yaw"), i("pitch"))
                "/stop"     -> gimbal?.stopRotation()
                "/angle"    -> gimbal?.setAngle(f("yaw"), f("pitch"))
                "/center"   -> gimbal?.center()
                "/zoom"     -> when {
                    q["x"] != null  -> gimbal?.absoluteZoom(f("x", 1f))
                    q["dir"] == "in"   -> gimbal?.manualZoom(1)
                    q["dir"] == "out"  -> gimbal?.manualZoom(-1)
                    else               -> gimbal?.manualZoom(0)
                }
                "/focus"    -> when (q["dir"]) {
                    "far"  -> gimbal?.manualFocus(1)
                    "near" -> gimbal?.manualFocus(-1)
                    else   -> gimbal?.manualFocus(0)
                }
                "/autofocus" -> gimbal?.autoFocus()
                "/photo"    -> gimbal?.takePhoto()
                "/record"   -> gimbal?.toggleRecord()
                "/hdr"      -> gimbal?.toggleHdr()
                "/mode"     -> when (q["m"]) {
                    "lock"   -> gimbal?.setLockMode()
                    "follow" -> gimbal?.setFollowMode()
                    "fpv"    -> gimbal?.setFpvMode()
                }
                "/track"    -> tracking?.let { t ->
                    when (q["on"]) { "1", "true" -> t.set(true); "0", "false" -> t.set(false); else -> t.set(!t.get()) }
                }
                "/pick"     -> {
                    val nx = q["nx"]?.toFloatOrNull(); val ny = q["ny"]?.toFloatOrNull()
                    if (follower != null && nx != null && ny != null) {
                        follower.requestPick(nx, ny); tracking?.set(true)
                    }
                }
                // Manual target capture: lock a hand-drawn rectangle (normalised) /
                // release it. Works without a gimbal (e.g. CSI camera).
                "/lock"     -> {
                    val x1 = f("x1", -1f); val y1 = f("y1", -1f)
                    val x2 = f("x2", -1f); val y2 = f("y2", -1f)
                    if (onLock != null && x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0) {
                        val a = minOf(x1, x2).coerceIn(0f, 1f); val b = minOf(y1, y2).coerceIn(0f, 1f)
                        val c = maxOf(x1, x2).coerceIn(0f, 1f); val d = maxOf(y1, y2).coerceIn(0f, 1f)
                        if (c - a > 0.01f && d - b > 0.01f) onLock.invoke(floatArrayOf(a, b, c, d))
                    }
                }
                "/unlock"   -> onClearLock?.invoke()
                "/config"   -> gimbal?.requestConfig()
                "/version"  -> { gimbal?.requestFirmwareVersion(); gimbal?.requestHardwareId() }
                else -> { sendText(ex, 404, "not found"); return }
            }
            sendJson(ex, status())
        } catch (e: Exception) {
            runCatching { sendText(ex, 500, "error: ${e.message}") }
        }
    }

    private fun status(): String {
        val g = gimbal
        val modeName = when (g?.motionMode) { 0 -> "lock"; 1 -> "follow"; 2 -> "fpv"; else -> "?" }
        // Guard against NaN/Infinity (invalid JSON) and JSON-escape the gimbal-reported
        // text — otherwise a stray byte from the camera breaks JSON.parse in the panel
        // and the page hangs on "loading…".
        fun n(v: Float) = if (v.isFinite()) v else 0f
        return """{"hasGimbal":${g != null},"yaw":${n(g?.yaw ?: 0f)},"pitch":${n(g?.pitch ?: 0f)},"roll":${n(g?.roll ?: 0f)},""" +
               """"recording":${g?.recording ?: false},"mode":"$modeName","tracking":${tracking?.get() ?: false},""" +
               """"firmware":${jsonStr(g?.firmware ?: "")},"hardwareId":${jsonStr(g?.hardwareId ?: "")}}"""
    }

    /** Minimal JSON string encoder (quotes + escapes) so gimbal-reported text can't break the JSON. */
    private fun jsonStr(s: String): String {
        val sb = StringBuilder("\"")
        for (c in s) when {
            c == '"'  -> sb.append("\\\"")
            c == '\\' -> sb.append("\\\\")
            c < ' '   -> sb.append("\\u%04x".format(c.code))
            else      -> sb.append(c)
        }
        return sb.append('"').toString()
    }

    private fun parseQuery(uri: URI): Map<String, String> =
        uri.query?.split("&")?.mapNotNull {
            val p = it.split("=", limit = 2); if (p.size == 2) p[0] to p[1] else null
        }?.toMap() ?: emptyMap()

    private fun sendJson(ex: HttpExchange, body: String) {
        ex.responseHeaders.add("Content-Type", "application/json")
        sendText(ex, 200, body)
    }
    private fun sendHtml(ex: HttpExchange, body: String) {
        ex.responseHeaders.add("Content-Type", "text/html; charset=utf-8")
        sendText(ex, 200, body)
    }
    private fun sendText(ex: HttpExchange, code: Int, body: String) {
        val bytes = body.toByteArray()
        ex.sendResponseHeaders(code, bytes.size.toLong())
        ex.responseBody.use { it.write(bytes) }
    }

    /**
     * Combined page: full-screen live MJPEG video with the manual-capture overlay
     * and (when a gimbal is attached) the gimbal controls overlaid on top.
     *
     * Manual capture: drag a rectangle on the video → POST /lock; a single tap
     * (with a gimbal) picks the YOLO target under the cursor. Keys: C/Esc clear
     * the lock, H toggles the gimbal panel (hidden by default for a CSI camera),
     * Space toggles gimbal auto-follow.
     */
    private fun panelHtml(): String = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO panel</title><style>
*{box-sizing:border-box} body{margin:0;background:#000;font-family:system-ui,sans-serif;color:#eee;overflow:hidden;-webkit-user-select:none;user-select:none}
#vid{position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;background:#000;cursor:crosshair;touch-action:none}
.ov{position:fixed;z-index:2} button{background:rgba(20,20,20,.6);color:#eee;border:1px solid #777;
border-radius:10px;padding:12px 14px;font-size:16px;margin:3px;cursor:pointer;backdrop-filter:blur(3px)}
button:active{background:#00e676;color:#000}
#st{top:8px;left:8px;font-family:monospace;white-space:pre;background:rgba(0,0,0,.5);padding:8px 10px;border-radius:8px;font-size:13px}
#hint{bottom:8px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.5);padding:6px 12px;border-radius:8px;font-size:12px;opacity:.85}
#sel{position:fixed;z-index:3;border:2px solid #00e6e6;background:rgba(0,230,230,.12);display:none;pointer-events:none}
#pad{bottom:12px;left:12px;display:grid;grid-template-columns:repeat(3,56px);gap:4px}
#side{bottom:12px;right:12px;display:flex;flex-direction:column;align-items:flex-end}
#top{top:8px;right:8px;display:flex;flex-wrap:wrap;justify-content:flex-end;max-width:60vw}
input{width:52px;background:rgba(0,0,0,.5);color:#eee;border:1px solid #777;border-radius:6px;padding:6px}
.grp{display:flex;align-items:center;margin:2px 0}
</style></head><body>
<img id="vid" alt="stream">
<div id="sel"></div>
<div id="st" class="ov">loading…</div>
<div id="trk" class="ov" style="top:8px;left:50%;transform:translateX(-50%);padding:6px 12px;border-radius:8px;font-weight:bold;background:rgba(0,0,0,.5)">TRACK OFF <span style="opacity:.7">(Space)</span></div>
<div id="hint" class="ov">drag = lock target · C = clear · H = gimbal panel</div>
<div id="pad" class="ov">
<span></span><button data-y="0" data-p="60">▲</button><span></span>
<button data-y="-60" data-p="0">◀</button><button onclick="g('/center')">●</button><button data-y="60" data-p="0">▶</button>
<span></span><button data-y="0" data-p="-60">▼</button><span></span>
</div>
<div id="top" class="ov">
<button onclick="g('/track')">track</button>
<button onclick="g('/mode?m=lock')">lock</button><button onclick="g('/mode?m=follow')">follow</button><button onclick="g('/mode?m=fpv')">fpv</button>
<button onclick="g('/photo')">📷</button><button onclick="g('/record')">⏺</button><button onclick="g('/hdr')">hdr</button>
</div>
<div id="side" class="ov">
<div class="grp"><button id="zin">＋</button><button id="zout">－</button>x<input id="zx" value="2"><button onclick="g('/zoom?x='+zx.value)">set</button></div>
<div class="grp"><button onclick="g('/autofocus')">AF</button><button id="ff">focus＋</button><button id="fn">focus－</button></div>
<div class="grp">yaw<input id="ay" value="0">pitch<input id="ap" value="0"><button onclick="g('/angle?yaw='+ay.value+'&pitch='+ap.value)">go</button></div>
</div>
<script>
var STREAM=$streamPort, HASGIMBAL=${gimbal != null}, showG=HASGIMBAL;
var vv=document.getElementById('vid'), sel=document.getElementById('sel');
vv.src=location.protocol+'//'+location.hostname+':'+STREAM;
function g(u){return fetch(u).then(function(r){if(!r.ok)throw 0;return r.json()}).then(show).catch(fail)}
function fail(){document.getElementById('st').textContent='disconnected — retrying…'}
function show(s){var st=document.getElementById('st');
 if(s.hasGimbal){st.textContent='yaw '+s.yaw+'  pitch '+s.pitch+'  roll '+s.roll+'\nmode '+s.mode+'  rec '+s.recording+'\nfw '+s.firmware+' hw '+s.hardwareId}
 else{st.textContent='no gimbal — video + manual capture only'}
 var t=document.getElementById('trk');t.firstChild.nodeValue=(s.tracking?'● TRACK ON ':'TRACK OFF ');
 t.style.background=s.tracking?'rgba(255,40,40,.75)':'rgba(0,0,0,.5)'}
// 'pad/side/top' are the gimbal controls: hidden by default when there's no gimbal,
// but H always toggles them so they can be summoned regardless. 'trk' stays visible.
function applyG(){['pad','side','top'].forEach(function(id){var el=document.getElementById(id);
 if(el)el.style.display=showG?'':'none'})}
applyG();
document.addEventListener('keydown',function(e){
 if(e.key==='h'||e.key==='H'){showG=!showG;applyG()}
 else if(e.key==='c'||e.key==='C'||e.key==='Escape'){g('/unlock')}
 else if(e.code==='Space'||e.key===' '){e.preventDefault();if(HASGIMBAL)g('/track')}});
// Map a screen point to normalised image coords, honouring object-fit:contain letterboxing.
function mapPt(e){var nw=vv.naturalWidth,nh=vv.naturalHeight;if(!nw||!nh)return null;
 var r=vv.getBoundingClientRect();var sc=Math.min(r.width/nw,r.height/nh);var dw=nw*sc,dh=nh*sc;
 var ox=r.left+(r.width-dw)/2,oy=r.top+(r.height-dh)/2;var x=(e.clientX-ox)/dw,y=(e.clientY-oy)/dh;
 if(x<0||x>1||y<0||y>1)return null;return {x:x,y:y}}
// Drag a rectangle to lock; a small drag (a tap) picks the gimbal target instead.
var drag=null;
vv.addEventListener('pointerdown',function(e){var p=mapPt(e);if(!p)return;e.preventDefault();
 drag={x1:p.x,y1:p.y,sx:e.clientX,sy:e.clientY};if(vv.setPointerCapture)vv.setPointerCapture(e.pointerId)});
vv.addEventListener('pointermove',function(e){if(!drag)return;
 var x=Math.min(drag.sx,e.clientX),y=Math.min(drag.sy,e.clientY),w=Math.abs(e.clientX-drag.sx),h=Math.abs(e.clientY-drag.sy);
 sel.style.display='block';sel.style.left=x+'px';sel.style.top=y+'px';sel.style.width=w+'px';sel.style.height=h+'px'});
vv.addEventListener('pointerup',function(e){if(!drag)return;sel.style.display='none';
 var p=mapPt(e),moved=Math.abs(e.clientX-drag.sx)+Math.abs(e.clientY-drag.sy);
 if(p&&moved>8)g('/lock?x1='+drag.x1.toFixed(4)+'&y1='+drag.y1.toFixed(4)+'&x2='+p.x.toFixed(4)+'&y2='+p.y.toFixed(4));
 else if(p&&HASGIMBAL)g('/pick?nx='+p.x.toFixed(4)+'&ny='+p.y.toFixed(4));
 drag=null});
function hold(el,down,up){if(!el)return;el.onmousedown=down;el.onmouseup=up;el.onmouseleave=up;
 el.ontouchstart=function(e){e.preventDefault();down()};el.ontouchend=function(e){e.preventDefault();up()};}
document.querySelectorAll('#pad button[data-y]').forEach(function(b){
 hold(b,function(){g('/rotate?yaw='+b.dataset.y+'&pitch='+b.dataset.p)},function(){g('/stop')})});
hold(document.getElementById('zin'),function(){g('/zoom?dir=in')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('zout'),function(){g('/zoom?dir=out')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('ff'),function(){g('/focus?dir=far')},function(){g('/focus?dir=stop')});
hold(document.getElementById('fn'),function(){g('/focus?dir=near')},function(){g('/focus?dir=stop')});
function poll(){fetch('/status').then(function(r){if(!r.ok)throw 0;return r.json()}).then(show).catch(fail)}
poll();setInterval(poll,1000);
</script></body></html>""".trimIndent()
}
