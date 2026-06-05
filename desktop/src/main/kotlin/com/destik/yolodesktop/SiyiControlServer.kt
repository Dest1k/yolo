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
    private val gimbal: SiyiGimbal,
    private val port: Int = 8081,
    private val streamPort: Int = 8080,
    private val tracking: java.util.concurrent.atomic.AtomicBoolean? = null,
    private val follower: GimbalFollower? = null
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
                    if (path == "/attitude") gimbal.requestAttitude()
                    sendJson(ex, status()); return
                }
                "/rotate"   -> gimbal.rotate(i("yaw"), i("pitch"))
                "/stop"     -> gimbal.stopRotation()
                "/angle"    -> gimbal.setAngle(f("yaw"), f("pitch"))
                "/center"   -> gimbal.center()
                "/zoom"     -> when {
                    q["x"] != null  -> gimbal.absoluteZoom(f("x", 1f))
                    q["dir"] == "in"   -> gimbal.manualZoom(1)
                    q["dir"] == "out"  -> gimbal.manualZoom(-1)
                    else               -> gimbal.manualZoom(0)
                }
                "/focus"    -> when (q["dir"]) {
                    "far"  -> gimbal.manualFocus(1)
                    "near" -> gimbal.manualFocus(-1)
                    else   -> gimbal.manualFocus(0)
                }
                "/autofocus" -> gimbal.autoFocus()
                "/photo"    -> gimbal.takePhoto()
                "/record"   -> gimbal.toggleRecord()
                "/hdr"      -> gimbal.toggleHdr()
                "/mode"     -> when (q["m"]) {
                    "lock"   -> gimbal.setLockMode()
                    "follow" -> gimbal.setFollowMode()
                    "fpv"    -> gimbal.setFpvMode()
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
                "/config"   -> gimbal.requestConfig()
                "/version"  -> { gimbal.requestFirmwareVersion(); gimbal.requestHardwareId() }
                else -> { sendText(ex, 404, "not found"); return }
            }
            sendJson(ex, status())
        } catch (e: Exception) {
            runCatching { sendText(ex, 500, "error: ${e.message}") }
        }
    }

    private fun status(): String {
        val modeName = when (gimbal.motionMode) { 0 -> "lock"; 1 -> "follow"; 2 -> "fpv"; else -> "?" }
        return """{"yaw":${gimbal.yaw},"pitch":${gimbal.pitch},"roll":${gimbal.roll},""" +
               """"recording":${gimbal.recording},"mode":"$modeName","tracking":${tracking?.get() ?: false},""" +
               """"firmware":"${gimbal.firmware}","hardwareId":"${gimbal.hardwareId}"}"""
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

    /** Combined page: live MJPEG video full-screen with the gimbal controls overlaid. */
    private fun panelHtml(): String = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SIYI Gimbal</title><style>
*{box-sizing:border-box} body{margin:0;background:#000;font-family:system-ui,sans-serif;color:#eee;overflow:hidden}
#vid{position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;background:#000;cursor:crosshair}
.ov{position:fixed;z-index:2} button{background:rgba(20,20,20,.6);color:#eee;border:1px solid #777;
border-radius:10px;padding:12px 14px;font-size:16px;margin:3px;cursor:pointer;backdrop-filter:blur(3px)}
button:active{background:#00e676;color:#000}
#st{top:8px;left:8px;font-family:monospace;white-space:pre;background:rgba(0,0,0,.5);padding:8px 10px;border-radius:8px;font-size:13px}
#pad{bottom:12px;left:12px;display:grid;grid-template-columns:repeat(3,56px);gap:4px}
#side{bottom:12px;right:12px;display:flex;flex-direction:column;align-items:flex-end}
#top{top:8px;right:8px;display:flex;flex-wrap:wrap;justify-content:flex-end;max-width:60vw}
input{width:52px;background:rgba(0,0,0,.5);color:#eee;border:1px solid #777;border-radius:6px;padding:6px}
.grp{display:flex;align-items:center;margin:2px 0}
</style></head><body>
<img id="vid" alt="stream">
<div id="st" class="ov">loading…</div>
<div id="trk" class="ov" style="top:8px;left:50%;transform:translateX(-50%);padding:6px 12px;border-radius:8px;font-weight:bold;background:rgba(0,0,0,.5)">TRACK OFF <span style="opacity:.7">(Space / click)</span></div>
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
var STREAM=$streamPort;
document.getElementById('vid').src=location.protocol+'//'+location.hostname+':'+STREAM;
function g(u){return fetch(u).then(r=>r.json()).then(show).catch(function(){})}
function show(s){document.getElementById('st').textContent=
 'yaw '+s.yaw+'  pitch '+s.pitch+'  roll '+s.roll+'\nmode '+s.mode+'  rec '+s.recording+'\nfw '+s.firmware+' hw '+s.hardwareId;
 var t=document.getElementById('trk');t.firstChild.nodeValue=(s.tracking?'● TRACK ON ':'TRACK OFF ');
 t.style.background=s.tracking?'rgba(255,40,40,.75)':'rgba(0,0,0,.5)'}
document.addEventListener('keydown',function(e){if(e.code==='Space'||e.key===' '){e.preventDefault();g('/track')}});
var vv=document.getElementById('vid');
vv.addEventListener('click',function(e){var nw=vv.naturalWidth,nh=vv.naturalHeight;if(!nw||!nh)return;
 var r=vv.getBoundingClientRect();var sc=Math.min(r.width/nw,r.height/nh);var dw=nw*sc,dh=nh*sc;
 var ox=r.left+(r.width-dw)/2,oy=r.top+(r.height-dh)/2;var x=(e.clientX-ox)/dw,y=(e.clientY-oy)/dh;
 if(x<0||x>1||y<0||y>1)return;g('/pick?nx='+x.toFixed(4)+'&ny='+y.toFixed(4))});
function hold(el,down,up){if(!el)return;el.onmousedown=down;el.onmouseup=up;el.onmouseleave=up;
 el.ontouchstart=function(e){e.preventDefault();down()};el.ontouchend=function(e){e.preventDefault();up()};}
document.querySelectorAll('#pad button[data-y]').forEach(function(b){
 hold(b,function(){g('/rotate?yaw='+b.dataset.y+'&pitch='+b.dataset.p)},function(){g('/stop')})});
hold(document.getElementById('zin'),function(){g('/zoom?dir=in')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('zout'),function(){g('/zoom?dir=out')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('ff'),function(){g('/focus?dir=far')},function(){g('/focus?dir=stop')});
hold(document.getElementById('fn'),function(){g('/focus?dir=near')},function(){g('/focus?dir=stop')});
setInterval(function(){fetch('/status').then(function(r){return r.json()}).then(show).catch(function(){})},1000);
</script></body></html>""".trimIndent()
}
