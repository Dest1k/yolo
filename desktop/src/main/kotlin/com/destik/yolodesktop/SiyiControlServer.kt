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
class SiyiControlServer(private val gimbal: SiyiGimbal, private val port: Int = 8081) {

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
                "/" -> { sendHtml(ex, PANEL); return }
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
               """"recording":${gimbal.recording},"mode":"$modeName",""" +
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

    companion object {
        private val PANEL = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SIYI Gimbal</title><style>
body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#eee;margin:0;padding:16px}
h2{margin:0 0 12px} .row{margin:10px 0} button{background:#333;color:#eee;border:1px solid #555;
border-radius:8px;padding:12px 16px;font-size:16px;margin:3px;cursor:pointer} button:active{background:#00e676;color:#000}
.pad{display:grid;grid-template-columns:repeat(3,64px);gap:4px;justify-content:center}
#st{font-family:monospace;white-space:pre;background:#222;padding:10px;border-radius:8px}
input{width:64px;background:#222;color:#eee;border:1px solid #555;border-radius:6px;padding:6px}
</style></head><body>
<h2>SIYI Gimbal</h2>
<div id="st">loading…</div>
<div class="row"><b>Move</b> (hold)
<div class="pad">
<span></span><button data-y="0" data-p="60">▲</button><span></span>
<button data-y="-60" data-p="0">◀</button><button onclick="g('/center')">●</button><button data-y="60" data-p="0">▶</button>
<span></span><button data-y="0" data-p="-60">▼</button><span></span>
</div></div>
<div class="row"><b>Angle</b> yaw<input id="ay" value="0"> pitch<input id="ap" value="0">
<button onclick="g('/angle?yaw='+ay.value+'&pitch='+ap.value)">go</button></div>
<div class="row"><b>Zoom</b>
<button onmousedown="g('/zoom?dir=in')" onmouseup="g('/zoom?dir=stop')" ontouchstart="g('/zoom?dir=in')" ontouchend="g('/zoom?dir=stop')">+</button>
<button onmousedown="g('/zoom?dir=out')" onmouseup="g('/zoom?dir=stop')" ontouchstart="g('/zoom?dir=out')" ontouchend="g('/zoom?dir=stop')">−</button>
x<input id="zx" value="2"><button onclick="g('/zoom?x='+zx.value)">set</button>
<button onclick="g('/autofocus')">AF</button>
<button onmousedown="g('/focus?dir=far')" onmouseup="g('/focus?dir=stop')">focus+</button>
<button onmousedown="g('/focus?dir=near')" onmouseup="g('/focus?dir=stop')">focus−</button></div>
<div class="row"><b>Mode</b>
<button onclick="g('/mode?m=lock')">lock</button><button onclick="g('/mode?m=follow')">follow</button>
<button onclick="g('/mode?m=fpv')">fpv</button></div>
<div class="row"><b>Camera</b>
<button onclick="g('/photo')">photo</button><button onclick="g('/record')">record</button><button onclick="g('/hdr')">hdr</button></div>
<script>
function g(u){return fetch(u).then(r=>r.json()).then(show).catch(()=>{})}
function show(s){document.getElementById('st').textContent=
 'yaw '+s.yaw+'  pitch '+s.pitch+'  roll '+s.roll+'\nmode '+s.mode+'  rec '+s.recording+'\nfw '+s.firmware+'  hw '+s.hardwareId}
document.querySelectorAll('.pad button[data-y]').forEach(b=>{
 const send=()=>g('/rotate?yaw='+b.dataset.y+'&pitch='+b.dataset.p), stop=()=>g('/stop');
 b.onmousedown=send;b.onmouseup=stop;b.onmouseleave=stop;b.ontouchstart=e=>{e.preventDefault();send()};b.ontouchend=stop;});
setInterval(()=>fetch('/status').then(r=>r.json()).then(show).catch(()=>{}),1000);
</script></body></html>""".trimIndent()
    }
}
