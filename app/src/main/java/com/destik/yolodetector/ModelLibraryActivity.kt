package com.destik.yolodetector

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

class ModelLibraryActivity : AppCompatActivity() {

    private val TAG = "ModelLibrary"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_library)

        val rv = findViewById<RecyclerView>(R.id.rvModels)
        rv.layoutManager = LinearLayoutManager(this)
        rv.adapter = ModelAdapter(ModelEntry.CATALOG) { entry -> downloadAndUse(entry) }
    }

    private fun localFiles(entry: ModelEntry): Pair<File, File> {
        val dir = File(filesDir, "models/${entry.id}").also { it.mkdirs() }
        return Pair(File(dir, "${entry.id}.param"), File(dir, "${entry.id}.bin"))
    }

    private fun isDownloaded(entry: ModelEntry): Boolean {
        val (p, b) = localFiles(entry)
        return p.exists() && b.exists() && b.length() > 10_000L
    }

    private fun downloadAndUse(entry: ModelEntry) {
        if (isDownloaded(entry)) { returnModel(entry); return }

        val (paramFile, binFile) = localFiles(entry)
        val adapter = (findViewById<RecyclerView>(R.id.rvModels).adapter as ModelAdapter)
        val idx = ModelEntry.CATALOG.indexOf(entry)

        lifecycleScope.launch {
            adapter.setProgress(idx, 0, "Загрузка .param…")
            val paramErr = downloadFile(entry.paramUrl, paramFile) { p ->
                adapter.setProgress(idx, p / 2, ".param $p%")
            }
            if (paramErr != null) {
                paramFile.delete()
                toast("Ошибка .param:\n$paramErr")
                adapter.setProgress(idx, -1, "")
                return@launch
            }

            adapter.setProgress(idx, 50, "Загрузка .bin…")
            val binErr = downloadFile(entry.binUrl, binFile) { p ->
                adapter.setProgress(idx, 50 + p / 2, ".bin $p%")
            }
            if (binErr != null || !isDownloaded(entry)) {
                binFile.delete()
                val msg = binErr ?: "Файл слишком мал (${binFile.length()} байт)"
                toast("Ошибка .bin:\n$msg")
                adapter.setProgress(idx, -1, "")
                return@launch
            }

            adapter.setProgress(idx, -1, "")
            returnModel(entry)
        }
    }

    /** Returns null on success, or an error description on failure. */
    private suspend fun downloadFile(
        urlStr: String,
        dest: File,
        onProgress: suspend (Int) -> Unit
    ): String? = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "download start: $urlStr")
            val realUrl = resolveLfsUrl(urlStr)
            Log.d(TAG, "resolved url: $realUrl")
            streamToFile(realUrl, dest, onProgress)
            Log.d(TAG, "download done: ${dest.name} ${dest.length()} bytes")
            null
        } catch (e: Exception) {
            Log.e(TAG, "download failed: $urlStr", e)
            "${e.javaClass.simpleName}: ${e.message}"
        }
    }

    private fun resolveLfsUrl(urlStr: String): String {
        val conn = openGet(urlStr)
        val code = conn.responseCode
        if (code !in 200..299) {
            conn.disconnect()
            throw Exception("HTTP $code от ${host(urlStr)}")
        }
        val contentType = conn.contentType ?: ""
        val contentLen  = conn.contentLengthLong
        Log.d(TAG, "probe: code=$code len=$contentLen type=$contentType")

        if (contentLen in 1..600 || contentType.startsWith("text/plain")) {
            val text = try {
                conn.inputStream.bufferedReader().readText()
            } finally {
                conn.disconnect()
            }
            Log.d(TAG, "probe body (${text.length} chars): ${text.take(120)}")
            if (text.trimStart().startsWith("version https://git-lfs.github.com")) {
                val oid = Regex("oid sha256:([0-9a-f]{64})").find(text)?.groupValues?.get(1)
                    ?: throw Exception("OID не найден в LFS-указателе")
                val size = Regex("size (\\d+)").find(text)?.groupValues?.get(1)?.toLongOrNull()
                    ?: throw Exception("size не найден в LFS-указателе")
                Log.d(TAG, "LFS pointer: oid=${oid.take(12)}… size=$size")
                return fetchLfsBatchUrl(urlStr, oid, size)
                    ?: throw Exception("LFS batch API не вернул URL")
            }
            return urlStr
        }

        conn.disconnect()
        return urlStr
    }

    private fun fetchLfsBatchUrl(originalUrl: String, oid: String, size: Long): String? {
        val m = Regex("https://raw\\.githubusercontent\\.com/([^/]+/[^/]+)/").find(originalUrl)
            ?: run { Log.e(TAG, "Cannot parse repo from: $originalUrl"); return null }
        val repoPath = m.groupValues[1]
        val batchUrl = "https://github.com/$repoPath.git/info/lfs/objects/batch"
        Log.d(TAG, "LFS batch POST → $batchUrl")

        val body = """{"operation":"download","transfers":["basic"],"objects":[{"oid":"$oid","size":$size}]}"""
        val conn = (URL(batchUrl).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 20_000; readTimeout = 20_000
            setRequestProperty("Content-Type", "application/vnd.git-lfs+json")
            setRequestProperty("Accept",       "application/vnd.git-lfs+json")
            setRequestProperty("User-Agent",   "YoloDetector/1.0")
        }
        conn.outputStream.use { it.write(body.toByteArray()) }
        val batchCode = conn.responseCode
        if (batchCode !in 200..299) {
            val err = runCatching { conn.errorStream?.bufferedReader()?.readText() }.getOrNull() ?: ""
            conn.disconnect()
            throw Exception("LFS batch HTTP $batchCode: $err".take(200))
        }
        val resp = conn.inputStream.bufferedReader().readText()
        conn.disconnect()
        Log.d(TAG, "LFS batch resp: ${resp.take(300)}")
        return runCatching {
            JSONObject(resp)
                .getJSONArray("objects").getJSONObject(0)
                .getJSONObject("actions").getJSONObject("download")
                .getString("href")
        }.getOrElse { e -> Log.e(TAG, "Parse LFS resp failed", e); null }
    }

    private fun streamToFile(urlStr: String, dest: File, onProgress: suspend (Int) -> Unit) {
        val conn = openGet(urlStr)
        val code = conn.responseCode
        if (code !in 200..299) {
            conn.disconnect()
            throw Exception("HTTP $code от ${host(urlStr)}")
        }
        val total = conn.contentLengthLong
        var done = 0L
        conn.inputStream.use { inp ->
            FileOutputStream(dest).use { out ->
                val buf = ByteArray(32 * 1024)
                var n: Int
                while (inp.read(buf).also { n = it } != -1) {
                    out.write(buf, 0, n); done += n
                    if (total > 0) kotlinx.coroutines.runBlocking {
                        onProgress((done * 100 / total).toInt())
                    }
                }
            }
        }
        conn.disconnect()
    }

    private fun openGet(urlStr: String): HttpURLConnection =
        (URL(urlStr).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 20_000
            readTimeout    = 120_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "YoloDetector/1.0")
            connect()
        }

    private fun host(url: String) = runCatching { URL(url).host }.getOrDefault(url.take(50))

    private fun returnModel(entry: ModelEntry) {
        val (paramFile, binFile) = localFiles(entry)
        // Производительность по максимуму, но без вылета на медленные LITTLE-ядра:
        // берём ~3/4 ядер (большой кластер), от 4 до 8 потоков.
        val cores = Runtime.getRuntime().availableProcessors()
        val threads = (cores * 3 / 4).coerceIn(4, 8)
        val config = ModelConfig(
            paramPath     = paramFile.absolutePath,
            binPath       = binFile.absolutePath,
            yoloVersion   = entry.yoloVersion,
            numClasses    = entry.numClasses,
            inputSize     = entry.inputSize,
            outputName0   = entry.outputName0,
            outputName1   = entry.outputName1,
            outputName2   = entry.outputName2,
            // Tuned defaults so the model works well immediately — no manual setup.
            confThreshold = entry.confThreshold,
            nmsThreshold  = entry.nmsThreshold,
            numThreads    = threads,
            classNames    = CocoLabels.NAMES,
            engine        = "ncnn",
            useGPU        = false
        )
        setResult(Activity.RESULT_OK, Intent().putExtra("config", Gson().toJson(config)))
        finish()
    }

    private fun toast(msg: String) = runOnUiThread {
        Log.w(TAG, "toast: $msg")
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    // ── Adapter ───────────────────────────────────────────────────────────────

    private inner class ModelAdapter(
        private val items: List<ModelEntry>,
        private val onUse: (ModelEntry) -> Unit
    ) : RecyclerView.Adapter<ModelAdapter.VH>() {

        private val progress = HashMap<Int, Pair<Int, String>>()

        fun setProgress(idx: Int, pct: Int, label: String) = runOnUiThread {
            if (pct < 0) progress.remove(idx) else progress[idx] = Pair(pct, label)
            notifyItemChanged(idx)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            VH(LayoutInflater.from(parent.context).inflate(R.layout.item_model_library, parent, false))

        override fun getItemCount() = items.size

        override fun onBindViewHolder(h: VH, position: Int) {
            val entry = items[position]
            h.name.text  = entry.name
            h.desc.text  = entry.description
            h.badge.text = if (entry.yoloVersion == 1) "ND" else "v${entry.yoloVersion}"

            val prog = progress[position]
            if (prog != null) {
                h.btnUse.isEnabled = false
                h.btnUse.text = prog.second
                h.progressBar.visibility = View.VISIBLE
                h.progressBar.progress   = prog.first
            } else {
                h.progressBar.visibility = View.GONE
                h.btnUse.isEnabled = true
                h.btnUse.text = if (isDownloaded(entry)) "Использовать"
                                else "Скачать (~${entry.approxMb} MB)"
            }
            h.btnUse.setOnClickListener { onUse(entry) }
        }

        inner class VH(v: View) : RecyclerView.ViewHolder(v) {
            val name: TextView       = v.findViewById(R.id.tvModelName)
            val desc: TextView       = v.findViewById(R.id.tvModelDesc)
            val badge: TextView      = v.findViewById(R.id.tvVersionBadge)
            val btnUse: Button       = v.findViewById(R.id.btnUseModel)
            val progressBar: ProgressBar = v.findViewById(R.id.progressDownload)
        }
    }
}
