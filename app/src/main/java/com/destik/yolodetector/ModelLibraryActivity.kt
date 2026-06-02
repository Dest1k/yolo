package com.destik.yolodetector

import android.app.Activity
import android.content.Intent
import android.os.Bundle
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_library)

        val rv = findViewById<RecyclerView>(R.id.rvModels)
        rv.layoutManager = LinearLayoutManager(this)
        rv.adapter = ModelAdapter(ModelEntry.CATALOG) { entry ->
            downloadAndUse(entry)
        }
    }

    private fun localFiles(entry: ModelEntry): Pair<File, File> {
        val dir = File(filesDir, "models/${entry.id}").also { it.mkdirs() }
        return Pair(File(dir, "${entry.id}.param"), File(dir, "${entry.id}.bin"))
    }

    private fun isDownloaded(entry: ModelEntry): Boolean {
        val (p, b) = localFiles(entry)
        // Guard against LFS pointer files (actual model is always > 10 KB)
        return p.exists() && b.exists() && b.length() > 10_000L
    }

    private fun downloadAndUse(entry: ModelEntry) {
        if (isDownloaded(entry)) {
            returnModel(entry)
            return
        }

        val (paramFile, binFile) = localFiles(entry)
        val adapter = (findViewById<RecyclerView>(R.id.rvModels).adapter as ModelAdapter)
        val idx = ModelEntry.CATALOG.indexOf(entry)

        lifecycleScope.launch {
            adapter.setProgress(idx, 0, "Загрузка .param…")
            val paramOk = downloadFile(entry.paramUrl, paramFile) { p ->
                adapter.setProgress(idx, p / 2, "Загрузка .param… $p%")
            }
            if (!paramOk) { toast("Ошибка загрузки .param"); adapter.setProgress(idx, -1, ""); return@launch }

            adapter.setProgress(idx, 50, "Загрузка .bin…")
            val binOk = downloadFile(entry.binUrl, binFile) { p ->
                adapter.setProgress(idx, 50 + p / 2, "Загрузка .bin… $p%")
            }
            if (!binOk || !isDownloaded(entry)) {
                binFile.delete()
                toast("Ошибка загрузки .bin")
                adapter.setProgress(idx, -1, "")
                return@launch
            }

            adapter.setProgress(idx, -1, "")
            returnModel(entry)
        }
    }

    private suspend fun downloadFile(urlStr: String, dest: File, onProgress: suspend (Int) -> Unit): Boolean =
        withContext(Dispatchers.IO) {
            runCatching {
                val realUrl = resolveLfsUrl(urlStr)
                streamToFile(realUrl, dest, onProgress)
            }.getOrDefault(false)
        }

    /** If the URL points to a Git LFS pointer, resolve it via the LFS batch API. */
    private fun resolveLfsUrl(urlStr: String): String {
        // First, do a HEAD request or small fetch to detect LFS pointer
        val probe = URL(urlStr).openConnection() as HttpURLConnection
        probe.connectTimeout = 15_000
        probe.readTimeout    = 10_000
        probe.instanceFollowRedirects = true
        probe.setRequestProperty("User-Agent", "YoloDetector/1.0")
        probe.connect()
        if (probe.responseCode !in 200..299) return urlStr
        val contentType = probe.contentType ?: ""
        val contentLen  = probe.contentLengthLong
        // LFS pointer files are small plain-text files (< 300 bytes)
        if (contentLen in 1..512 || contentType.startsWith("text/plain")) {
            val text = probe.inputStream.bufferedReader().readText()
            if (text.startsWith("version https://git-lfs.github.com")) {
                // Parse OID and size from pointer
                val oid  = Regex("oid sha256:([0-9a-f]+)").find(text)?.groupValues?.get(1) ?: return urlStr
                val size = Regex("size (\\d+)").find(text)?.groupValues?.get(1)?.toLongOrNull() ?: 0L
                return fetchLfsBatchUrl(urlStr, oid, size) ?: urlStr
            }
        }
        probe.disconnect()
        return urlStr  // not LFS, return as-is
    }

    /** Call GitHub LFS batch API to get the real download URL for an object. */
    private fun fetchLfsBatchUrl(originalUrl: String, oid: String, size: Long): String? {
        // Derive batch API URL from raw URL
        // https://raw.githubusercontent.com/OWNER/REPO/BRANCH/path  →
        // https://github.com/OWNER/REPO.git/info/lfs/objects/batch
        val m = Regex("https://raw\\.githubusercontent\\.com/([^/]+/[^/]+)/").find(originalUrl)
            ?: return null
        val repoPath = m.groupValues[1]
        val batchUrl = "https://github.com/$repoPath.git/info/lfs/objects/batch"

        val body = """{"operation":"download","transfers":["basic"],"objects":[{"oid":"$oid","size":$size}]}"""
        val conn = URL(batchUrl).openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.doOutput = true
        conn.setRequestProperty("Content-Type", "application/vnd.git-lfs+json")
        conn.setRequestProperty("Accept",       "application/vnd.git-lfs+json")
        conn.setRequestProperty("User-Agent",   "YoloDetector/1.0")
        conn.connectTimeout = 15_000
        conn.readTimeout    = 15_000
        conn.outputStream.use { it.write(body.toByteArray()) }
        if (conn.responseCode !in 200..299) return null
        val resp = conn.inputStream.bufferedReader().readText()
        return runCatching {
            JSONObject(resp)
                .getJSONArray("objects").getJSONObject(0)
                .getJSONObject("actions").getJSONObject("download")
                .getString("href")
        }.getOrNull()
    }

    private fun streamToFile(urlStr: String, dest: File, onProgress: suspend (Int) -> Unit): Boolean {
        val conn = URL(urlStr).openConnection() as HttpURLConnection
        conn.connectTimeout = 30_000
        conn.readTimeout    = 120_000
        conn.instanceFollowRedirects = true
        conn.setRequestProperty("User-Agent", "YoloDetector/1.0")
        conn.connect()
        if (conn.responseCode !in 200..299) return false
        val total = conn.contentLengthLong
        var done  = 0L
        conn.inputStream.use { inp ->
            FileOutputStream(dest).use { out ->
                val buf = ByteArray(32 * 1024)
                var n: Int
                while (inp.read(buf).also { n = it } != -1) {
                    out.write(buf, 0, n)
                    done += n
                    if (total > 0) kotlinx.coroutines.runBlocking { onProgress((done * 100 / total).toInt()) }
                }
            }
        }
        return true
    }

    private fun returnModel(entry: ModelEntry) {
        val (paramFile, binFile) = localFiles(entry)
        val config = ModelConfig(
            paramPath    = paramFile.absolutePath,
            binPath      = binFile.absolutePath,
            yoloVersion  = entry.yoloVersion,
            numClasses   = entry.numClasses,
            inputSize    = entry.inputSize,
            outputName0  = entry.outputName0,
            outputName1  = entry.outputName1,
            outputName2  = entry.outputName2
        )
        setResult(Activity.RESULT_OK, Intent().putExtra("config", Gson().toJson(config)))
        finish()
    }

    private fun toast(msg: String) = runOnUiThread {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    // ── Adapter ───────────────────────────────────────────────────────────────

    private inner class ModelAdapter(
        private val items: List<ModelEntry>,
        private val onUse: (ModelEntry) -> Unit
    ) : RecyclerView.Adapter<ModelAdapter.VH>() {

        private val progress = HashMap<Int, Pair<Int, String>>() // idx → (pct, label)

        fun setProgress(idx: Int, pct: Int, label: String) {
            runOnUiThread {
                if (pct < 0) progress.remove(idx) else progress[idx] = Pair(pct, label)
                notifyItemChanged(idx)
            }
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
            VH(LayoutInflater.from(parent.context).inflate(R.layout.item_model_library, parent, false))

        override fun getItemCount() = items.size

        override fun onBindViewHolder(h: VH, position: Int) {
            val entry = items[position]
            h.name.text = entry.name
            h.desc.text = entry.description
            h.badge.text = "v${entry.yoloVersion}"

            val prog = progress[position]
            if (prog != null) {
                h.btnUse.isEnabled = false
                h.btnUse.text = prog.second
                h.progressBar.visibility = View.VISIBLE
                h.progressBar.progress   = prog.first
            } else {
                h.progressBar.visibility = View.GONE
                val downloaded = isDownloaded(entry)
                h.btnUse.isEnabled = true
                h.btnUse.text = if (downloaded) "Использовать" else "Скачать (~${entry.approxMb} MB)"
            }
            h.btnUse.setOnClickListener { onUse(entry) }
        }

        inner class VH(v: View) : RecyclerView.ViewHolder(v) {
            val name: TextView    = v.findViewById(R.id.tvModelName)
            val desc: TextView    = v.findViewById(R.id.tvModelDesc)
            val badge: TextView   = v.findViewById(R.id.tvVersionBadge)
            val btnUse: Button    = v.findViewById(R.id.btnUseModel)
            val progressBar: ProgressBar = v.findViewById(R.id.progressDownload)
        }
    }
}
