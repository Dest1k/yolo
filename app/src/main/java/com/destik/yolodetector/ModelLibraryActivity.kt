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
                val conn = URL(urlStr).openConnection() as HttpURLConnection
                conn.connectTimeout = 15_000
                conn.readTimeout    = 60_000
                conn.instanceFollowRedirects = true
                conn.connect()
                if (conn.responseCode !in 200..299) return@runCatching false
                val total = conn.contentLengthLong
                var done  = 0L
                conn.inputStream.use { inp ->
                    FileOutputStream(dest).use { out ->
                        val buf = ByteArray(32 * 1024)
                        var n: Int
                        while (inp.read(buf).also { n = it } != -1) {
                            out.write(buf, 0, n)
                            done += n
                            if (total > 0) onProgress((done * 100 / total).toInt())
                        }
                    }
                }
                true
            }.getOrDefault(false)
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
