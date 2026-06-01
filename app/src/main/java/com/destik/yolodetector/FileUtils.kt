package com.destik.yolodetector

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream

object FileUtils {

    fun copyToCache(ctx: Context, uri: Uri, name: String): String? = runCatching {
        val dest = File(ctx.cacheDir, name)
        ctx.contentResolver.openInputStream(uri)!!.use { inp ->
            FileOutputStream(dest).use { out -> inp.copyTo(out) }
        }
        dest.absolutePath
    }.getOrNull()

    fun getFileName(ctx: Context, uri: Uri): String {
        var name = uri.lastPathSegment ?: "unknown"
        ctx.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) name = cursor.getString(idx)
            }
        }
        return name
    }
}
