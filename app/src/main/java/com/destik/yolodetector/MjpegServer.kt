package com.destik.yolodetector

import android.graphics.Bitmap
import android.util.Log
import java.io.ByteArrayOutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

class MjpegServer(val port: Int = 8080) {

    private val TAG = "MjpegServer"

    private val latestJpeg = AtomicReference<ByteArray>(null)
    private var serverSocket: ServerSocket? = null
    @Volatile var running = false; private set
    private val clients = CopyOnWriteArrayList<Socket>()
    private val clientCount = AtomicInteger(0)
    private val acceptThread = Executors.newSingleThreadExecutor()
    private val clientPool  = Executors.newCachedThreadPool()

    fun start() {
        if (running) return
        running = true
        serverSocket = ServerSocket(port).also { it.reuseAddress = true }
        Log.d(TAG, "MJPEG server started on port $port")
        acceptThread.execute {
            while (running) {
                try {
                    val client = serverSocket!!.accept()
                    client.soTimeout = 5_000
                    clients.add(client)
                    clientCount.incrementAndGet()
                    clientPool.execute { serveClient(client) }
                } catch (e: Exception) {
                    if (running) Log.w(TAG, "accept: ${e.message}")
                }
            }
        }
    }

    fun stop() {
        running = false
        clients.forEach { runCatching { it.close() } }
        clients.clear()
        clientCount.set(0)
        runCatching { serverSocket?.close() }
        serverSocket = null
        Log.d(TAG, "MJPEG server stopped")
    }

    fun hasClients(): Boolean = clientCount.get() > 0

    /**
     * Push a frame. JPEG encoding happens here, on the caller's thread.
     * No-op if no clients are connected (zero overhead).
     */
    fun pushFrame(bitmap: Bitmap, quality: Int = 55) {
        if (!running || !hasClients()) return
        val baos = ByteArrayOutputStream(64 * 1024)
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, baos)
        latestJpeg.set(baos.toByteArray())
    }

    fun clientCount(): Int = clientCount.get()

    private fun serveClient(socket: Socket) {
        Log.d(TAG, "client connected: ${socket.inetAddress.hostAddress}")
        try {
            val inp = socket.getInputStream().bufferedReader()
            var line = inp.readLine()
            while (!line.isNullOrEmpty()) line = inp.readLine()

            val out = socket.getOutputStream()
            out.write(
                "HTTP/1.1 200 OK\r\n" +
                "Content-Type: multipart/x-mixed-replace; boundary=--mjpeg\r\n" +
                "Cache-Control: no-store, no-cache, must-revalidate\r\n" +
                "Pragma: no-cache\r\n" +
                "Connection: close\r\n\r\n"
            )

            var lastSent: ByteArray? = null
            while (running && !socket.isClosed) {
                val frame = latestJpeg.get()
                if (frame == null || frame === lastSent) { Thread.sleep(5); continue }
                lastSent = frame
                try {
                    out.write("----mjpeg\r\nContent-Type: image/jpeg\r\nContent-Length: ${frame.size}\r\n\r\n")
                    out.write(frame)
                    out.write("\r\n")
                    out.flush()
                } catch (e: Exception) { break }
            }
        } catch (e: Exception) {
            Log.d(TAG, "client ${socket.inetAddress.hostAddress} done: ${e.message}")
        } finally {
            clientCount.decrementAndGet()
            clients.remove(socket)
            runCatching { socket.close() }
        }
    }
}
