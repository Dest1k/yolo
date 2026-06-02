package com.destik.yolodesktop

import java.awt.Graphics2D
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import javax.imageio.ImageIO

class MjpegServer(val port: Int = 8080) {

    private val latestJpeg = AtomicReference<ByteArray>(null)
    private var serverSocket: ServerSocket? = null
    @Volatile var running = false; private set
    private val clients = CopyOnWriteArrayList<Socket>()
    private val clientCount = AtomicInteger(0)
    private val acceptThread = Executors.newSingleThreadExecutor()
    private val clientPool = Executors.newCachedThreadPool()

    fun start() {
        if (running) return
        running = true
        serverSocket = ServerSocket(port).also { it.reuseAddress = true }
        acceptThread.execute {
            while (running) {
                try {
                    val client = serverSocket!!.accept()
                    client.soTimeout = 5_000
                    clients.add(client)
                    clientCount.incrementAndGet()
                    clientPool.execute { serveClient(client) }
                } catch (e: Exception) {
                    if (running) System.err.println("MJPEG accept: ${e.message}")
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
    }

    fun hasClients(): Boolean = clientCount.get() > 0
    fun clientCount(): Int = clientCount.get()

    fun pushFrame(image: BufferedImage, quality: Int = 75) {
        if (!running || !hasClients()) return
        val baos   = ByteArrayOutputStream(64 * 1024)
        val iter   = ImageIO.getImageWritersByFormatName("jpeg")
        if (!iter.hasNext()) return
        val writer = iter.next()
        try {
            val param = writer.defaultWriteParam.apply {
                compressionMode    = javax.imageio.ImageWriteParam.MODE_EXPLICIT
                compressionQuality = quality / 100f
            }
            val ios = ImageIO.createImageOutputStream(baos)
            writer.output = ios
            writer.write(null, javax.imageio.IIOImage(image, null, null), param)
            ios.close()
        } finally {
            writer.dispose()
        }
        latestJpeg.set(baos.toByteArray())
    }

    private fun serveClient(socket: Socket) {
        try {
            val inp = socket.getInputStream().bufferedReader()
            var line = inp.readLine()
            while (!line.isNullOrEmpty()) line = inp.readLine()

            val out = socket.getOutputStream()
            out.write(
                ("HTTP/1.1 200 OK\r\n" +
                 "Content-Type: multipart/x-mixed-replace; boundary=--mjpeg\r\n" +
                 "Cache-Control: no-store, no-cache, must-revalidate\r\n" +
                 "Pragma: no-cache\r\n" +
                 "Connection: close\r\n\r\n").toByteArray()
            )

            var lastSent: ByteArray? = null
            while (running && !socket.isClosed) {
                val frame = latestJpeg.get()
                if (frame == null || frame === lastSent) { Thread.sleep(5); continue }
                lastSent = frame
                try {
                    out.write("----mjpeg\r\nContent-Type: image/jpeg\r\nContent-Length: ${frame.size}\r\n\r\n".toByteArray())
                    out.write(frame)
                    out.write("\r\n".toByteArray())
                    out.flush()
                } catch (e: Exception) { break }
            }
        } catch (e: Exception) {
            // client disconnected
        } finally {
            clientCount.decrementAndGet()
            clients.remove(socket)
            runCatching { socket.close() }
        }
    }
}
