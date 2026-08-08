// main.swift — Soul-TTY macOS voice-processing helper entry point。
//
// 协议:
//   1. 创建 /tmp/soul-tty-audio.sock Unix domain socket,listen(1)
//   2. accept → 客户端连上
//   3. 等 Python 发 START → 启动 AudioEngine
//   4. capture tap 收到 AEC-clean PCM → 发 CAPTURE_PCM 回 Python
//   5. 主循环读 Python 消息:PLAYBACK_PCM / STOP / PING / SET_GAIN
//   6. STOP 收到或 socket 断开 → 关 engine,清 socket,退出
//
// 修复:capture tap callback 不再直接写 socket(可能导致 partial frame 破坏协议)。
// 改成 bounded queue + writer thread,确保每条消息完整写入。
//
// Python 协议镜像: src/soul_tty/audio/io/macos_voice.py。

import AVFoundation
import Foundation

#if canImport(Darwin)
import Darwin
#endif

@available(macOS 13.0, *)
final class Helper {
    let socketPath: String
    var listenFd: Int32 = -1
    var clientFd: Int32 = -1
    var engine: AudioEngine?
    var stats = Stats()
    let lock = NSLock()

    // capture queue + writer thread(修复:audio callback 不应直接写 socket)
    private let captureQueue: DispatchQueue
    private let captureQueueMax = 8
    private var captureBuffer: [Data] = []
    private let captureBufferLock = NSLock()

    init(socketPath: String) {
        self.socketPath = socketPath
        self.captureQueue = DispatchQueue(
            label: "soul-tty-capture-writer",
            qos: .userInitiated
        )
    }

    func run() throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: socketPath) {
            try fm.removeItem(atPath: socketPath)
        }

        listenFd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard listenFd >= 0 else { try throwHelperErr("socket") }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(socketPath.utf8) + [UInt8](repeating: 0,
            count: MemoryLayout<sockaddr_un>.size - 2 - socketPath.utf8.count)
        withUnsafeMutableBytes(of: &addr.sun_path) { dst in
            pathBytes.withUnsafeBytes { src in
                dst.copyBytes(from: UnsafeRawBufferPointer(start: src.baseAddress,
                                                          count: pathBytes.count))
            }
        }
        let bindOK = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                bind(listenFd, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindOK >= 0 else { try throwHelperErr("bind") }
        guard listen(listenFd, 1) >= 0 else { try throwHelperErr("listen") }

        fputs("[macos-voice-io] listening at \(socketPath)\n", stderr)
        fflush(stderr)

        clientFd = accept(listenFd, nil, nil)
        guard clientFd >= 0 else { try throwHelperErr("accept") }
        fputs("[macos-voice-io] client connected\n", stderr)
        fflush(stderr)

        // 启 writer thread(capture tap 回调推送,writer 线程 blocking write)
        self.captureQueue.async { [weak self] in
            self?.captureWriterLoop()
        }

        // 阻塞消息循环。STOP / EOF → 退出。
        try messageLoop()

        // 清理
        if let engine = self.engine {
            engine.stop()
        }
        close(clientFd)
        close(listenFd)
        try? fm.removeItem(atPath: socketPath)
        fputs("[macos-voice-io] shutdown complete\n", stderr)
    }

    // MARK: - Capture writer thread

    /// writer 线程:从 bounded buffer 拉 PCM 并通过 blocking write 写入 socket。
    /// 确保每条消息完整发送,不会破坏协议 framing。
    private func captureWriterLoop() {
        while true {
            let pcm: Data
            captureBufferLock.lock()
            if captureBuffer.isEmpty {
                captureBufferLock.unlock()
                // 空队列:稍等再试
                usleep(5_000) // 5ms
                continue
            }
            pcm = captureBuffer.removeFirst()
            captureBufferLock.unlock()

            // blocking writeAll
            var sent = 0
            let data = pcm
            while sent < data.count {
                let n = data.withUnsafeBytes { raw -> Int in
                    guard let base = raw.baseAddress else { return 0 }
                    return send(clientFd, base.advanced(by: sent), data.count - sent, 0)
                }
                if n < 0 {
                    if errno == EINTR { continue }
                    // 写入失败:退出 writer 线程
                    fputs("[macos-voice-io] capture writer write error: \(String(cString: strerror(errno)))\n", stderr)
                    return
                }
                sent += n
            }
            lock.lock()
            stats.captureFrames += 1
            lock.unlock()
        }
    }

    // MARK: - Message loop

    /// 主循环:读消息 → 分发 → 处理 capture / playback / ping / stop。
    func messageLoop() throws {
        while true {
            let msg: Message
            do {
                msg = try Message.read(from: clientFd)
            } catch let proto as ProtocolError {
                // EOF 时干净退出,其它 protocol 错误上报。
                if case .unexpectedEOF = proto {
                    fputs("[macos-voice-io] client disconnected\n", stderr)
                    return
                }
                try sendError("protocol: \(proto)")
                throw proto
            }

            switch msg.type {
            case .start:
                try handleStart()
            case .stop:
                fputs("[macos-voice-io] STOP received\n", stderr)
                return
            case .ping:
                try Message(type: .pong, payload: Data()).write(to: clientFd)
                stats.pingCount += 1
            case .playbackPCM:
                try handlePlayback(msg.payload)
            case .capturePCM, .pong, .stats, .error:
                // 客户端不应发送这些,收到 → 上报 + 继续
                try sendError("unexpected message type 0x\(String(msg.type.rawValue, radix: 16))")
            }
        }
    }

    func handleStart() throws {
        // 幂等:START 来两次,第二次 no-op。
        if engine != nil {
            fputs("[macos-voice-io] START already applied, ignoring\n", stderr)
            return
        }
        let eng = try AudioEngine()
        try eng.installCaptureTap { [weak self] pcm, sr in
            self?.enqueueCapture(pcm: pcm, sampleRate: sr)
        }
        try eng.start()
        self.engine = eng

        // 启动静音检测:给 engine 1.5 秒收集 tap 帧,如果 peak 始终 < 阈值则失败
        let checkStartup = { [weak self] in
            guard let self = self, let eng = self.engine else { return }
            if eng.isStartupSilent {
                fputs("[macos-voice-io] FATAL: microphone capture is silent "
                      + "(peak=\(eng.startupPeak) after 30 frames)\n", stderr)
                // 用 error 消息通知 Python,不自己 exit(让 Python 做决策)
                try? self.sendError("microphone_capture_silent: "
                    + "启动后前 30 帧 peak=\(eng.startupPeak),怀疑麦克风静音/权限/设备占用")
            }
        }
        // 延迟 1.5 秒检查(30 帧 @ 30ms ≈ 1s,加 0.5 余量)
        DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) {
            checkStartup()
        }

        fputs("[macos-voice-io] AudioEngine started, voice-processing=\(eng.voiceProcessingActive)\n",
              stderr)
        fflush(stderr)
    }

    /// Capture tap callback:把 PCM 推到 bounded buffer。
    /// 不直接写 socket(避免 audio render thread 阻塞 + partial frame 破坏协议)。
    func enqueueCapture(pcm: Data, sampleRate: Int) {
        // 头 4 bytes 写 sampleRate(留作将来多采样率扩展,commit 05 先固定 16k)。
        var payload = Data(capacity: 4 + pcm.count)
        var sr = UInt32(sampleRate).bigEndian
        withUnsafeBytes(of: &sr) { payload.append(contentsOf: $0) }
        payload.append(pcm)

        let encoded = Message(type: .capturePCM, payload: payload).encode()

        captureBufferLock.lock()
        if captureBuffer.count < captureQueueMax {
            captureBuffer.append(encoded)
        } else {
            // 队列满:丢最旧
            captureBuffer.removeFirst()
            captureBuffer.append(encoded)
            lock.lock()
            stats.droppedCaptureFrames += 1
            lock.unlock()
        }
        captureBufferLock.unlock()
    }

    func handlePlayback(_ payload: Data) throws {
        guard let engine = self.engine else {
            try sendError("playback before START")
            return
        }
        // payload: [4 bytes uint32 BE sampleRate][int16 PCM]
        guard payload.count >= 4 else {
            try sendError("playback payload too short")
            return
        }
        let srBytes = payload.prefix(4)
        let sampleRate: UInt32 = (UInt32(srBytes[srBytes.startIndex]) << 24)
                                | (UInt32(srBytes[srBytes.startIndex + 1]) << 16)
                                | (UInt32(srBytes[srBytes.startIndex + 2]) << 8)
                                |  UInt32(srBytes[srBytes.startIndex + 3])
        let pcm = payload.subdata(in: 4..<payload.count)
        do {
            try engine.writePlayback(pcm, sampleRate: Int(sampleRate))
            lock.lock()
            stats.playbackFrames += 1
            stats.playbackBytes += UInt64(pcm.count)
            lock.unlock()
        } catch {
            try sendError("writePlayback: \(error)")
        }
    }

    /// payload: [1 byte type byte][varies]。SET_GAIN=0x10 in payload(预留,本 commit 不接)。
    /// 当前 commit 仅实现播放 / capture,所以 SET_GAIN 不解析;留空函数。
    func sendError(_ text: String) throws {
        let msg = Message(type: .error, payload: Data(text.utf8))
        try msg.write(to: clientFd)
    }
}

@available(macOS 13.0, *)
struct Stats {
    var captureFrames: UInt64 = 0
    var playbackFrames: UInt64 = 0
    var playbackBytes: UInt64 = 0
    var droppedCaptureFrames: UInt64 = 0
    var pingCount: UInt64 = 0

    var json: String {
        return """
        {"capture_frames":\(captureFrames),"playback_frames":\(playbackFrames),\
        "playback_bytes":\(playbackBytes),"dropped_capture_frames":\(droppedCaptureFrames),\
        "ping_count":\(pingCount)}
        """
    }
}

@available(macOS 13.0, *)
func throwHelperErr(_ op: String) throws -> Never {
    fputs("[macos-voice-io] \(op) failed: \(String(cString: strerror(errno)))\n",
          stderr)
    throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
}

@available(macOS 13.0, *)
func main() throws {
    let path = ProcessInfo.processInfo.environment["SOUL_TTY_AUDIO_SOCK"]
        ?? "/tmp/soul-tty-audio.sock"
    let helper = Helper(socketPath: path)
    try helper.run()
}

if #available(macOS 13.0, *) {
    do {
        try main()
    } catch {
        fputs("[macos-voice-io] fatal: \(error)\n", stderr)
        exit(1)
    }
} else {
    fputs("[macos-voice-io] requires macOS 13+\n", stderr)
    exit(1)
}