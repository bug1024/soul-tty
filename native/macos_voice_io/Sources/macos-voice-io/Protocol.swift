// Protocol.swift — Soul-TTY ↔ macos-voice-io IPC 协议。
//
// 线协议:每条消息 = [1 byte type][4 bytes uint32 big-endian payload_len][payload]
//
// 消息类型:
//   0x01 PLAYBACK_PCM  (Python → Swift) payload = int16 PCM (任意 sample rate)
//   0x02 CAPTURE_PCM   (Swift → Python) payload = int16 mono PCM (16k AEC-clean)
//   0x03 START         (Python → Swift) payload = "" 空启动
//   0x04 STOP          (Python → Swift) payload = "" 优雅停止
//   0x05 PING          (双向)             payload = "" 返回 PONG (0x06)
//   0x06 PONG          (双向)             payload = ""
//   0x07 STATS         (Swift → Python) payload = JSON stats 文本
//   0x08 ERROR         (Swift → Python) payload = 错误文本
//   0x09 PLAYBACK_FLUSH (Python → Swift) payload = "" 清空已排队播放 buffer
//   0x0A PLAYBACK_DRAINED (Swift → Python) payload = "" 扬声器已播完所有 PCM
//
// Python 镜像见 src/soul_tty/audio/io/macos_voice.py。修改任何字段都要同步两端。

import Foundation

@available(macOS 13.0, *)
enum MessageType: UInt8 {
    case playbackPCM = 0x01
    case capturePCM  = 0x02
    case start       = 0x03
    case stop        = 0x04
    case ping        = 0x05
    case pong        = 0x06
    case stats       = 0x07
    case error       = 0x08
    case playbackFlush   = 0x09
    case playbackDrained = 0x0A
}

@available(macOS 13.0, *)
enum ProtocolError: Error, CustomStringConvertible {
    case truncated
    case unknownType(UInt8)
    case unexpectedEOF
    case payloadTooLarge(UInt32)

    var description: String {
        switch self {
        case .truncated: return "frame too short for header"
        case .unknownType(let b): return String(format: "unknown message type 0x%02x", b)
        case .unexpectedEOF: return "socket closed mid-frame"
        case .payloadTooLarge(let n): return "payload length \(n) exceeds limit"
        }
    }
}

/// 协议层消息上限(4 MiB)。正常的 30 ms / 16 kHz 帧 ≈ 960 B;
/// 4 MiB 足以容纳一条整句 TTS 长 PCM,远超实时需求。
@available(macOS 13.0, *)
let kMaxPayload: UInt32 = 4 * 1024 * 1024

@available(macOS 13.0, *)
struct Message {
    let type: MessageType
    let payload: Data

    /// 序列化为字节序列。header 5 字节 + payload。
    func encode() -> Data {
        var data = Data(capacity: 5 + payload.count)
        data.append(type.rawValue)
        var len = UInt32(payload.count).bigEndian
        withUnsafeBytes(of: &len) { data.append(contentsOf: $0) }
        data.append(payload)
        return data
    }

    /// 阻塞读一条消息。EOF / 截断 → 抛错,调用方负责关 socket。
    static func read(from fd: Int32) throws -> Message {
        var header = [UInt8](repeating: 0, count: 5)
        if !readExactly(fd: fd, into: &header) {
            throw ProtocolError.unexpectedEOF
        }
        let typeByte = header[0]
        guard let msgType = MessageType(rawValue: typeByte) else {
            throw ProtocolError.unknownType(typeByte)
        }
        let len = (UInt32(header[1]) << 24)
                | (UInt32(header[2]) << 16)
                | (UInt32(header[3]) << 8)
                |  UInt32(header[4])
        if len > kMaxPayload {
            throw ProtocolError.payloadTooLarge(len)
        }
        var payload = Data(count: Int(len))
        if len > 0 {
            let ok = payload.withUnsafeMutableBytes { raw -> Bool in
                guard let base = raw.baseAddress else { return false }
                return readExactly(fd: fd, into: base, count: Int(len))
            }
            if !ok {
                throw ProtocolError.unexpectedEOF
            }
        }
        return Message(type: msgType, payload: payload)
    }

    /// 写到 socket。返回写入字节数,出错抛 errno。
    @discardableResult
    func write(to fd: Int32) throws -> Int {
        let encoded = encode()
        return try encoded.withUnsafeBytes { raw -> Int in
            guard let base = raw.baseAddress else { return 0 }
            return try writeExactly(fd: fd, ptr: base, count: encoded.count)
        }
    }
}

/// 把 buf 写满 count 字节或抛错。短写 → 循环重试。
@available(macOS 13.0, *)
func writeExactly(fd: Int32, ptr: UnsafeRawPointer, count: Int) throws -> Int {
    var written = 0
    while written < count {
        let n = write(fd, ptr.advanced(by: written), count - written)
        if n < 0 {
            if errno == EINTR { continue }
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(errno))])
        }
        if n == 0 { break }  // 不会发生,但保留保护
        written += n
    }
    return written
}

/// 把 buf 读满 count 字节;返回 false 表示 EOF。
@available(macOS 13.0, *)
func readExactly(fd: Int32, into ptr: UnsafeMutableRawPointer, count: Int) -> Bool {
    var read = 0
    while read < count {
        let n = recv(fd, ptr.advanced(by: read), count - read, 0)
        if n < 0 {
            if errno == EINTR { continue }
            return false
        }
        if n == 0 { return false }  // peer closed
        read += n
    }
    return true
}

/// Array 形式的小 helper,用于 header 这种短缓冲。
@available(macOS 13.0, *)
func readExactly(fd: Int32, into buf: inout [UInt8]) -> Bool {
    return buf.withUnsafeMutableBytes { raw -> Bool in
        guard let base = raw.baseAddress else { return false }
        return readExactly(fd: fd, into: base, count: raw.count)
    }
}