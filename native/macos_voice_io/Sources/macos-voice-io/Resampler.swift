// Resampler.swift — int16 PCM ↔ AVAudioPCMBuffer,AVAudioConverter 重采样。
//
// 设计目标:
//   - TTS 是 24 kHz int16 mono PCM(``AVAudioEngine`` 输出节点硬件格式多为 48 kHz float)
//   - capture 必须产 16 kHz mono int16(喂给 sherpa-onnx / WebRTC VAD)
//   - 双向都用 ``AVAudioConverter`` 一次性完成采样率 + 格式转换
//
// Python 永远只看到 16 kHz / mono / int16,所以上游不感知硬件采样率。

import AVFoundation
import Foundation

@available(macOS 13.0, *)
enum ResamplerError: Error, CustomStringConvertible {
    case converterCreationFailed
    case conversionFailed(OSStatus)
    case invalidInt16Frame
    case invalidSourceFormat

    var description: String {
        switch self {
        case .converterCreationFailed:
            return "AVAudioConverter 创建失败"
        case .conversionFailed(let s):
            return "AVAudioConverter.convert 失败,status=\(s)"
        case .invalidInt16Frame:
            return "int16 PCM 帧长度不是 2 的倍数"
        case .invalidSourceFormat:
            return "输入 buffer 的 format 不被 converter 接受"
        }
    }
}

@available(macOS 13.0, *)
enum Resampler {

    /// 标准的 16 kHz mono int16 AVAudioPCMBuffer 格式(给 capture 用)。
    static let captureFormat: AVAudioFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16_000,
        channels: 1,
        interleaved: true
    )!

    /// 把任意 ``AVAudioPCMBuffer`` 重采样并转换到 ``targetFormat``。
    /// 适用 16 kHz int16 ↔ 48 kHz float 之间的任意组合。
    static func resample(
        buffer source: AVAudioPCMBuffer,
        to targetFormat: AVAudioFormat
    ) throws -> AVAudioPCMBuffer {
        // 输入已经匹配:浅拷贝一份即可,避免无谓的重采样。
        if source.format.sampleRate == targetFormat.sampleRate
            && source.format.commonFormat == targetFormat.commonFormat
            && source.format.channelCount == targetFormat.channelCount
            && source.format.isInterleaved == targetFormat.isInterleaved {
            guard let out = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: source.frameLength
            ) else {
                throw ResamplerError.converterCreationFailed
            }
            out.frameLength = source.frameLength
            // 直接复制样本数据。audioBufferList 是 UnsafePointer<AudioBufferList>,
            // mBuffers 在 Swift 中是 tuple(取 .0)。
            let srcList = source.audioBufferList.pointee
            let dstList = out.audioBufferList.pointee
            if let src = srcList.mBuffers.mData, let dst = dstList.mBuffers.mData {
                let bytesPerFrame = Int(source.format.streamDescription.pointee.mBytesPerFrame)
                memcpy(dst, src, Int(source.frameLength) * bytesPerFrame)
            }
            return out
        }

        guard let converter = AVAudioConverter(from: source.format, to: targetFormat) else {
            throw ResamplerError.converterCreationFailed
        }

        // 计算输出容量：按比例 + 一个 chunk 的余量。
        let ratio = targetFormat.sampleRate / source.format.sampleRate
        let capacity = AVAudioFrameCount(Double(source.frameLength) * ratio + 1024)
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            throw ResamplerError.converterCreationFailed
        }

        var supplied = false
        var nsError: NSError?
        converter.convert(to: out, error: &nsError) { _, statusOut in
            if supplied {
                statusOut.pointee = .endOfStream
                return nil
            }
            supplied = true
            statusOut.pointee = .haveData
            return source
        }
        if let nsError = nsError {
            throw ResamplerError.conversionFailed(OSStatus(nsError.code))
        }
        return out
    }

    /// int16 mono PCM → AVAudioPCMBuffer。frame count = pcm.count / 2。
    /// 不重采样 —— 假设上游给的 sample rate 与 ``format.sampleRate`` 一致。
    static func int16PCMToBuffer(
        _ pcm: Data,
        format: AVAudioFormat
    ) throws -> AVAudioPCMBuffer {
        guard format.commonFormat == .pcmFormatInt16,
              format.isInterleaved,
              format.channelCount == 1 else {
            throw ResamplerError.invalidSourceFormat
        }
        guard pcm.count % 2 == 0 else {
            throw ResamplerError.invalidInt16Frame
        }
        let frames = AVAudioFrameCount(pcm.count / 2)
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw ResamplerError.converterCreationFailed
        }
        buf.frameLength = frames
        if frames == 0 { return buf }
        let dstList = buf.audioBufferList.pointee
        guard let dst = dstList.mBuffers.mData else {
            throw ResamplerError.converterCreationFailed
        }
        pcm.withUnsafeBytes { src in
            if let base = src.baseAddress {
                memcpy(dst, base, pcm.count)
            }
        }
        return buf
    }

    /// AVAudioPCMBuffer → int16 mono PCM Data。重采样到 16 kHz 后输出。
    static func bufferToInt16PCM(
        _ buffer: AVAudioPCMBuffer,
        targetSampleRate: Double = 16_000
    ) throws -> Data {
        let target = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: true
        )!
        let resampled = try resample(buffer: buffer, to: target)
        let srcList = resampled.audioBufferList.pointee
        guard let src = srcList.mBuffers.mData else { return Data() }
        let bytesPerFrame = Int(resampled.format.streamDescription.pointee.mBytesPerFrame)
        return Data(bytes: src, count: Int(resampled.frameLength) * bytesPerFrame)
    }
}