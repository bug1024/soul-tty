// AudioEngine.swift — AVAudioEngine + setVoiceProcessingEnabled + tap + playback。
//
// Apple 官方约束:
// - ``AVAudioIONode.setVoiceProcessingEnabled(true)`` 主要用于 echo cancellation / VoIP。
// - voice-processing 必须在 ``engine`` 停止状态下配置,不能在运行时动态切换。
// - input/output 两端需要处于 voice-processing mode 才能互相配合(AEC 需要 playback reference)。
//
// 启动顺序:
//   1. 实例化 AVAudioEngine
//   2. engine.inputNode.setVoiceProcessingEnabled(true)  ← 必须停机状态
//   3. 接 playerNode → mainMixer
//   4. 在 inputNode 上 install tap(取 AEC-clean PCM)
//   5. engine.prepare() → engine.start()
//   6. playerNode.play()
//
// 播放路径:int16 PCM(int24 / 48 k 任意)→ resample 到 mainMixer 格式 → playerNode.scheduleBuffer
// 采集路径:inputNode tap(float 48k stereo)→ resample 到 16 k mono int16 → callback 上抛 Python

import AVFoundation
import Foundation

@available(macOS 13.0, *)
enum AudioEngineError: Error, CustomStringConvertible {
    case voiceProcessingUnavailable
    case voiceProcessingConfigFailed(String)
    case engineStartFailed(String)
    case tapInstallFailed
    case tapAlreadyInstalled
    case tapNotInstalled

    var description: String {
        switch self {
        case .voiceProcessingUnavailable:
            return "AVAudioEngine.setVoiceProcessingEnabled 不可用(可能不在 macOS 13+)"
        case .voiceProcessingConfigFailed(let s):
            return "voice-processing 配置失败:\(s)"
        case .engineStartFailed(let s):
            return "engine.start() 失败:\(s)"
        case .tapInstallFailed:
            return "inputNode.installTap 失败"
        case .tapAlreadyInstalled:
            return "tap 已存在,先 removeCaptureTap"
        case .tapNotInstalled:
            return "tap 未安装"
        }
    }
}

@available(macOS 13.0, *)
final class AudioEngine {

    /// 把 int16 mono PCM → engine 兼容 buffer 的目标格式(mainMixer 输出格式)。
    /// 大多数 macOS 默认设备是 48 kHz float32。
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let captureFormat: AVAudioFormat
    private let playbackFormat: AVAudioFormat

    private var captureHandler: ((Data, Int) -> Void)?
    private var tapInstalled = false
    private var _tapFrames: Int = 0
    private(set) var voiceProcessingActive = false

    init() throws {
        // capture 16 kHz int16 mono —— 给 Python 端的固定契约。
        self.captureFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16_000,
            channels: 1,
            interleaved: true
        )!

        // playback 用 mainMixer 输出格式(通常 48 kHz float)。
        self.playbackFormat = engine.mainMixerNode.outputFormat(forBus: 0)

        // 接 player → mainMixer(必须在 engine.start 之前)。
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: playbackFormat)

        // 关键:voice-processing 必须在 engine 停机状态下配置。
        // 启用失败 → 抛错,不要静默降级成普通模式(那就不是 AEC 了)。
        // SOUL_TTY_AEC=0 可以临时关掉,做对照实验。
        let aecEnabled = ProcessInfo.processInfo.environment["SOUL_TTY_AEC"] != "0"
        if aecEnabled {
            do {
                try engine.inputNode.setVoiceProcessingEnabled(true)
                voiceProcessingActive = true
                fputs("[macos-voice-io] voice-processing enabled\n", stderr)
            } catch {
                throw AudioEngineError.voiceProcessingConfigFailed(String(describing: error))
            }
        } else {
            fputs("[macos-voice-io] voice-processing DISABLED (SOUL_TTY_AEC=0)\n",
                  stderr)
        }
    }

    /// 注册 capture tap 的回调。tap 拿到的是 48 k float,与 ``captureFormat``
    /// 不匹配,因此内部统一经过 ``Resampler.bufferToInt16PCM`` 转 16 k int16 mono。
    func installCaptureTap(_ handler: @escaping (Data, Int) -> Void) throws {
        if tapInstalled {
            throw AudioEngineError.tapAlreadyInstalled
        }
        captureHandler = handler
        let input = engine.inputNode
        // 用 inputNode 自身的格式(可能不是 48 k,但一定是 input 物理格式)。
        let hwFormat = input.outputFormat(forBus: 0)
        // tap 缓冲大小:对应 30 ms @ 16 k ≈ 480 samples。物理 48 k 下 ≈ 1440 frames。
        let frames: AVAudioFrameCount = 1440
        input.installTap(onBus: 0, bufferSize: frames, format: hwFormat) { [weak self] buffer, _ in
            guard let self = self, let handler = self.captureHandler else { return }
            // 调试:第一帧 + 每 50 帧打一次 peak,看 inputNode 真拿到啥。
            self._tapFrames += 1
            if self._tapFrames == 1 || self._tapFrames % 50 == 0 {
                let chData = buffer.floatChannelData?[0]
                var peak: Float = 0
                if let ch = chData {
                    for i in 0..<Int(buffer.frameLength) {
                        let v = abs(ch[i])
                        if v > peak { peak = v }
                    }
                }
                fputs("[AudioEngine] tap frame=\(self._tapFrames) "
                      + "len=\(buffer.frameLength) peak=\(peak)\n", stderr)
            }
            do {
                let pcm = try Resampler.bufferToInt16PCM(buffer, targetSampleRate: 16_000)
                if !pcm.isEmpty {
                    handler(pcm, 16_000)
                }
            } catch {
                fputs("[AudioEngine] capture resample failed: \(error)\n", stderr)
            }
        }
        fputs("[AudioEngine] hwFormat=\(hwFormat) sampleRate=\(hwFormat.sampleRate) "
              + "channels=\(hwFormat.channelCount)\n", stderr)
        tapInstalled = true
    }

    func removeCaptureTap() throws {
        if !tapInstalled {
            throw AudioEngineError.tapNotInstalled
        }
        engine.inputNode.removeTap(onBus: 0)
        tapInstalled = false
        captureHandler = nil
    }

    /// 启动 AVAudioEngine + player。
    func start() throws {
        engine.prepare()
        do {
            try engine.start()
        } catch {
            throw AudioEngineError.engineStartFailed(String(describing: error))
        }
        // playerNode 启动播放(否则 schedule 进去的 buffer 不出声)。
        playerNode.play()
    }

    func stop() {
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        playerNode.stop()
        if engine.isRunning {
            engine.stop()
        }
    }

    /// 把 int16 mono PCM 推到扬声器。``sampleRate`` 任意(24k / 16k)，
    /// 内部 resample 到 ``playbackFormat``。
    /// 同步等待至 schedule 完成(调用方负责线程管理)。
    func writePlayback(_ pcm: Data, sampleRate: Int) throws {
        guard !pcm.isEmpty else { return }
        let srcFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: true
        )!
        let buffer = try Resampler.int16PCMToBuffer(pcm, format: srcFormat)
        let engineBuffer = try Resampler.resample(buffer: buffer, to: playbackFormat)
        playerNode.scheduleBuffer(engineBuffer, at: nil, options: [], completionHandler: nil)
    }

    /// 0.0 = 静音,1.0 = 原始音量,>1.0 可能爆音(系统会 clip)。
    func setPlaybackGain(_ value: Float) {
        engine.mainMixerNode.outputVolume = max(0.0, value)
    }
}