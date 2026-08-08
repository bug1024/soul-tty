// AudioEngine.swift — AVAudioEngine + setVoiceProcessingEnabled + tap + playback。
//
// Apple 官方约束:
// - ``AVAudioIONode.setVoiceProcessingEnabled(true)`` 主要用于 echo cancellation / VoIP。
// - voice-processing 必须在 ``engine`` 停止状态下配置,不能在运行时动态切换。
// - input/output 两端需要处于 voice-processing mode 才能互相配合(AEC 需要 playback reference)。
//
// 启动顺序(修复:VPIO 必须在 format/graph 查询之前开启):
//   1. 实例化 AVAudioEngine
//   2. 检查麦克风权限
//   3. engine.inputNode.setVoiceProcessingEnabled(true)  ← 必须停机状态,在 I/O 节点切换之前
//   4. VPIO 开启后再查 input/output format(voice-processing 可能改变 format)
//   5. 接 playerNode → mainMixer
//   6. 在 inputNode 上 install tap(取 AEC-clean PCM)
//   7. engine.prepare() → engine.start()
//   8. playerNode.play()
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
    case microphonePermissionDenied

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
        case .microphonePermissionDenied:
            return "麦克风权限被拒绝,请在系统设置 → 隐私与安全性 → 麦克风 中授权"
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
    private var playbackFormat: AVAudioFormat!

    private var captureHandler: ((Data, Int) -> Void)?
    private var tapInstalled = false
    private(set) var voiceProcessingActive = false

    // 启动静音检测
    private(set) var startupPeak: Float = 0
    private var startupFrames = 0
    private let startupSilentThreshold: Float = 0.00001
    private let startupMaxFrames = 30

    init() throws {
        // 1) 检查麦克风权限
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            fputs("[AudioEngine] microphone permission=authorized\n", stderr)
        case .denied, .restricted:
            fputs("[AudioEngine] microphone permission=denied/restricted\n", stderr)
            throw AudioEngineError.microphonePermissionDenied
        case .notDetermined:
            fputs("[AudioEngine] microphone permission=notDetermined, requesting...\n", stderr)
            // 同步请求权限(用户可能在启动时看到弹窗)
            let sem = DispatchSemaphore(value: 0)
            var granted = false
            AVCaptureDevice.requestAccess(for: .audio) { ok in
                granted = ok
                sem.signal()
            }
            sem.wait()
            if !granted {
                throw AudioEngineError.microphonePermissionDenied
            }
            fputs("[AudioEngine] microphone permission granted\n", stderr)
        @unknown default:
            break
        }

        // capture 16 kHz int16 mono —— 给 Python 端的固定契约。
        self.captureFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16_000,
            channels: 1,
            interleaved: true
        )!

        // 2) 先开启 voice-processing(必须在 engine 停机、format 查询之前)
        let aecEnabled = ProcessInfo.processInfo.environment["SOUL_TTY_AEC"] != "0"
        if aecEnabled {
            let input = engine.inputNode
            do {
                try input.setVoiceProcessingEnabled(true)

                guard input.isVoiceProcessingEnabled else {
                    throw AudioEngineError.voiceProcessingConfigFailed(
                        "input voice-processing not active after enable"
                    )
                }

                // 确保 input 不 mute(默认可能 mute 了)
                input.isVoiceProcessingInputMuted = false

                voiceProcessingActive = true
                fputs("[AudioEngine] voice-processing enabled\n", stderr)
            } catch {
                throw AudioEngineError.voiceProcessingConfigFailed(String(describing: error))
            }
        } else {
            fputs("[AudioEngine] voice-processing DISABLED (SOUL_TTY_AEC=0)\n", stderr)
        }

        // 3) VPIO 切换完成后再查 format(VPIO 可能改变 I/O format)
        let input = engine.inputNode
        let output = engine.outputNode
        playbackFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        let inputFormat = input.outputFormat(forBus: 0)
        let outputFormat = output.outputFormat(forBus: 0)
        fputs("[AudioEngine] post-VPIO input=\(inputFormat) output=\(outputFormat)\n", stderr)

        // 4) 最后才构造播放 graph
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: playbackFormat)
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
            // 同时做启动静音检测(前 30 帧累计 peak)。
            let chData = buffer.floatChannelData?[0]
            var peak: Float = 0
            if let ch = chData {
                for i in 0..<Int(buffer.frameLength) {
                    let v = abs(ch[i])
                    if v > peak { peak = v }
                }
            }
            self._trackStartupPeak(peak, frameLength: Int(buffer.frameLength))

            self._tapFrames += 1
            if self._tapFrames == 1 || self._tapFrames % 50 == 0 {
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

    /// 启动静音检测:累计前 30 个 tap 帧的 peak。
    /// 如果启动后 1.5 秒 peak 始终 < startupSilentThreshold,上层
    /// (main.swift handleStart) 应抛 "microphone_capture_silent"。
    var isStartupSilent: Bool {
        return startupFrames >= startupMaxFrames && startupPeak < startupSilentThreshold
    }

    private var _tapFrames: Int = 0
    private var _startupPeakLock = NSLock()

    private func _trackStartupPeak(_ peak: Float, frameLength: Int) {
        _startupPeakLock.lock()
        startupFrames += 1
        if peak > startupPeak { startupPeak = peak }
        let silent = startupFrames >= startupMaxFrames && startupPeak < startupSilentThreshold
        _startupPeakLock.unlock()
        // 达到诊断阈值时打一条日志(只打一次)
        if silent && startupFrames == startupMaxFrames {
            fputs("[AudioEngine] WARNING: startup silent detection triggered "
                  + "(peak=\(startupPeak) < threshold=\(startupSilentThreshold) "
                  + "after \(startupMaxFrames) frames)\n", stderr)
        }
    }
}