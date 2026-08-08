// swift-tools-version:5.9
//
// macOS voice-processing AEC helper for Soul-TTY.
//
// commit 03 阶段:只暴露架构与 IPC 协议骨架,真正 AVAudioEngine
// 接线在 commit 05+ 完成。AVAudioEngine 的 voice-processing 模式
// 需要 macOS 13+(AVAudioIONode.setVoiceProcessingEnabled)。
//

import PackageDescription

let package = Package(
    name: "macos-voice-io",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "macos-voice-io",
            path: "Sources/macos-voice-io"
        )
    ]
)