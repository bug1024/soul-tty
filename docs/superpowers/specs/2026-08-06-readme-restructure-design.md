# Soul-TTY README 重构设计

## 目标

将 README 从“设计过程与实现细节的集合”重组为一条清晰的开源项目叙事：先说明 Soul-TTY 是什么、为什么不同，再说明 Agent 架构与状态模型，最后提供技术实现、安装和配置参考。

本次只修改文档，不修改产品代码、命令、配置项或功能状态。

## 定位

Soul-TTY 的定位不止“陪伴”，还包含“本地 Agent 架构实验”。产品简介必须同时体现两者：

> Soul-TTY is a local AI companion and agent architecture experiment.
>
> Soul-TTY 是一个运行在终端中的本地 AI 伙伴，也是一个探索状态化 Agent 架构的开源项目。

Design Philosophy 之后给出一句总纲，作为整份 README 的核心：

> Soul-TTY separates what an agent says from what an agent becomes.
>
> Soul-TTY 将“即时回复”和“持续成长”拆分，让 Agent 不只是生成文本，而是拥有持续演化的内部状态。

## 信息架构

README 采用以下顺序：

1. Title + Positioning：项目标题、英文/中文一句话定位与产品简介
2. Features：当前能力与状态
3. Design Philosophy：Real-time first、State over prompt、Separate talking and thinking、Local first
4. Agent Architecture：Conversation Brain、Reflection Brain、State Layer、Expression Layer
5. Emotion System：Concept、Model、Evolution、Integration
6. Bond System：Concept、Model、Evolution，以及与 Emotion 的边界
7. Outfit & Modes：人格模式与换装行为
8. Technical Implementation：Audio、ASR、LLM、TTS、UI、Reflection 的完整链路
9. Project Structure
10. Installation
11. Configuration：怎么配置（环境变量、人格 YAML 扩展）
12. Development：怎么开发（旁路调试、子命令、测试）
13. Interaction Reference：用户怎么操作
14. Roadmap
15. License & Thanks

Configuration / Development / Interaction Reference 三章目标不同，必须拆开，不合并为单一“Development / 调试与交互参考”。

## 内容处理规则

- 保留现有安装、配置、项目结构、技术链路、交互、人格扩展和功能状态信息。
- 将 `Design Decisions` 中仍然对用户有价值的稳定设计结论吸收到 `Design Philosophy`、`Agent Architecture` 和状态系统章节。
- 删除“统一称 Bond”“记录几个不会轻易改回去”等面向作者的命名规范、过程记录和未来自我提醒。
- 合并重复的 Roadmap，只保留一份，并按 Memory Layer、Expression Layer、Interaction Layer 等能力层组织。Roadmap 只列能力方向，不写“支持 xxx / 增加 xxx / 优化 xxx”式的开发 TODO。
- Emotion 和 Bond 采用渐进式信息层级：先解释概念和产品作用，再展示模型与演化方式，最后说明 Prompt/TTS/Avatar 或后台评估等技术集成。
- 将 EMA、delta cap、idle decay、worker、限频、接口路径等实现细节保留在技术细节或对应的 Technical Details 小节，不在章节开头阻断产品叙事。
- 统一术语和语气：产品叙事可保留适度拟人化表达；技术章节优先使用 Soul-TTY、the agent 或系统组件名称，避免无必要地反复使用“她”。
- 不虚构尚未实现的能力；保留已实现、实验中、计划中的准确状态标记。

## 成功标准

- 新读者在阅读前几屏后能理解 Soul-TTY 的产品定位及其区别于普通 ChatGPT CLI 的核心理念。
- 读者随后能顺着架构、状态系统和技术链路理解“它为什么像一个 Agent”。
- 开发者仍能在同一份 README 中找到运行、配置和实现参考，而无需依赖被隐藏的附属文档。
- README 不再出现重复 Roadmap、作者内部备注或明显的开发过程讨论。
- 所有命令、环境变量、路径和功能状态与当前项目保持一致。
