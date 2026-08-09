# Agency MVP

Soul-TTY 的 Conversation Brain 只负责“决定说话之后，具体说什么”。在它之前，
Response Policy 根据持续 Need 状态决定本轮采用：`ANSWER`、`SHORT_REPLY`、
`SILENCE`、`CHANGE_TOPIC` 或 `ASK`。

## 约束

- Policy 是本地常数时间逻辑，不新增 LLM 往返。
- Need 写盘走后台阻塞队列；失败降级为内存状态，不阻塞语音主链路。
- 明确问题、任务、制止词和需要关怀的表达不可沉默或转题。
- Silence 只在低交流倾向、高独处需求、低风险输入和冷却条件同时满足时发生，
  且不可连续两次；UI 明确表现为“听见但选择安静”，区别于故障。
- Response 指令只临时进入本轮 LLM 请求，不写入正式对话历史。
- `desire_to_talk`、`desire_for_company` 和 `solitude_need` 相互独立。

## 后续


下一阶段由 OpenLoop Memory 提供真正的 unresolved context，再通过带冷却和概率的
Spontaneous Recall 交给 Response Policy。InnerThread 与主动发起交流在此之后实现，
避免尚未建立护栏时让角色随机抢话。
