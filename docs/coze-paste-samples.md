# Coze 工作流粘贴样本与输出记录

本文件记录 Task 11 的 Coze 工作流验证过程。

**工作流结构**：开始节点 → 大模型节点 → 结束节点。

**大模型节点配置**：系统提示词取设计文档 `docs/superpowers/specs/2026-09-18-hearing-accessibility-sed-design.md` 第 7.3 节代码块内全文；温度 0.3。

**生成方式**：`python scripts/detect.py data/samples --threshold 0.1 --out data/out/coze_samples`

---

## 一级事件（火警类）

**源文件**：`data/samples/5-210612-A-37.wav`（5.0 秒）

### 粘贴的 JSON

```json
{
  "session_id": "5-210612-A-37",
  "audio_file": "5-210612-A-37.wav",
  "duration_sec": 5.0,
  "max_level": 1,
  "events": [
    {
      "event_id": "alarm_clock",
      "name_cn": "闹钟",
      "level": 2,
      "level_name": "注意",
      "confidence": 0.5472,
      "audio_set_labels": [
        "Alarm clock"
      ]
    },
    {
      "event_id": "beep",
      "name_cn": "电子提示音",
      "level": 2,
      "level_name": "注意",
      "confidence": 0.513,
      "audio_set_labels": [
        "Beep, bleep"
      ]
    },
    {
      "event_id": "buzzer",
      "name_cn": "蜂鸣器",
      "level": 2,
      "level_name": "注意",
      "confidence": 0.3878,
      "audio_set_labels": [
        "Buzzer"
      ]
    },
    {
      "event_id": "alarm",
      "name_cn": "警报器",
      "level": 1,
      "level_name": "危险",
      "confidence": 0.3794,
      "audio_set_labels": [
        "Alarm"
      ]
    },
    {
      "event_id": "fire_alarm",
      "name_cn": "火警",
      "level": 1,
      "level_name": "危险",
      "confidence": 0.3438,
      "audio_set_labels": [
        "Smoke detector, smoke alarm",
        "Fire alarm"
      ]
    }
  ]
}
```

### Coze 输出

_（待填入：把上面的 JSON 粘贴进 Coze 工作流，记录实际输出文案）_

### 观察

_（待填入：模型是否遵守了"不推测未出现声音""低置信度用可能"等规则）_

---

## 二级事件（狗叫）

**源文件**：`data/samples/1-100032-A-0.wav`（5.0 秒）

### 粘贴的 JSON

```json
{
  "session_id": "1-100032-A-0",
  "audio_file": "1-100032-A-0.wav",
  "duration_sec": 5.0,
  "max_level": 2,
  "events": [
    {
      "event_id": "dog_bark",
      "name_cn": "狗叫声",
      "level": 2,
      "level_name": "注意",
      "confidence": 0.1107,
      "audio_set_labels": [
        "Dog"
      ]
    }
  ]
}
```

### Coze 输出

_（待填入：把上面的 JSON 粘贴进 Coze 工作流，记录实际输出文案）_

### 观察

_（待填入：模型是否遵守了"不推测未出现声音""低置信度用可能"等规则）_

---

## 无事件

**源文件**：`data/samples/1-100210-A-36.wav`（5.0 秒）

### 粘贴的 JSON

```json
{
  "session_id": "1-100210-A-36",
  "audio_file": "1-100210-A-36.wav",
  "duration_sec": 5.0,
  "max_level": 0,
  "events": []
}
```

### Coze 输出

_（待填入：把上面的 JSON 粘贴进 Coze 工作流，记录实际输出文案）_

### 观察

_（待填入：模型是否遵守了"不推测未出现声音""低置信度用可能"等规则）_

---

---

## 平台现状记录（2026-09-18 实测）

尝试用浏览器自动化在新版 Coze 界面完成验证，未能成功，记录实际情况备查：

1. **界面已改版**。`coze.cn` 当前为对话式「扣子」工作台，项目打开后是会话视图，
   未找到经典工作流画布（开始节点 → 大模型节点 → 结束节点）的入口。
   课程模板所描述的工作流编辑器在当前版本中无法从该项目直接进入。

2. **协作模式要求 @Agent**。项目内提示「普通发言将同步给所有成员；如需让Agent
   执行任务，请在消息中 @对应Agent」。未 @ 时消息发送按钮处于禁用状态。

3. **自动化受限**。该页面使用闭合 Shadow DOM 渲染，且内容区为 Lexical
   富文本编辑器。所用浏览器自动化工具集不含真实键盘注入接口，通过 JS 合成的
   `insertText` 与 `paste` 事件虽能把文字写入 DOM，但编辑器内部状态不更新，
   发送按钮保持禁用。故无法在无人工介入下完成发送。

**结论**：场景三的实机验证需人工在 Coze 界面完成，或改由用户在经典工作流编辑器
中搭建。上文三份 JSON 可直接用于粘贴。
