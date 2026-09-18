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

