# 听障无障碍环境声音提示智能体 —— 设计文档

日期：2026-09-18
课程：《数字音频处理》
交付物：一份实验报告（.docx）+ 可运行的本地识别代码 + Coze 侧工作流设计

---

## 1. 实验目标

构建一个面向听障人群的环境声音提示系统：本地识别环境中的关键声音事件，
输出结构化事件数据，再由 Coze 智能体生成分级中文提示文案。

本实验**不做**语音识别（ASR），做的是**声学事件检测**（Sound Event Detection, SED）——
识别门铃、火警、婴儿哭、玻璃破碎这类非语音的环境声音。这是与参考报告
（音视频转写、TTS、声音克隆）的本质区别，也是本实验选题的立足点。

## 2. 设计约束（来自需求确认）

| 约束 | 取值 | 影响 |
|---|---|---|
| 交付形态 | 本地代码 + Coze 智能体混合 | 两条链路，JSON 衔接 |
| 识别实现 | 直接调用预训练模型，不训练 | 零样本类别映射路线 |
| 系统形态 | 离线批处理 + 指标评估 | 不处理实时流 |
| 数据集 | 不下载 UrbanSound8K（完整包 5.6 GB） | 改用 ESC-50 按需抽样，总量 MB 级 |
| 运行程度 | 轻量跑通，不装 torch | 必须走 ONNX Runtime |
| Coze 职责 | 仅 LLM 提示文案生成 | 不含 TTS、不含知识库 |
| 链路衔接 | 手动粘贴 JSON 进 Coze 工作流 | 不调 Coze API |

## 3. 关键技术决策：为什么用 YAMNet ONNX

原始需求要求"调用预训练模型"（排除自训分类器），同时又"不装 torch"
（排除 PANNs/torchlibrosa 路线）。这两条约束联立后只剩一条可行路径：**ONNX 运行时**。

选定模型：`niobures/YAMNet` 仓库的 `yamnetonnx/yamnet.onnx`

- 体积 16.1 MB，自包含（权重内嵌）
- 输入：原始 float32 单声道波形 @ 16 kHz（**预处理在计算图内部**，无需自己实现梅尔滤波器组）
- 输出：`[num_frames, 521]` AudioSet 得分矩阵
- 类别表：同仓库 `yamnetonnx/yamnet_class_map.csv`（14 KB）
- 依赖：仅 `onnxruntime` + `numpy`

网络约束（本机实测）：

| 主机 | 状态 |
|---|---|
| `huggingface.co` | **不通（HTTP 000）** |
| `hf-mirror.com` | 可达（HTTP 200）——模型下载必须走镜像 |
| `raw.githubusercontent.com` | 可达（HTTP 200）——ESC-50 抽样走这里 |
| `pypi.org` | 可达（HTTP 200） |
| `github.com` API | 可达（HTTP 200） |
| `zenodo.org` | 可达（HTTP 200）——本实验不需要 |

已确认的本机环境：Python 3.13.3、numpy 2.4.6、scipy 1.17.1、pandas 3.0.3、
matplotlib 3.10.9、requests 2.34.2 均已安装。**唯一需要新增安装的是 `onnxruntime`。**

## 4. 系统架构

### 4.1 链路 A：本地识别（Python）

```
环境声音 WAV
  → 预处理（重采样 16 kHz / 单声道 / 幅值归一化）
  → YAMNet ONNX 推理 → [n_frames, 521] 得分矩阵
  → 帧级聚合（均值 + 最大值）→ 521 类逐类得分
  → 映射层：AudioSet 521 类 → 听障关注事件（中文名 + 紧急等级）
  → 多帧一致性去抖 + 置信度阈值
  → 事件 JSON
```

### 4.2 链路 B：Coze 智能体（手动衔接）

```
事件 JSON（人工粘贴）→ Coze 工作流「大模型节点」→ 分级中文提示文案
```

### 4.3 组件职责

每个模块单一职责、可独立测试：

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `audio_io.py` | 读取音频、重采样至 16 kHz、转单声道、幅值归一化 | 音频文件路径 | 一维 float32 数组 |
| `yamnet_runner.py` | 加载 ONNX 会话、执行推理、帧级聚合 | float32 波形 | `(521,)` 得分向量 + 帧级矩阵 |
| `event_mapping.py` | 类别映射表、紧急分级、多类归并、去抖决策 | 521 类得分 | 事件列表 |
| `detect.py` | 串联主流程、输出 JSON | 音频文件 / 目录 | 事件 JSON |
| `make_test_signals.py` | 生成合成测试信号 | 参数 | WAV 文件 |
| `evaluate.py` | 指标统计、混淆矩阵、图表绘制 | 带标注样本目录 | 指标表 + 图 |

## 5. 映射层设计（本实验技术核心）

零样本路线的全部价值集中在映射层。需解决三个问题：

### 5.1 紧急度分级

| 等级 | 含义 | 提示行为 | AudioSet 类别举例 |
|---|---|---|---|
| 1 | 危险，立即提示 | 强提示 | Smoke detector/smoke alarm、Fire alarm、Siren、Gunshot/gunfire、Glass、Shatter、Screaming、Car alarm、Explosion、Civil defense siren |
| 2 | 注意，需要知晓 | 常规提示 | Doorbell、Knock、Telephone bell ringing、Baby cry/infant cry、Dog、Bark、Car horn、Vehicle horn、Buzzer、Alarm clock、Kettle、Water tap/faucet、Dishes |
| 3 | 环境音，不提示 | 静默 | Music、Speech、Silence、Air conditioning、Traffic noise、Static |

等级 3 显式列出而非"未命中即忽略"，是为了区分「已知可忽略」与「未知未覆盖」——
后者是覆盖盲区，需要单独统计（见 5.3）。

### 5.2 多类归并

AudioSet 本体中同一物理事件常分布在多个类。例如 `Smoke detector, smoke alarm`
与 `Fire alarm` 属同一事件族；`Siren` / `Civil defense siren` / `Police car (siren)`
同属警报族。若不归并，同一声源会触发多条重复提示。

归并规则：在映射表中为每个 AudioSet 类别指定 `event_id`，多个类别可共享同一 `event_id`；
同一 `event_id` 下取最高置信度作为该事件置信度。

### 5.3 覆盖盲区统计

统计三类数量，作为报告第五章的核心发现：

1. 521 类中已建立映射的类别数（覆盖宽度）
2. 映射到等级 1 / 2 / 3 的类别数分布
3. **听障安全场景需要、但 AudioSet 本体不存在的关键声音**（如居家场景的门铃细分、
   具体家电报警声等）—— 这是零样本路线的本质局限，必须如实呈现

### 5.4 误报控制

单帧 argmax 在连续音频上抖动严重。控制手段：

- **置信度阈值**：低于阈值的得分不产生事件
- **多帧一致性**：事件需在窗口内多帧重复出现才输出

这两个参数（阈值、窗口长度）的取值对比实验，是第五章实验数据的主要来源。

## 6. 评估策略

因不下载 UrbanSound8K（完整包 5.6 GB），评估改用 **ESC-50 抽样**，分四部分，
**每部分在报告中必须标注数据性质**：

| 数据来源 | 用途 | 性质 |
|---|---|---|
| 合成信号（正弦、白噪、脉冲、扫频） | 验证管线连通性；观察各类得分响应 | 自测 |
| **ESC-50 抽样片段**（见 6.1） | 零样本识别的按类命中率统计 | 自测 |
| 公开文献在 UrbanSound8K 上的 YAMNet 指标 | 与自测结果对照 | **文献值，明确标注，不与自测数据混排** |
| 映射覆盖率分析 | 覆盖宽度与盲区 | 自测，本实验独有价值 |

自测样本量小，**不声称任何统计意义上的准确率**。报告结论限定为"管线可用性验证"
与"小样本定性观察"，凡引用外部数字一律标注来源。

### 6.1 ESC-50 抽样方案

ESC-50 仓库（`karolpiczak/ESC-50`）满足全部约束：

- 结构：2000 条片段，50 个类别，每条为 16-bit 单声道 44.1 kHz WAV、时长 5 s、约 441 KB
- **单个片段可独立下载**，无需拉取全量包
  - 音频：`https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}.wav`
  - 标注：`https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv`
    （字段：filename, fold, target, category, esc10, src_file, take）
- 连通性已实测：raw 音频 HTTP 200、标注 CSV HTTP 200
- 读取：16-bit WAV 用 `scipy.io.wavfile` 即可，**无需 soundfile / librosa**

**抽样范围**：从 50 类中挑选与听障提示相关的类别（初步为 `crying_baby`、
`glass_breaking`、`siren`、`car_horn`、`clock_alarm`、`door_wood_knock`、`door_wood_creaks`、
`dog`、`fireworks`、`thunderstorm`、`church_bells`、`vacuum_cleaner`、`washing_machine`
等，最终以实际映射表为准逐条核对）。每类抽 5 条，约 13 类 × 5 条 × 441 KB ≈ **28 MB**。

**许可证**：ESC-50 整体为 **CC BY-NC 3.0（署名—非商业性使用）**。
本实验仅用于课程教学，不商用、不再分发，须在报告第六章伦理部分明确声明并署名
（Piczak, K. J. *ESC: Dataset for Environmental Sound Classification*, ACM MM 2015）。

**指标口径**：报告统计的是"ESC-50 类别 → 映射后听障事件的命中情况"，
即零样本预测是否落进该类别应有的听障事件族，**不是**标准 50 类分类准确率。
口径必须在报告中写明，避免与文献数字直接比较。

## 7. Coze 侧设计

### 7.1 工作流结构

单节点工作流：开始节点 → 大模型节点 → 结束节点。
大模型节点输入为事件 JSON 字符串，输出为分级中文提示文案。

### 7.2 粘贴内容规格

粘贴进 Coze 的是一段 JSON，格式如下：

```json
{
  "session_id": "sample_001",
  "audio_file": "doorbell_01.wav",
  "duration_sec": 3.2,
  "max_level": 2,
  "events": [
    {
      "event_id": "doorbell",
      "name_cn": "门铃",
      "level": 2,
      "level_name": "注意",
      "confidence": 0.73,
      "audio_set_labels": ["Doorbell"]
    }
  ]
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `session_id` | 本次识别会话标识 |
| `audio_file` | 源音频文件名 |
| `duration_sec` | 音频时长（秒） |
| `max_level` | 本次全部事件中的最高紧急等级 |
| `events[].event_id` | 归并后的事件标识 |
| `events[].name_cn` | 事件中文名 |
| `events[].level` / `level_name` | 紧急等级及其名称 |
| `events[].confidence` | 该事件置信度 |
| `events[].audio_set_labels` | 触发该事件的原始 AudioSet 类别名（可追溯性） |

### 7.3 大模型节点提示词（完整文本）

以下为 Coze 工作流大模型节点的**系统提示词全文**，实施阶段直接粘贴：

```
你是听障人士的无障碍环境声音提示助手。用户听不到环境中的关键声音，
由本系统检测后把结果交给你，你要把结构化事件数据转写成一句到三句
简洁、易读的中文提示。

【输入】
你会收到一段 JSON，字段含义：
- max_level：本次全部事件中的最高紧急等级（1=危险，2=注意）
- events：事件数组，每项含 name_cn（中文名）、level（等级）、
  confidence（置信度，0~1）、audio_set_labels（原始声音类别，仅供追溯）

【生成规则】
1. 只依据 events 数组中实际出现的事件陈述。严禁推测、补全或提及
   JSON 中没有出现的声音。
2. 按 max_level 决定语气：
   - max_level 为 1：语气明确、直接，指出是什么声音，并给出一条可执行的
     行动建议（例如"请立即查看厨房""请确认是否有人敲门"）。
   - max_level 为 2：语气平和，说明发生了什么即可，不给强指令。
3. confidence 低于 0.5 的事件，措辞要用"可能""似乎"表示不确定，
   不得表述为确定发生。
4. 同一等级有多个事件时，全部都要提到，不要遗漏。
5. 若 events 为空数组，只输出：当前环境未检测到需要提示的声音。
6. 面向听障用户，表达必须直白易懂：不使用"置信度""模型""AudioSet"
   "类别"等技术词汇，不说"根据数据""经检测"这类套话。
7. 不输出 JSON、不输出字段名、不加解释性前言，直接给出提示文案本身。
8. 全文控制在 1~3 句。

【输出示例】
输入 max_level=1，events 含 name_cn="玻璃破碎"（confidence 0.82）：
玻璃破碎的声音，可能来自窗户或餐具。请立即查看厨房和客厅。

输入 max_level=2，events 含 name_cn="门铃"（confidence 0.71）：
有人在按门铃，请留意门口。

输入 max_level=2，events 含 name_cn="狗叫"（confidence 0.34）：
似乎有狗在叫，声音不太确定，可以留意一下。

输入 events 为空数组：
当前环境未检测到需要提示的声音。
```

**节点参数建议**：温度设为较低值（0.3 左右），避免语言发散导致幻觉出
未检测到的声音。

## 8. 代码结构

```
音频智能体实验/
├── requirements.txt
├── README.md
├── scripts/
│   ├── download_model.py          # 从 hf-mirror 拉 yamnet.onnx + 类别表
│   ├── download_samples.py        # 从 raw.githubusercontent 拉 ESC-50 抽样片段
│   ├── audio_io.py
│   ├── yamnet_runner.py
│   ├── event_mapping.py
│   ├── detect.py
│   ├── make_test_signals.py
│   └── evaluate.py
├── models/                        # yamnet.onnx + yamnet_class_map.csv
├── data/
│   ├── samples/                   # 测试音频
│   └── out/                       # 事件 JSON、图表
└── docs/
    ├── specs/                     # 本文档
    └── 听障无障碍环境声音提示智能体实验报告.docx
```

## 9. 报告章节映射

严格对齐参考模板 `音频智能体实验报告.bak.docx` 的八节结构：

| 模板章节 | 本次内容 |
|---|---|
| 一、实验目的 | 无障碍 AI 背景；声学事件检测定位；零样本迁移 |
| 二、实验原理 | SED 概念、AudioSet 本体、YAMNet 架构、ONNX 推理、零样本迁移 |
| 三、实验环境 | Python 3.13 + onnxruntime + hf-mirror + Coze 平台 |
| 四、实验内容与方案设计 | **3 个场景**，每个含「实验目的/使用组件/实现步骤/输入输出/注意事项」五段式 |
| 五、实验结果与分析 | 管线验证、识别结果、映射覆盖率、去抖参数对比 |
| 六、实验伦理与风险讨论 | 误报漏报的安全后果、持续监听隐私、数据集偏见、责任边界、**ESC-50 的 CC BY-NC 3.0 署名与非商用声明** |
| 七、实验提交要求 | 同模板 |
| 八、思考题 | 同模板 |

三个场景：

1. **YAMNet 零样本环境声音识别**——音频到 521 类得分
2. **AudioSet→听障关注事件映射与分级决策**——得分到事件 JSON
3. **Coze 智能体分级提示文案生成**——JSON 到中文提示

## 10. 非目标（明确不做）

- 不训练或微调任何模型
- 不做实时麦克风流式监听
- 不自建梅尔滤波器组等音频特征提取（由 ONNX 计算图内部完成）
- 不接 TTS 语音播报、不建知识库问答
- 不调用 Coze API，链路 B 为人工粘贴
- 不声称任何统计意义上的识别准确率

## 11. 已知风险

| 风险 | 应对 |
|---|---|
| `onnxruntime` 在 Python 3.13 无可用 wheel | **实施第一步就验证**（`pip install onnxruntime`）；失败则回退到 3.12 虚拟环境，并在报告中记录该环境约束 |
| hf-mirror 不可用 | 下载脚本报错时明确提示，并给出手动下载说明 |
| ESC-50 单条音频下载失败 | 脚本逐条重试并报告失败清单，不静默跳过 |
| 零样本识别效果差 | 这本身是预期结果，作为第五章的发现如实报告，不粉饰 |
| 零样本识别效果差 | 这本身是预期结果，作为第五章的发现如实报告，不粉饰 |
| AudioSet 类别名与预期不符 | 映射表以实际下载的 `yamnet_class_map.csv` 为准逐条核对，不凭记忆填写 |
