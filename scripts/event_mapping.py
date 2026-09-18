"""AudioSet 521 类 → 听障关注事件的映射与紧急分级。

类别索引取自实际下载的 yamnet_class_map.csv，逐条核对过。
同一事件族共享 event_id，避免同一声源重复告警。

等级约定：
  1 = 危险，立即提示
  2 = 注意，需要知晓
  3 = 已识别但可忽略的环境音
  0 = 未覆盖，AudioSet 本体不含该声音（覆盖盲区，不是"可忽略"）
"""
from dataclasses import dataclass
from pathlib import Path

LEVEL_ALERT_1 = 1
LEVEL_ALERT_2 = 2
LEVEL_IGNORE = 3
LEVEL_UNMAPPED = 0

LEVEL_NAMES = {
    LEVEL_ALERT_1: "危险",
    LEVEL_ALERT_2: "注意",
    LEVEL_IGNORE: "可忽略",
    LEVEL_UNMAPPED: "未覆盖",
}


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    name_cn: str
    level: int


# 事件族定义：event_id → (中文名, 等级, [AudioSet 类别索引])
EVENT_TABLE: dict[str, tuple[str, int, list[int]]] = {
    # ---- 一级：危险，立即提示 ----
    "fire_alarm":   ("火警",        LEVEL_ALERT_1, [393, 394]),
    "siren":        ("警报声",      LEVEL_ALERT_1, [316, 317, 318, 319, 390, 391]),
    "car_alarm":    ("汽车防盗警报", LEVEL_ALERT_1, [304]),
    "explosion":    ("爆炸声",      LEVEL_ALERT_1, [420, 424]),
    "gunshot":      ("枪声",        LEVEL_ALERT_1, [421, 422, 425]),
    "glass_break":  ("玻璃破碎",    LEVEL_ALERT_1, [435, 437]),
    "screaming":    ("尖叫声",      LEVEL_ALERT_1, [11]),
    "alarm":        ("警报器",      LEVEL_ALERT_1, [382]),
    "fire":         ("燃烧声",      LEVEL_ALERT_1, [292]),
    "bang":         ("爆响",        LEVEL_ALERT_1, [460, 426, 427, 428]),
    # ---- 二级：注意，需要知晓 ----
    "doorbell":     ("门铃",        LEVEL_ALERT_2, [349]),
    "knock":        ("敲门声",      LEVEL_ALERT_2, [353]),
    "door":         ("开关门声",    LEVEL_ALERT_2, [348, 351]),
    "phone_ring":   ("电话铃声",    LEVEL_ALERT_2, [383, 384, 385]),
    "baby_cry":     ("婴儿哭声",    LEVEL_ALERT_2, [19, 20]),
    "dog_bark":     ("狗叫声",      LEVEL_ALERT_2, [69, 70, 75, 117]),
    "car_horn":     ("汽车鸣笛",    LEVEL_ALERT_2, [302, 312]),
    "alarm_clock":  ("闹钟",        LEVEL_ALERT_2, [389]),
    "buzzer":       ("蜂鸣器",      LEVEL_ALERT_2, [392]),
    "whistle":      ("哨声/汽笛",   LEVEL_ALERT_2, [395, 396, 397, 324, 325]),
    "bell":         ("铃声",        LEVEL_ALERT_2, [195, 196, 198, 202]),
    "beep":         ("电子提示音",  LEVEL_ALERT_2, [475, 313]),
    "thunder":      ("雷声",        LEVEL_ALERT_2, [280, 281]),
}

# 可忽略环境音的判定关键词，作用于 display_name（小写匹配）
IGNORE_KEYWORDS = [
    "music", "speech", "silence", "noise", "static", "hum",
    "traffic", "vehicle", "engine", "car", "bus", "train", "subway",
    "aircraft", "airplane", "helicopter", "motorcycle", "boat",
    "vacuum", "air conditioning", "fan",
    "typing", "footstep", "walk",
]


def _ignored(event_id: str, name_cn: str) -> EventSpec:
    return EventSpec(event_id, name_cn, LEVEL_IGNORE)


# 可忽略类别：索引 → 中文名。索引取自实际类别表。
IGNORE_TABLE: dict[int, str] = {
    0: "说话声", 1: "儿童说话声", 5: "语音合成器", 12: "耳语", 65: "嘈杂人声",
    132: "音乐", 133: "乐器", 211: "流行乐", 212: "嘻哈乐", 214: "摇滚乐",
    232: "古典乐", 262: "背景音乐", 265: "影视配乐", 269: "舞曲",
    48: "脚步声", 378: "打字声",
    294: "车辆声", 300: "机动车声", 301: "汽车声", 308: "车辆驶过",
    315: "公交车声", 320: "摩托车声", 321: "交通噪声", 323: "火车声",
    328: "地铁声", 330: "飞机引擎声", 331: "喷气引擎声", 333: "直升机声",
    334: "固定翼飞机声", 337: "引擎声", 338: "高频引擎声",
    342: "中频引擎声", 343: "低频引擎声", 345: "引擎启动声", 346: "怠速声",
    371: "吸尘器声", 406: "机械风扇声", 407: "空调声",
    494: "静音", 507: "噪声", 508: "环境噪声", 509: "静电噪声",
    510: "电源交流声", 514: "白噪声", 515: "粉红噪声", 279: "风噪",
}

UNMAPPED_SPEC = EventSpec("__unmapped__", "未覆盖类别", LEVEL_UNMAPPED)


def build_lookup(class_names: dict[int, str]) -> dict[int, EventSpec]:
    """构建 类别索引 → EventSpec 的查找表。

    class_names 为 {索引: display_name}，通常来自 download_model.parse_class_map。
    优先级：事件表 > 忽略表 > 关键词匹配 > 未覆盖。
    """
    lookup: dict[int, EventSpec] = {}

    for event_id, (name_cn, level, indices) in EVENT_TABLE.items():
        for index in indices:
            lookup[index] = EventSpec(event_id, name_cn, level)

    for index, name_cn in IGNORE_TABLE.items():
        lookup.setdefault(index, _ignored(f"ignore_{index}", name_cn))

    for index, display_name in class_names.items():
        if index in lookup:
            continue
        lowered = display_name.lower()
        if any(keyword in lowered for keyword in IGNORE_KEYWORDS):
            lookup[index] = _ignored(f"ignore_{index}", "可忽略环境音")
        else:
            lookup[index] = UNMAPPED_SPEC

    return lookup


def build_lookup_from_csv(csv_path: str | Path) -> dict[int, EventSpec]:
    """从 yamnet_class_map.csv 直接构建查找表。"""
    from download_model import parse_class_map

    return build_lookup(parse_class_map(Path(csv_path)))


def coverage_report(lookup: dict[int, EventSpec]) -> dict[str, int]:
    """统计映射覆盖率：各级别类别数与未覆盖数。"""
    counts = {
        "total": len(lookup),
        "alert_1": 0,
        "alert_2": 0,
        "ignore": 0,
        "unmapped": 0,
    }
    for spec in lookup.values():
        if spec.level == LEVEL_ALERT_1:
            counts["alert_1"] += 1
        elif spec.level == LEVEL_ALERT_2:
            counts["alert_2"] += 1
        elif spec.level == LEVEL_IGNORE:
            counts["ignore"] += 1
        else:
            counts["unmapped"] += 1
    return counts
