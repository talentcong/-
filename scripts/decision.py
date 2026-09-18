"""从 YAMNet 得分到听障提示事件的决策逻辑。

三级过滤：
  1) 帧级聚合——把 (n_frames, 521) 压成 (521,)
  2) 阈值判定——低于阈值的类别不产生事件
  3) 多帧一致性——事件需在足够多帧中重复出现，抑制瞬时误报
"""
from dataclasses import dataclass

import numpy as np

from event_mapping import (
    LEVEL_ALERT_1,
    LEVEL_ALERT_2,
    LEVEL_NAMES,
    EventSpec,
)


@dataclass(frozen=True)
class DetectedEvent:
    event_id: str
    name_cn: str
    level: int
    confidence: float
    audio_set_labels: tuple[str, ...] = ()

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.level, "未知")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "name_cn": self.name_cn,
            "level": self.level,
            "level_name": self.level_name,
            "confidence": round(float(self.confidence), 4),
            "audio_set_labels": list(self.audio_set_labels),
        }


def aggregate_frames(frame_scores: np.ndarray) -> np.ndarray:
    """把 (n_frames, n_classes) 的帧级得分沿帧轴取均值，返回 (n_classes,)。"""
    return frame_scores.mean(axis=0)


def _is_alertable(spec: EventSpec) -> bool:
    return spec.level in (LEVEL_ALERT_1, LEVEL_ALERT_2)


def decide(scores: np.ndarray, lookup: dict[int, EventSpec],
           threshold: float = 0.5,
           class_names: dict[int, str] | None = None) -> list[DetectedEvent]:
    """按阈值从聚合得分中选出事件，同一 event_id 取最高置信度。

    返回按置信度降序排列的事件列表。
    """
    class_names = class_names or {}

    # event_id → (最高置信度, 中文名, 等级, [触发该事件的类别索引])
    grouped: dict[str, tuple[float, str, int, list[int]]] = {}
    for index, score in enumerate(scores):
        spec = lookup.get(index)
        if spec is None or not _is_alertable(spec):
            continue
        value = float(score)
        if value < threshold:
            continue
        best, name_cn, level, indices = grouped.get(
            spec.event_id, (0.0, spec.name_cn, spec.level, [])
        )
        grouped[spec.event_id] = (
            max(best, value), name_cn, level, indices + [index]
        )

    events: list[DetectedEvent] = []
    for event_id, (confidence, name_cn, level, indices) in grouped.items():
        events.append(DetectedEvent(
            event_id=event_id,
            name_cn=name_cn,
            level=level,
            confidence=confidence,
            audio_set_labels=tuple(
                class_names.get(i, str(i)) for i in sorted(indices)
            ),
        ))

    events.sort(key=lambda e: e.confidence, reverse=True)
    return events


def decide_temporal(frame_scores: np.ndarray, lookup: dict[int, EventSpec],
                    threshold: float = 0.5, min_frames: int = 1,
                    class_names: dict[int, str] | None = None) -> list[DetectedEvent]:
    """在多帧上做一致性判定：类别需在至少 min_frames 帧中超过阈值才算命中。

    置信度取这些命中帧的均值。
    """
    class_names = class_names or {}

    # event_id → {帧号: 该帧中属于此事件的最高类别得分}
    per_event: dict[str, dict[int, float]] = {}
    specs: dict[str, EventSpec] = {}

    for frame in range(frame_scores.shape[0]):
        for index in np.where(frame_scores[frame] >= threshold)[0]:
            idx = int(index)
            spec = lookup.get(idx)
            if spec is None or not _is_alertable(spec):
                continue
            specs.setdefault(spec.event_id, spec)
            frame_scores_of_event = per_event.setdefault(spec.event_id, {})
            frame_scores_of_event[frame] = max(
                frame_scores_of_event.get(frame, 0.0),
                float(frame_scores[frame][idx]),
            )

    events: list[DetectedEvent] = []
    for event_id, frames in per_event.items():
        if len(frames) < min_frames:
            continue
        spec = specs[event_id]
        events.append(DetectedEvent(
            event_id=event_id,
            name_cn=spec.name_cn,
            level=spec.level,
            confidence=float(np.mean(list(frames.values()))),
            audio_set_labels=tuple(
                class_names.get(i, str(i))
                for i, s in lookup.items() if s.event_id == event_id
            ),
        ))

    events.sort(key=lambda e: e.confidence, reverse=True)
    return events
