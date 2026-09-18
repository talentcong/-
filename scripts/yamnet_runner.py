"""YAMNet ONNX 推理封装。

模型输入为 16 kHz 单声道 float32 波形（预处理在计算图内部完成），
输出为 (n_frames, 521) 的 AudioSet 得分矩阵。

已实测确认（scripts/probe_model.py）：
  输入  waveform    shape=[unk__413]      动态长度，变长输入可用
  输出  output_0    shape=[unk__414, 521] AudioSet 得分
        output_1    shape=[unk__415, 1024] 嵌入向量
        output_2    shape=[unk__416, 64]   梅尔谱
"""
from pathlib import Path

import numpy as np
import onnxruntime

N_CLASSES = 521


class YamnetRunner:
    """加载一次 ONNX 会话，可重复调用 score()。"""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型不存在: {self.model_path}\n请先运行: python scripts/download_model.py"
            )
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def score(self, waveform: np.ndarray) -> np.ndarray:
        """对 16 kHz 单声道波形推理，返回 (n_frames, 521) 的 float32 得分矩阵。"""
        wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if wave.size == 0:
            raise ValueError("输入波形为空，无法推理")

        output = self.session.run(None, {self.input_name: wave})[0]
        scores = np.asarray(output, dtype=np.float32)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        if scores.shape[1] != N_CLASSES:
            raise ValueError(
                f"模型输出类别数为 {scores.shape[1]}，预期 {N_CLASSES}。"
                f"请确认模型文件是否为 yamnetonnx/yamnet.onnx"
            )
        return scores
