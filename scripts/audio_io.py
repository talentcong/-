"""音频读取与预处理：转单声道、重采样到 16 kHz、峰值归一化。

YAMNet 要求 16 kHz 单声道、幅值落在 [-1, 1] 的 float32 波形。
"""
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly

TARGET_SR = 16000


def to_mono(audio: np.ndarray) -> np.ndarray:
    """多声道取各声道均值，单声道原样返回。"""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    """峰值归一化到 [-1, 1]。全零（静音）输入原样返回，避免除零。"""
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    return audio / peak


def to_float32(audio: np.ndarray) -> np.ndarray:
    """整型 PCM 转为 [-1, 1] 的 float32；已是浮点则直接转类型。"""
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        return (audio.astype(np.float32) / scale).astype(np.float32)
    return audio.astype(np.float32)


def resample_to_target(audio: np.ndarray, sr_in: int,
                       sr_out: int = TARGET_SR) -> np.ndarray:
    """重采样到目标采样率。采样率已匹配则原样返回。"""
    if sr_in == sr_out:
        return audio
    divisor = gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // divisor
    down = int(sr_in) // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def load_audio(path: str | Path, sr_out: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """读取音频文件，返回 (单声道 float32 波形, 采样率)。

    依次执行：整型转浮点 → 转单声道 → 重采样 → 峰值归一化。
    """
    sr_in, raw = wavfile.read(str(path))
    audio = to_float32(raw)
    audio = to_mono(audio)
    audio = resample_to_target(audio, sr_in, sr_out)
    audio = normalize_peak(audio)
    return audio.astype(np.float32), sr_out
