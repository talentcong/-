"""打印 YAMNet ONNX 的输入输出名称与形状，并用合成信号跑一次推理。"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yamnet.onnx"


def main() -> int:
    session = onnxruntime.InferenceSession(
        str(MODEL_PATH), providers=["CPUExecutionProvider"]
    )

    print("=== 输入 ===")
    for inp in session.get_inputs():
        print(f"  name={inp.name!r} shape={inp.shape} type={inp.type}")

    print("=== 输出 ===")
    for out in session.get_outputs():
        print(f"  name={out.name!r} shape={out.shape} type={out.type}")

    input_name = session.get_inputs()[0].name
    for seconds in (1.0, 5.0):
        n = int(16000 * seconds)
        wave = (0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / 16000)).astype(np.float32)
        result = session.run(None, {input_name: wave})[0]
        print(f"{seconds}s 输入 -> 输出形状 {result.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
