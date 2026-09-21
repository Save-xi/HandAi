"""用已保存的 M1 模型从 H2O NPZ 历史预测未来 21 点，只输出 JSON。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", EXPERIMENT_ROOT):
    sys.path.insert(0, str(path))

from intent_prediction.keypoint_data import read_sequence, validate_task  # noqa: E402
from prediction.keypoint_model_loader import load_keypoint_prediction_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="训练输出的 model.json")
    parser.add_argument("--sequence", required=True, type=Path, help="H2O 原生相机 3D NPZ")
    parser.add_argument("--task-spec", type=Path, default=EXPERIMENT_ROOT / "configs/keypoint_m0.json",
                        help="明确输入坐标、单位、点序和时间，不按形状猜测")
    parser.add_argument("--anchor-frame-id", type=int, help="原始帧号；默认序列最后一帧")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True, type=Path, help="预测 JSON；不写入设备输出")
    args = parser.parse_args()
    task = json.loads(args.task_spec.read_text(encoding="utf-8"))
    validate_task(task)
    if args.output.resolve() in {p.resolve() for p in (args.model, args.sequence, args.task_spec)}:
        raise ValueError("输出不能覆盖输入文件")
    with np.load(args.sequence, allow_pickle=False) as data:
        entry = {key: str(data[key].item()) for key in ("sequence_id", "subject", "split")}
    entry["path"] = args.sequence.name
    seq = read_sequence(args.sequence.parent, entry, task)
    if args.anchor_frame_id is None:
        anchor = len(seq.timestamps) - 1
    else:
        matches = np.flatnonzero(seq.frame_ids == args.anchor_frame_id)
        if not len(matches):
            raise ValueError("指定原始帧号不在序列中")
        anchor = int(matches[0])
    loaded = load_keypoint_prediction_model(args.model, expected_task_id=task["task_id"], device=args.device)
    result = loaded.predict(seq.inputs[:anchor + 1], seq.timestamps[:anchor + 1], input_task=task,
                            frame_ids=seq.frame_ids[:anchor + 1], mask=seq.input_mask[:anchor + 1],
                            segment_ids=seq.segments[:anchor + 1])
    result.update(sequence_id=seq.sequence_id, anchor_frame_id=int(seq.frame_ids[anchor]), device=loaded.device)
    serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"21 点预测已保存：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
