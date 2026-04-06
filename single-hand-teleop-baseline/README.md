# Single Right Hand Teleop Baseline

`single-hand-teleop-baseline/` is currently the most mature runnable part of this repository.

If you are cloning the repo for the first time, this subproject is the intended starting point. The supported baseline today is a single right-hand visual pipeline that runs from a webcam or a local video file and emits a frozen per-frame payload contract.

Current stage:

- The main focus is a single right-hand vision baseline.
- Webcam and local video input are the supported baseline entry points.
- The exported frame payload contract is intentionally frozen for downstream consumers.
- `control_representation` and `svh_preview` are optional extensions on top of the baseline payload.
- Unity integration, real hardware transport, and real SVH control are not required to install or start the baseline.

Baseline chain:

`input -> MediaPipe hand detection -> right-hand filter -> feature extraction -> gesture stabilization -> JSON / JSONL / OpenCV visualization`

Optional extension chain:

`baseline payload -> control_representation -> svh_preview -> mock transport`

## Scope

Current scope:

- single right hand only
- right-hand-focused runtime path
- webcam or local video input
- MediaPipe-based hand perception
- rule-based gesture classification with stabilization
- baseline JSON / JSONL export with a frozen frame payload contract
- optional `control_representation`
- optional SVH preview adapter and mock transport

Out of scope for the baseline startup path:

- dual-hand / multi-hand runtime flow
- Unity runtime integration
- real TCP / serial / RS485 transport
- real SVH hardware control
- ROS / database / web frontend

Future extensions may grow around these areas, but they should be treated as downstream work on top of the baseline rather than assumptions baked into the default startup path.

## Key Files

Configs:

- [configs/default.yaml](configs/default.yaml)
- [configs/svh_9ch_preview.yaml](configs/svh_9ch_preview.yaml)

Examples:

- [examples/sample_output.json](examples/sample_output.json)
- [examples/sample_output_svh_9ch.json](examples/sample_output_svh_9ch.json)

Schema and payload contract:

- [schemas/frame_payload.schema.json](schemas/frame_payload.schema.json)
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)

Docs and tests:

- [docs/downstream_preview_contract.md](docs/downstream_preview_contract.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [tests/](tests/)
- [tests/test_cli_smoke.py](tests/test_cli_smoke.py)
- [tests/test_json_schema.py](tests/test_json_schema.py)

Repository-level workflow:

- [../.github/workflows/single-hand-teleop-baseline-ci.yml](../.github/workflows/single-hand-teleop-baseline-ci.yml)

## Environment

Recommended Python version:

- Python `3.10`

You can install dependencies either from the subproject directory or from the repository root.

### Option A: Run From `single-hand-teleop-baseline/`

Conda:

```bash
cd single-hand-teleop-baseline
conda env create -f environment.yml
conda activate single-right-hand-baseline
```

Pip:

```bash
cd single-hand-teleop-baseline
python -m pip install -r requirements.txt
```

### Option B: Run From Repository Root

Conda:

```bash
conda env create -f single-hand-teleop-baseline/environment.yml
conda activate single-right-hand-baseline
```

Pip:

```bash
python -m pip install -r single-hand-teleop-baseline/requirements.txt
```

## Minimal Commands

### If Your Current Directory Is `single-hand-teleop-baseline/`

Show CLI help:

```bash
python src/main.py --help
```

Run the default baseline:

```bash
python src/main.py --config configs/default.yaml
```

Run the smallest smoke test:

```bash
pytest -q tests/test_cli_smoke.py
```

### If Your Current Directory Is The Repository Root

Show CLI help:

```bash
python single-hand-teleop-baseline/src/main.py --help
```

Run the default baseline:

```bash
python single-hand-teleop-baseline/src/main.py --config single-hand-teleop-baseline/configs/default.yaml
```

Run the smallest smoke test:

```bash
pytest -q single-hand-teleop-baseline/tests/test_cli_smoke.py
```

## Common Runtime Options

- `--camera-index 1`
- `--video-file path/to/demo.mp4`
- `--input-mirrored`
- `--no-gui`
- `--headless`
- `--enable-control`
- `--preview-svh`
- `--print-json`
- `--save-jsonl`
- `--max-frames 300`

Example commands from `single-hand-teleop-baseline/`:

```bash
python src/main.py --config configs/default.yaml --print-json
python src/main.py --config configs/default.yaml --headless --video-file path/to/demo.mp4 --max-frames 300
python src/main.py --config configs/default.yaml --enable-control --print-json
python src/main.py --config configs/svh_9ch_preview.yaml --print-json
```

## Run Modes

| Mode | How to enter | What is enabled | Extra environment needed |
|---|---|---|---|
| Baseline | `configs/default.yaml` | detection, right-hand filtering, features, gesture, visualization, JSON / JSONL | camera for live mode, or a video file with `--video-file` |
| Baseline headless | `--no-gui` or `--headless` | same baseline chain without an OpenCV window | no GUI required |
| Control extension | `enable_control_extension: true` or `--enable-control` | baseline + `control_representation` | no hardware required |
| SVH preview extension | `svh_enable_preview: true` or `--preview-svh` | baseline + `control_representation` + `svh_preview` + mock transport | no real SVH required |

Default behavior:

- [configs/default.yaml](configs/default.yaml) runs in baseline-only mode.
- The default startup target is a single right-hand webcam/video baseline.
- Control and SVH preview are off by default.
- Extension failures degrade to invalid preview objects instead of crashing the baseline loop.
- Unsupported non-mock SVH transports log a warning and stay in preview-only mode.

## Payload Contract

The exported per-frame payload is frozen around a canonical schema.

Canonical top-level fields:

- `timestamp`
- `frame_index`
- `detected`
- `handedness`
- `confidence`
- `control_ready`
- `gesture_raw`
- `gesture_stable`
- `pinch_distance_norm`
- `hand_open_ratio`
- `finger_curl`
- `landmarks_2d`
- `landmarks_3d`
- `control_representation`
- `svh_preview`
- `fps`
- `latency_ms`

Deprecated aliases:

- `gesture` -> `gesture_stable`
- `svh` -> `svh_preview`

Reference files:

- [schemas/frame_payload.schema.json](schemas/frame_payload.schema.json)
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)

Sample payload files:

- [examples/sample_output.json](examples/sample_output.json) for baseline default mode
- [examples/sample_output_svh_9ch.json](examples/sample_output_svh_9ch.json) for `svh_9ch` preview mode

Runtime session JSONL logs are written to `outputs/` when `--save-jsonl` is enabled.

## What This Baseline Is Not

- Unity runtime is not required to launch, test, or validate the baseline.
- Real SVH hardware control is not a prerequisite for baseline startup.
- Dual-hand or broader multi-hand behavior is not part of the current supported baseline path.
- Preview-oriented extensions should not be treated as proof of a production-ready robot or game-engine integration.

## Extension Notes

`control_representation`:

- optional extension layer
- hardware-agnostic intermediate representation
- not required for baseline startup

`svh_preview`:

- optional preview-only extension
- useful for JSON / JSONL recording and future integration planning
- not a real hardware-safe SVH command path

For downstream field semantics, see [docs/downstream_preview_contract.md](docs/downstream_preview_contract.md).

## Future Extension Directions

Likely future work includes:

- broader dual-hand or multi-hand perception experiments
- Unity or other downstream runtime adapters
- real transport and hardware control layers beyond the current mock preview path

Those directions are intentionally separate from the default baseline so that the single right-hand visual chain stays easy to install, run, and validate.

## Validation

Minimal validation from `single-hand-teleop-baseline/`:

```bash
python src/main.py --help
pytest -q tests/test_cli_smoke.py
```

Broader validation commands used in this subproject:

```bash
python -m compileall -q src
pytest -q
```

If you install CI-only tooling locally, you can also run:

```bash
python -m ruff check src tests
```
