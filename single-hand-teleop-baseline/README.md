# Single Right Hand Teleop Baseline

## 1. Project Scope

This directory implements a single-right-hand vision baseline.

Baseline chain:

`input -> MediaPipe hand detection -> right-hand filter -> feature extraction -> gesture stabilization -> JSON / JSONL / OpenCV visualization`

Optional extension chain:

`baseline payload -> control_representation -> svh_preview -> mock transport`

Current scope:

- single right hand only
- MediaPipe-based hand perception
- rule-based gesture classification with stabilization
- baseline-only perception / gesture / JSON output
- optional `control_representation`
- optional SVH preview adapter, protocol skeleton, and mock transport
- unified per-frame JSON / JSONL output

Out of scope for this baseline:

- dual-hand main flow
- Unity runtime integration
- real TCP/IP + serial server + RS485 transport
- real SVH hardware control
- ROS / database / web frontend

## 2. Environment

Recommended Python version:

- Python `3.10`

Recommended:

```bash
cd single-hand-teleop-baseline
conda env create -f environment.yml
conda activate single-right-hand-baseline
```

Or:

```bash
cd single-hand-teleop-baseline
conda create -n single-right-hand-baseline python=3.10 -y
conda activate single-right-hand-baseline
pip install -r requirements.txt
```

Dependency strategy:

- `requirements.txt` is pinned to a tested set of versions for reproducible installs and CI runs
- `environment.yml` mirrors the same pinned Python/runtime stack for Conda users
- CI-only tools such as `ruff` and `pip-audit` are installed in CI separately so the runtime environment stays lean

## 3. Run

Default baseline run:

```bash
python src/main.py --config configs/default.yaml
```

Common options:

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

Examples:

```bash
python src/main.py --config configs/default.yaml --print-json
python src/main.py --config configs/default.yaml --headless --video-file path/to/demo.mp4 --max-frames 300
python src/main.py --config configs/default.yaml --print-json --save-jsonl
python src/main.py --config configs/default.yaml --enable-control --print-json
python src/main.py --config configs/default.yaml --preview-svh --print-json
python src/main.py --config configs/svh_9ch_preview.yaml --print-json
```

## 4. Run Modes

| Mode | How to enter | What is enabled | Extra environment needed |
|---|---|---|---|
| Baseline | `configs/default.yaml` | detection, right-hand filtering, features, gesture, visualization, JSON / JSONL | camera for live mode, or video file if `--video-file` is used |
| Baseline headless | `--no-gui` or `--headless` | same baseline chain without OpenCV window | no GUI required |
| Control extension | `enable_control_extension: true` or `--enable-control` | baseline + `control_representation` | no hardware required |
| SVH preview extension | `svh_enable_preview: true` or `--preview-svh` | baseline + `control_representation` + `svh_preview` + optional mock transport | no real SVH required |

Default behavior:

- `configs/default.yaml` runs in baseline-only mode
- control / SVH preview are off by default
- extension failures degrade to invalid preview objects instead of crashing the baseline loop
- unsupported non-mock SVH transports log a warning and stay in preview-only mode
## 5. Directory Layout

- `src/capture/`: camera input
- `src/perception/`: MediaPipe detector, right-hand filter, quality gate
- `src/features/`: geometric helpers and hand features
- `src/gesture/`: rule-based gesture logic and stabilizer
- `src/control/`: continuous control representation
- `src/svh/`: SVH preview adapter, layout, protocol skeleton, mock transport
- `src/output/`: payload contract, exporter, JSON / JSONL output
- `src/visualize/`: OpenCV overlay and status panel
- `configs/`: runtime configs
- `examples/`: frozen sample payloads
- `schemas/`: JSON schema
- `tests/`: tests

## 6. Frozen Frame Payload Contract

The exported per-frame payload is now frozen around one canonical schema.

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

Design choices:

- `gesture_stable` is the debounced gesture label used by control and preview logic.
- `control_ready` means the frame is ready to drive control output this frame.
- `landmarks_3d` is the current MediaPipe-style `(x, y, z)` landmark stream. It is not named `landmarks_world` because this baseline does not yet guarantee world-space semantics.
- `svh_preview` is the canonical preview object for downstream Unity / SVH / JSONL use.

### Deprecated Aliases

These old top-level names are deprecated:

- `gesture` -> `gesture_stable`
- `svh` -> `svh_preview`

Compatibility policy:

- the compatibility layer can still read these old aliases during transition
- canonical JSON / JSONL output no longer emits them
- they should be removed completely before the project grows a real transport / hardware integration path

Schema files:

- [frame_payload.schema.json](/D:/VR/HandAi/single-hand-teleop-baseline/schemas/frame_payload.schema.json)
- [frame_payload_contract.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/output/frame_payload_contract.py)

## 7. Sample JSON

```json
{
  "timestamp": 1774965000.123,
  "frame_index": 184,
  "detected": true,
  "handedness": "Right",
  "confidence": 0.972,
  "control_ready": false,
  "gesture_raw": "open",
  "gesture_stable": "open",
  "pinch_distance_norm": 0.92,
  "hand_open_ratio": 0.99,
  "finger_curl": {
    "thumb": 0.022,
    "index": 0.015,
    "middle": 0.014,
    "ring": 0.011,
    "little": 0.013
  },
  "landmarks_2d": [
    [0.282, 0.895],
    [0.336, 0.866],
    [0.370, 0.795]
  ],
  "landmarks_3d": [
    [0.282, 0.895, -0.012],
    [0.336, 0.866, -0.017],
    [0.370, 0.795, -0.021]
  ],
  "control_representation": {
    "valid": false,
    "features_valid": false,
    "command_ready": false,
    "source": null,
    "gesture_context": null,
    "preferred_mapping": null,
    "grasp_close": null,
    "thumb_index_proximity": null,
    "effective_pinch_strength": null,
    "pinch_strength": null,
    "support_flex": null,
    "finger_flex": {
      "thumb": null,
      "index": null,
      "middle": null,
      "ring": null,
      "little": null
    }
  },
  "svh_preview": {
    "enabled": false,
    "mode": "disabled",
    "valid": false,
    "command_source": null,
    "target_channels": [],
    "target_positions": [],
    "target_ticks_preview": [],
    "protocol_hint": {
      "set_control_state_addr": "0x09",
      "set_all_channels_addr": "0x03",
      "transport": "mock",
      "channel_layout": "compact5",
      "channel_order": "thumb,index,middle,ring,little",
      "position_units": "normalized_preview",
      "target_tick_units": "none"
    }
  },
  "fps": 27.77,
  "latency_ms": 20.037
}
```

Frozen samples:

- [sample_output.json](/D:/VR/HandAi/single-hand-teleop-baseline/examples/sample_output.json) for baseline default mode
- [sample_output_svh_9ch.json](/D:/VR/HandAi/single-hand-teleop-baseline/examples/sample_output_svh_9ch.json) for `svh_9ch` preview mode
- [sample_session.jsonl](/D:/VR/HandAi/single-hand-teleop-baseline/examples/sample_session.jsonl) for a two-frame baseline + extension session example

## 8. control_representation

`control_representation` is an optional extension layer. It stays hardware-agnostic and exposes a stable intermediate layer rather than pretending to be a final motor command.

Important fields:

- `features_valid`: continuous features are available this frame
- `command_ready`: gesture context is stable enough to choose a control mapping
- `grasp_close`: overall grasp closure
- `thumb_index_proximity`: raw thumb-index closeness cue
- `effective_pinch_strength`: pinch cue after gesture-aware gating
- `finger_flex`: per-finger normalized flexion cues in `[0, 1]`

`control_ready` at the top level is aligned with `control_representation.command_ready`.

Compatibility mirrors still present:

- `control_representation.valid`
  - mirror of `command_ready`
- `control_representation.pinch_strength`
  - mirror of `effective_pinch_strength`

If the extension is disabled or fails, the payload still emits the same object shape with `valid=false`.

## 9. svh_preview

`svh_preview` is also an optional extension layer. It is still preview / skeleton only.

What it is:

- a preview mapping from `control_representation` into an SVH-like command object
- suitable for JSON / JSONL recording, visualization, and future integration planning

What it is not:

- not a real TCP client
- not a real serial-server bridge
- not a real RS485 implementation
- not a completed real SVH protocol implementation

Current layouts:

- `compact5`
- `svh_9ch`

The `svh_9ch` layout aligns with the current Unity/C# reference ordering assumption, but it is still a preview assumption rather than a confirmed hardware fact.

Downstream-facing semantics:

- `svh_preview.target_positions`
  - normalized preview command values in `[0, 1]`
  - preferred direct-consumption values for Unity / socket preview integrations
- `svh_preview.target_ticks_preview`
  - preview-only encoder-like values
  - useful for debugging and planning
  - not a hardware-safe final command unit
- `svh_preview.valid=false`
  - downstream should not consume this frame as a new preview command

If the extension is disabled or fails, the payload keeps `svh_preview.enabled=false` or `svh_preview.valid=false` instead of aborting the baseline loop.

## 10. Downstream Integration

For a focused summary of what Unity / socket / mock-SVH consumers should read
from the payload, see:

- [downstream_preview_contract.md](/D:/VR/HandAi/single-hand-teleop-baseline/docs/downstream_preview_contract.md)

Short version:

- direct-consumption fields:
  - `gesture_stable`
  - `control_ready`
  - `control_representation.grasp_close`
  - `control_representation.effective_pinch_strength`
  - `control_representation.finger_flex`
  - `svh_preview.target_positions`
  - `svh_preview.target_channels`
- preview-only fields:
  - `svh_preview.target_ticks_preview`
  - `svh_preview.protocol_hint`
- before real SVH:
  - transport ACK / timeout handling
  - homing / zeroing
  - safety limits
  - watchdog / e-stop
  - real packet and unit validation

## 11. Testing

Run:

```bash
python -m compileall -q src
pytest -q
python src/main.py --help
```

Optional local checks if you install the CI-only tooling:

```bash
python -m ruff check src tests
```

Current tests cover:

- frozen frame payload contract
- schema file consistency
- exporter and JSONL output
- feature extraction shape
- gesture logic
- control representation
- SVH preview mapping
- 9-channel preview layout
- protocol skeleton constants
- mock transport

## 12. CI, License, and Contribution Guide

Engineering guardrails now live in:

- [single-hand-teleop-baseline-ci.yml](/D:/VR/HandAi/.github/workflows/single-hand-teleop-baseline-ci.yml)
- [CONTRIBUTING.md](/D:/VR/HandAi/single-hand-teleop-baseline/CONTRIBUTING.md)
- [LICENSE](/D:/VR/HandAi/LICENSE)

CI currently runs on GitHub Actions for pushes and pull requests touching this subproject. The workflow:

1. checks out the repository
2. installs the pinned Python dependencies
3. runs a small `ruff` guardrail check
4. runs `python -m compileall -q src`
5. runs `pytest -q`
6. runs `pip-audit` as a non-blocking advisory step

`pip-audit` is intentionally advisory right now so vulnerability data can be surfaced without turning transient ecosystem issues into hard false negatives for every PR.

The repository now carries an MIT license to make reuse and collaboration straightforward for this student-project baseline.

## 13. Next Steps

If the project later moves toward real SVH integration, the next work should be:

1. confirm real channel mapping
2. confirm `target_ticks_preview` vs real encoder units
3. confirm packet packing, lengths, and checksums
4. add a real transport implementation in a separate module
5. only then connect the vision output to hardware
