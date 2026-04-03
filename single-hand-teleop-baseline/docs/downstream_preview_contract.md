# Downstream Preview Contract

This note explains which parts of the frozen frame payload are intended for
downstream Unity / socket / SVH-preview consumers.

## 1. Contract Scope

This baseline does **not** emit real hardware-safe SVH commands.

It does emit two extension objects that are useful for downstream integration
work before real hardware is connected:

- `control_representation`
- `svh_preview`

Both objects are always present in the canonical payload so downstream code can
depend on a stable shape. When an extension is disabled or the current frame is
not ready, the object remains present but degrades to a safe invalid/disabled
state.

## 2. Preferred Downstream Fields

### Top-level fields

These fields are the best first read for Unity, socket, or session-logging
consumers:

- `frame_index`
- `timestamp`
- `detected`
- `handedness`
- `gesture_stable`
- `control_ready`
- `control_representation`
- `svh_preview`

### control_representation

Preferred fields:

- `command_ready`
- `preferred_mapping`
- `grasp_close`
- `effective_pinch_strength`
- `finger_flex`
- `support_flex`

Compatibility mirrors kept for now:

- `valid`
  - compatibility mirror of `command_ready`
- `pinch_strength`
  - compatibility mirror of `effective_pinch_strength`

Units and ranges:

- `grasp_close`: normalized `[0, 1]`
- `thumb_index_proximity`: normalized `[0, 1]`
- `effective_pinch_strength`: normalized `[0, 1]`
- `support_flex`: normalized `[0, 1]`
- `finger_flex.*`: normalized `[0, 1]`

Interpretation:

- `preferred_mapping="grasp"` means the stable gesture context is using the
  grasp family.
- `preferred_mapping="pinch"` means the stable gesture context is using the
  pinch family.
- `command_ready=false` means downstream code should not turn this frame into a
  new control action.

### svh_preview

Preferred fields:

- `enabled`
- `valid`
- `command_source`
- `target_channels`
- `target_positions`
- `target_ticks_preview`
- `protocol_hint`

Units and ranges:

- `target_positions`
  - normalized preview values in `[0, 1]`
  - these are the preferred direct-consumption values for Unity / socket preview
    integrations
- `target_ticks_preview`
  - preview-only encoder-like integers
  - useful for UI/debugging and future transport planning
  - **not** a final hardware-safe command unit

Important semantics:

- `enabled=false`
  - the SVH preview extension is off for this run
- `valid=false`
  - no usable preview command should be consumed from this frame
- `command_source="control_representation"`
  - preview came from stable continuous control output
- `command_source="gesture_fallback"`
  - preview came from demo-oriented fallback logic

## 3. Example JSON Shape

```json
{
  "frame_index": 233,
  "timestamp": 1775039301.241,
  "detected": true,
  "gesture_stable": "pinch",
  "control_ready": true,
  "control_representation": {
    "command_ready": true,
    "preferred_mapping": "pinch",
    "grasp_close": 0.115,
    "effective_pinch_strength": 0.922,
    "finger_flex": {
      "thumb": 0.048,
      "index": 0.272,
      "middle": 0.022,
      "ring": 0.023,
      "little": 0.025
    }
  },
  "svh_preview": {
    "enabled": true,
    "valid": true,
    "command_source": "control_representation",
    "target_channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "target_positions": [0.791, 0.707, 0.792, 0.678, 0.185, 0.185, 0.185, 0.185, 0.092],
    "target_ticks_preview": [-139440, -107555, -37637, 28381, 6695, 6695, 6695, 6695, -6155],
    "protocol_hint": {
      "channel_layout": "svh_9ch",
      "position_units": "normalized_preview",
      "target_tick_units": "encoder_ticks_preview"
    }
  }
}
```

## 4. What Downstream Can Consume Directly

Unity / virtual-hand preview can safely consume:

- `gesture_stable`
- `control_ready`
- `control_representation.grasp_close`
- `control_representation.effective_pinch_strength`
- `control_representation.finger_flex`
- `svh_preview.target_positions`
- `svh_preview.target_channels`

Socket / logging clients can safely record:

- the full frozen payload
- `target_ticks_preview` as preview metadata
- `protocol_hint` for layout/unit awareness

## 5. Preview-only Fields

These should be treated as preview/debug metadata rather than final command
truth:

- `svh_preview.target_ticks_preview`
- `svh_preview.protocol_hint`
- `control_representation.valid`
- `control_representation.pinch_strength`

## 6. Safety Gaps Before Real SVH

Before connecting a real dexterous hand, at least these measures are still
missing:

1. transport ACK / retry / timeout handling
2. homing and zero-position confirmation
3. per-channel hard safety limits and calibration
4. watchdog / heartbeat and fault reset flow
5. operator emergency stop path
6. command rate limiting and smoothing policy
7. confirmation that `target_ticks_preview` matches real device units
8. real packet packing / checksum / response parsing validation

Until those exist, `svh_preview` must remain preview/mock only.
