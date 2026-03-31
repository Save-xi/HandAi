from __future__ import annotations

from typing import Dict, Iterable

# Paper-informed default sync header. This is still only a preview skeleton:
# real packet framing, padding, and checksum behavior must be validated against
# the actual SVH device protocol before any hardware transport is implemented.
SYNC_BYTES = bytes([0x4C, 0xAA])
SET_CONTROL_STATE_ADDR = 0x09
SET_ALL_CHANNELS_ADDR = 0x03


def _endianness_label(use_little_endian: bool) -> str:
    return "little" if use_little_endian else "big"


def _resolve_sync_bytes(cfg: Dict | None = None) -> bytes:
    if cfg is None:
        return SYNC_BYTES
    configured = cfg.get("svh_protocol_sync_bytes")
    if configured is None:
        return SYNC_BYTES
    values = [int(v) & 0xFF for v in configured]
    if len(values) != 2:
        raise ValueError("svh_protocol_sync_bytes must contain exactly two byte values")
    return bytes(values)


def _resolve_endianness(use_little_endian: bool | None, cfg: Dict | None = None) -> bool:
    if use_little_endian is not None:
        return bool(use_little_endian)
    if cfg is None:
        return True
    return bool(cfg.get("svh_protocol_use_little_endian", True))


def build_set_control_state_packet(
    control_state: int,
    use_little_endian: bool | None = None,
    cfg: Dict | None = None,
) -> Dict:
    """Build a preview skeleton for the SetControlState frame.

    This intentionally returns a structured preview rather than claiming a fully
    validated hardware packet. The paper describes a frame with sync bytes,
    address, length, payload, and checksum fields, but exact packing still needs
    real-device calibration before hardware use.
    """

    little_endian = _resolve_endianness(use_little_endian, cfg)
    sync_bytes = _resolve_sync_bytes(cfg)
    payload = int(control_state).to_bytes(1, _endianness_label(little_endian), signed=False)
    return {
        "sync_bytes": sync_bytes.hex(" "),
        "address": f"0x{SET_CONTROL_STATE_ADDR:02X}",
        "payload_length": len(payload),
        "payload_preview_hex": payload.hex(),
        "endianness": _endianness_label(little_endian),
        "checksum": None,
        "expected_command_frame_size_bytes": 40,
        "expected_response_frame_size_bytes": 64,
        "note": "preview skeleton only; checksum/length framing/zero padding still must be verified on real SVH protocol",
    }


def build_set_all_channels_packet(
    channel_indices: Iterable[int],
    target_positions: Iterable[float],
    use_little_endian: bool | None = None,
    cfg: Dict | None = None,
) -> Dict:
    """Build a preview skeleton for the SetControlCommand AllChannels frame.

    The paper notes that channel information may be encoded in the address
    field's upper nibble on the real device. We keep that as a future calibration
    concern and only expose a structured preview here.
    """

    little_endian = _resolve_endianness(use_little_endian, cfg)
    sync_bytes = _resolve_sync_bytes(cfg)
    channels = [int(idx) for idx in channel_indices]
    positions = [float(pos) for pos in target_positions]
    return {
        "sync_bytes": sync_bytes.hex(" "),
        "address": f"0x{SET_ALL_CHANNELS_ADDR:02X}",
        "payload_length": len(channels),
        "channels": channels,
        "positions": positions,
        "endianness": _endianness_label(little_endian),
        "checksum": None,
        "expected_command_frame_size_bytes": 40,
        "expected_response_frame_size_bytes": 64,
        "note": "preview skeleton only; byte packing/little-endian fields/checksum still need final hardware calibration",
    }
