from __future__ import annotations

from functools import reduce
from operator import xor
from typing import Dict, Iterable

# Paper-informed default sync header. This is still only a preview skeleton:
# real packet framing, padding, and checksum behavior must be validated against
# the actual SVH device protocol before any hardware transport is implemented.
SYNC_BYTES = bytes([0x4C, 0xAA])
SET_CONTROL_STATE_ADDR = 0x09
SET_ALL_CHANNELS_ADDR = 0x03
COMMAND_PAYLOAD_SIZE_BYTES = 40
COMMAND_FRAME_SIZE_BYTES = 48
RESPONSE_PAYLOAD_SIZE_BYTES = 64
RESPONSE_FRAME_SIZE_BYTES = 72


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


def _pad_payload(payload: bytes, size: int = COMMAND_PAYLOAD_SIZE_BYTES) -> bytes:
    if len(payload) > size:
        raise ValueError(f"payload length {len(payload)} exceeds preview payload size {size}")
    return payload + bytes(size - len(payload))


def _check1(payload: bytes) -> int:
    return sum(payload) & 0xFF


def _check2(payload: bytes) -> int:
    return reduce(xor, payload, 0) & 0xFF


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
    padded_payload = _pad_payload(payload)
    return {
        "sync_bytes": sync_bytes.hex(" "),
        "address": f"0x{SET_CONTROL_STATE_ADDR:02X}",
        "address_low_nibble_command": f"0x{SET_CONTROL_STATE_ADDR & 0x0F:01X}",
        "address_upper_nibble_role": "channel selector on channel-addressed commands",
        "payload_length": COMMAND_PAYLOAD_SIZE_BYTES,
        "payload_model": "40-byte command payload; control-state preview currently uses the first byte and zero-pads the remainder",
        "payload_preview_hex": padded_payload.hex(" "),
        "endianness": _endianness_label(little_endian),
        "check1_preview_hex": f"{_check1(padded_payload):02x}",
        "check2_preview_hex": f"{_check2(padded_payload):02x}",
        "command_payload_size_bytes": COMMAND_PAYLOAD_SIZE_BYTES,
        "command_frame_size_bytes": COMMAND_FRAME_SIZE_BYTES,
        "response_payload_size_bytes": RESPONSE_PAYLOAD_SIZE_BYTES,
        "response_frame_size_bytes": RESPONSE_FRAME_SIZE_BYTES,
        "note": "preview skeleton only; control-state payload packing, checksum framing, and zero-padding rules still need real SVH verification",
    }


def build_set_all_channels_packet(
    channel_indices: Iterable[int],
    target_positions: Iterable[float],
    target_ticks_preview: Iterable[int] | None = None,
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
    tick_values = [int(v) for v in (target_ticks_preview or [])]
    payload_preview_hex = None
    check1_preview_hex = None
    check2_preview_hex = None
    if tick_values:
        packed = b"".join(int(v).to_bytes(4, _endianness_label(little_endian), signed=True) for v in tick_values[:9])
        padded_payload = _pad_payload(packed)
        payload_preview_hex = padded_payload.hex(" ")
        check1_preview_hex = f"{_check1(padded_payload):02x}"
        check2_preview_hex = f"{_check2(padded_payload):02x}"
    return {
        "sync_bytes": sync_bytes.hex(" "),
        "address": f"0x{SET_ALL_CHANNELS_ADDR:02X}",
        "address_low_nibble_command": f"0x{SET_ALL_CHANNELS_ADDR & 0x0F:01X}",
        "address_upper_nibble_role": "channel selector on single-channel commands; all-channels command keeps the upper nibble clear",
        "payload_length": COMMAND_PAYLOAD_SIZE_BYTES,
        "payload_model": "C# reference driver packs 9 int32 target positions (36 bytes) plus 4 bytes padding into a 40-byte payload",
        "position_slots": 9,
        "channels": channels,
        "positions": positions,
        "target_ticks_preview": tick_values,
        "payload_preview_hex": payload_preview_hex,
        "endianness": _endianness_label(little_endian),
        "check1_preview_hex": check1_preview_hex,
        "check2_preview_hex": check2_preview_hex,
        "command_payload_size_bytes": COMMAND_PAYLOAD_SIZE_BYTES,
        "command_frame_size_bytes": COMMAND_FRAME_SIZE_BYTES,
        "response_payload_size_bytes": RESPONSE_PAYLOAD_SIZE_BYTES,
        "response_frame_size_bytes": RESPONSE_FRAME_SIZE_BYTES,
        "note": "preview skeleton only; when 9 preview ticks are provided this builder packs them as signed int32 values, but final length/checksum framing still must be validated on real hardware",
    }
