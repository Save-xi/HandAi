from __future__ import annotations

from typing import Dict, Iterable, List

SYNC_BYTES = bytes([0xAA, 0x55])  # Preview placeholder until the real SVH frame header is confirmed.
SET_CONTROL_STATE_ADDR = 0x09
SET_ALL_CHANNELS_ADDR = 0x03


def _endianness_label(use_little_endian: bool) -> str:
    return "little" if use_little_endian else "big"


def build_set_control_state_packet(control_state: int, use_little_endian: bool = True) -> Dict:
    """Build a preview skeleton for the SetControlState frame.

    This intentionally returns a structured preview rather than claiming a fully
    validated hardware packet. Length/checksum details still need calibration
    against the real device protocol.
    """

    payload = int(control_state).to_bytes(1, _endianness_label(use_little_endian), signed=False)
    return {
        "sync_bytes": SYNC_BYTES.hex(" "),
        "address": f"0x{SET_CONTROL_STATE_ADDR:02X}",
        "payload_length": len(payload),
        "payload_preview_hex": payload.hex(),
        "endianness": _endianness_label(use_little_endian),
        "checksum": None,
        "note": "preview skeleton only; checksum/length framing must be verified on real SVH protocol",
    }


def build_set_all_channels_packet(
    channel_indices: Iterable[int],
    target_positions: Iterable[float],
    use_little_endian: bool = True,
) -> Dict:
    """Build a preview skeleton for the SetControlCommand AllChannels frame."""

    channels = [int(idx) for idx in channel_indices]
    positions = [float(pos) for pos in target_positions]
    return {
        "sync_bytes": SYNC_BYTES.hex(" "),
        "address": f"0x{SET_ALL_CHANNELS_ADDR:02X}",
        "payload_length": len(channels),
        "channels": channels,
        "positions": positions,
        "endianness": _endianness_label(use_little_endian),
        "checksum": None,
        "note": "preview skeleton only; byte packing/little-endian fields need final hardware calibration",
    }
