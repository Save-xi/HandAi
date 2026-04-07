from __future__ import annotations

from functools import reduce
from operator import xor
from typing import Dict, Iterable

# 这是受论文启发的默认 sync header。
# 当前仍然只是 preview skeleton：
# 真正的 packet framing、padding 和 checksum 行为都必须在实现硬件传输前，
# 结合真实 SVH 设备协议继续验证。
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
        raise ValueError("svh_protocol_sync_bytes 必须恰好包含 2 个字节值")
    return bytes(values)


def _resolve_endianness(use_little_endian: bool | None, cfg: Dict | None = None) -> bool:
    if use_little_endian is not None:
        return bool(use_little_endian)
    if cfg is None:
        return True
    return bool(cfg.get("svh_protocol_use_little_endian", True))


def _pad_payload(payload: bytes, size: int = COMMAND_PAYLOAD_SIZE_BYTES) -> bytes:
    if len(payload) > size:
        raise ValueError(f"payload 长度 {len(payload)} 超过了 preview payload 大小 {size}")
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
    """为 SetControlState frame 构建一个 preview skeleton。

    这里刻意返回结构化的 preview，而不是宣称已经得到可直接下发的
    硬件 packet。论文描述了 sync bytes、address、length、payload
    和 checksum 等字段，但精确打包方式仍需在真实设备上继续校准。
    """

    little_endian = _resolve_endianness(use_little_endian, cfg)
    sync_bytes = _resolve_sync_bytes(cfg)
    payload = int(control_state).to_bytes(1, _endianness_label(little_endian), signed=False)
    padded_payload = _pad_payload(payload)
    return {
        "sync_bytes": sync_bytes.hex(" "),
        "address": f"0x{SET_CONTROL_STATE_ADDR:02X}",
        "address_low_nibble_command": f"0x{SET_CONTROL_STATE_ADDR & 0x0F:01X}",
        "address_upper_nibble_role": "在按通道寻址的命令里作为 channel selector",
        "payload_length": COMMAND_PAYLOAD_SIZE_BYTES,
        "payload_model": "40 字节命令 payload；当前 control-state preview 只使用首字节，其余部分补零",
        "payload_preview_hex": padded_payload.hex(" "),
        "endianness": _endianness_label(little_endian),
        "check1_preview_hex": f"{_check1(padded_payload):02x}",
        "check2_preview_hex": f"{_check2(padded_payload):02x}",
        "command_payload_size_bytes": COMMAND_PAYLOAD_SIZE_BYTES,
        "command_frame_size_bytes": COMMAND_FRAME_SIZE_BYTES,
        "response_payload_size_bytes": RESPONSE_PAYLOAD_SIZE_BYTES,
        "response_frame_size_bytes": RESPONSE_FRAME_SIZE_BYTES,
        "note": "仅为 preview skeleton；control-state payload 打包、checksum framing 和补零规则仍需真实 SVH 验证",
    }


def build_set_all_channels_packet(
    channel_indices: Iterable[int],
    target_positions: Iterable[float],
    target_ticks_preview: Iterable[int] | None = None,
    use_little_endian: bool | None = None,
    cfg: Dict | None = None,
) -> Dict:
    """为 SetControlCommand AllChannels frame 构建一个 preview skeleton。

    论文提到真实设备上，通道信息可能编码在 address 字段的高四位里。
    这里把它保留为后续校准事项，当前只暴露结构化 preview。
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
        "address_upper_nibble_role": "单通道命令时作为 channel selector；all-channels 命令保持高四位为空",
        "payload_length": COMMAND_PAYLOAD_SIZE_BYTES,
        "payload_model": "C# 参考驱动将 9 个 int32 目标位置（36 字节）再加 4 字节 padding，组成 40 字节 payload",
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
        "note": "仅为 preview skeleton；当提供 9 个 preview ticks 时，这里会把它们按有符号 int32 打包，但最终长度 / checksum framing 仍需真实硬件验证",
    }
