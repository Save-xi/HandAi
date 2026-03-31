from svh.svh_protocol import (
    SET_ALL_CHANNELS_ADDR,
    SET_CONTROL_STATE_ADDR,
    SYNC_BYTES,
    build_set_all_channels_packet,
    build_set_control_state_packet,
)


def test_svh_protocol_constants_exist():
    assert SET_CONTROL_STATE_ADDR == 0x09
    assert SET_ALL_CHANNELS_ADDR == 0x03
    assert SYNC_BYTES == bytes([0x4C, 0xAA])


def test_protocol_preview_builders_return_expected_addresses():
    control_packet = build_set_control_state_packet(1, use_little_endian=True)
    command_packet = build_set_all_channels_packet([0, 1], [0.1, 0.2], use_little_endian=True)

    assert control_packet["address"] == "0x09"
    assert command_packet["address"] == "0x03"
    assert control_packet["sync_bytes"] == "4c aa"
    assert command_packet["sync_bytes"] == "4c aa"
    assert control_packet["endianness"] == "little"
    assert command_packet["endianness"] == "little"


def test_protocol_preview_builders_allow_sync_override_from_config():
    cfg = {"svh_protocol_sync_bytes": [0x12, 0x34], "svh_protocol_use_little_endian": True}

    control_packet = build_set_control_state_packet(1, cfg=cfg)
    command_packet = build_set_all_channels_packet([0, 1], [0.1, 0.2], cfg=cfg)

    assert control_packet["sync_bytes"] == "12 34"
    assert command_packet["sync_bytes"] == "12 34"
