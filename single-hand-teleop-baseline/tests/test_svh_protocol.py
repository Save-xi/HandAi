from svh.svh_protocol import (
    COMMAND_FRAME_SIZE_BYTES,
    COMMAND_PAYLOAD_SIZE_BYTES,
    RESPONSE_FRAME_SIZE_BYTES,
    RESPONSE_PAYLOAD_SIZE_BYTES,
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
    command_packet = build_set_all_channels_packet(
        list(range(9)),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.1, 0.0],
        target_ticks_preview=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        use_little_endian=True,
    )

    assert control_packet["address"] == "0x09"
    assert command_packet["address"] == "0x03"
    assert control_packet["sync_bytes"] == "4c aa"
    assert command_packet["sync_bytes"] == "4c aa"
    assert control_packet["endianness"] == "little"
    assert command_packet["endianness"] == "little"
    assert control_packet["payload_length"] == COMMAND_PAYLOAD_SIZE_BYTES
    assert command_packet["payload_length"] == COMMAND_PAYLOAD_SIZE_BYTES
    assert control_packet["command_frame_size_bytes"] == COMMAND_FRAME_SIZE_BYTES
    assert command_packet["command_frame_size_bytes"] == COMMAND_FRAME_SIZE_BYTES
    assert control_packet["response_payload_size_bytes"] == RESPONSE_PAYLOAD_SIZE_BYTES
    assert command_packet["response_frame_size_bytes"] == RESPONSE_FRAME_SIZE_BYTES
    assert control_packet["check1_preview_hex"] == "01"
    assert control_packet["check2_preview_hex"] == "01"
    assert command_packet["position_slots"] == 9
    assert command_packet["target_ticks_preview"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert command_packet["payload_preview_hex"] is not None
    assert command_packet["check1_preview_hex"] is not None
    assert command_packet["check2_preview_hex"] is not None


def test_protocol_preview_builders_allow_sync_override_from_config():
    cfg = {"svh_protocol_sync_bytes": [0x12, 0x34], "svh_protocol_use_little_endian": True}

    control_packet = build_set_control_state_packet(1, cfg=cfg)
    command_packet = build_set_all_channels_packet([0, 1], [0.1, 0.2], cfg=cfg)

    assert control_packet["sync_bytes"] == "12 34"
    assert command_packet["sync_bytes"] == "12 34"
