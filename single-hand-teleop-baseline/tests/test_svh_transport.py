from svh.svh_transport_mock import MockSvhTransport


def test_mock_transport_records_latest_command():
    transport = MockSvhTransport()
    command = {
        "enabled": True,
        "mode": "preview",
        "valid": True,
        "command_source": "control_representation",
        "target_channels": [0, 1, 2, 3, 4],
        "target_positions": [0.1, 0.2, 0.3, 0.4, 0.5],
        "protocol_hint": {"set_control_state_addr": "0x09", "set_all_channels_addr": "0x03", "transport": "mock"},
    }

    result = transport.send(command)

    assert result["transport"] == "mock"
    assert result["accepted"] is True
    assert result["recorded_count"] == 1
    assert transport.last_command == command
    assert transport.sent_commands[-1] == command


def test_mock_transport_keeps_bounded_history_but_monotonic_total_count():
    transport = MockSvhTransport(history_size=3)

    result = None
    for index in range(100):
        result = transport.send({"valid": True, "frame": index})

    assert result is not None
    assert result["recorded_count"] == 100
    assert result["retained_history_count"] == 3
    assert len(transport.sent_commands) == 3
    assert [item["frame"] for item in transport.sent_commands] == [97, 98, 99]
