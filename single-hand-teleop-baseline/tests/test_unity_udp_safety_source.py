from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNITY_SCRIPT = (
    PROJECT_ROOT
    / "integrations"
    / "unity_phase15_snapshot"
    / "Assets"
    / "Scripts"
    / "RobotControlScript.cs"
)


def test_unity_udp_receiver_is_loopback_only_and_bounded():
    source = UNITY_SCRIPT.read_text(encoding="utf-8")

    assert "new IPEndPoint(IPAddress.Loopback, baselineUdpListenPort)" in source
    assert "packet.Length > Math.Max(1024, baselineUdpMaxPacketBytes)" in source
    assert "new UdpClient(baselineUdpListenPort)" not in source


def test_unity_enforces_validity_order_watchdog_and_virtual_only_udp():
    source = UNITY_SCRIPT.read_text(encoding="utf-8")

    for condition in (
        "packet.detected",
        "packet.control_ready",
        "packet.svh_preview.valid",
        "control.valid",
        "control.features_valid",
        "control.command_ready",
    ):
        assert condition in source
    assert "TryAcceptBaselinePacketOrder" in source
    assert "CheckBaselineUdpWatchdog" in source
    assert "ApplyBaselineSafeOpen" in source
    assert "BaselineUdpHardwareForwardingCompiled = false" in source
    assert "ApplyBaselinePreviewTargets(expandedTargets, applyBaselinePreviewToHardware)" not in source


def test_legacy_hardware_requires_explicit_master_gate():
    source = UNITY_SCRIPT.read_text(encoding="utf-8")

    assert "private bool allowLegacyHardwareControl = false" in source
    assert "if (!allowLegacyHardwareControl" in source
    assert "已忽略本次 COM/IP 参数" in source
