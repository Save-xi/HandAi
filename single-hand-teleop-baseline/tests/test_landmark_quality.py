from perception.landmark_quality import assess_control_readiness


def test_landmark_quality_accepts_centered_hand():
    cfg = {
        "control_ready_min_in_bounds_ratio": 0.90,
        "control_ready_palm_center_margin": 0.08,
        "control_ready_palm_core_oob_tolerance": 0.02,
    }
    landmarks = [(0.30 + (i % 5) * 0.04, 0.35 + (i // 5) * 0.04) for i in range(21)]

    quality = assess_control_readiness(landmarks, cfg)

    assert quality["control_ready"] is True
    assert quality["in_bounds_ratio"] == 1.0


def test_landmark_quality_rejects_out_of_frame_hand():
    cfg = {
        "control_ready_min_in_bounds_ratio": 0.90,
        "control_ready_palm_center_margin": 0.08,
        "control_ready_palm_core_oob_tolerance": 0.02,
    }
    landmarks = [(0.30 + (i % 5) * 0.04, 0.35 + (i // 5) * 0.04) for i in range(21)]
    shifted = [(x + 0.75, y) for x, y in landmarks]

    quality = assess_control_readiness(shifted, cfg)

    assert quality["control_ready"] is False
    assert quality["in_bounds_ratio"] < 0.90


def test_landmark_quality_rejects_border_hugging_palm_even_if_still_in_bounds():
    cfg = {
        "control_ready_min_in_bounds_ratio": 0.90,
        "control_ready_palm_center_margin": 0.08,
        "control_ready_palm_core_oob_tolerance": 0.02,
    }
    landmarks = [(0.01 + (i % 5) * 0.02, 0.25 + (i // 5) * 0.04) for i in range(21)]

    quality = assess_control_readiness(landmarks, cfg)

    assert quality["control_ready"] is False
    assert quality["in_bounds_ratio"] == 1.0
