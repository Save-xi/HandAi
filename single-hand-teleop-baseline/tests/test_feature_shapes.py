from features.hand_features import empty_features, extract_hand_features


def test_feature_structure():
    # 21 pseudo landmarks in normalized image coordinates
    lms = [(0.5 + i * 0.001, 0.5 + i * 0.001) for i in range(21)]
    feat = extract_hand_features(lms, "Right", 0.9, 123.0)

    assert feat["detected"] is True
    assert feat["handedness"] == "Right"
    assert isinstance(feat["pinch_distance_norm"], float)
    assert isinstance(feat["hand_open_ratio"], float)
    assert feat["confidence"] == 0.9
    assert feat["gesture_raw"] is None
    assert feat["gesture"] is None
    assert set(feat["finger_curl"].keys()) == {"thumb", "index", "middle", "ring", "little"}
    assert len(feat["landmarks_2d"]) == 21


def test_empty_feature_structure():
    feat = empty_features(123.0)

    assert feat["detected"] is False
    assert feat["handedness"] is None
    assert feat["confidence"] is None
    assert feat["gesture_raw"] == "unknown"
    assert feat["gesture"] == "unknown"
    assert feat["pinch_distance_norm"] is None
    assert feat["landmarks_2d"] == []
