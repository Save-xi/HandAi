from features.hand_features import extract_hand_features


def test_feature_structure():
    # 21 pseudo landmarks in normalized image coordinates
    lms = [(0.5 + i * 0.001, 0.5 + i * 0.001) for i in range(21)]
    feat = extract_hand_features(lms, "Right", 0.9, 123.0)

    assert feat["detected"] is True
    assert feat["handedness"] == "Right"
    assert isinstance(feat["pinch_distance"], float)
    assert isinstance(feat["hand_open_ratio"], float)
    assert set(feat["finger_curl"].keys()) == {"thumb", "index", "middle", "ring", "little"}
    assert len(feat["landmarks_2d"]) == 21
