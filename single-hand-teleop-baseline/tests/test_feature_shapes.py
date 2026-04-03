from features.hand_features import empty_features, extract_hand_features, invalidate_control_features


def _make_hand_pose(curled: bool = False, thumb_folded: bool = False):
    xyz = [
        (0.0, 0.0, 0.0),  # wrist
        (-1.0, 0.5, 0.0),  # thumb cmc
        (-1.4, 1.0, 0.0),  # thumb mcp
        (-1.8, 1.4, 0.0),  # thumb ip
        (-2.2, 1.8, 0.0),  # thumb tip
        (-0.8, 1.0, 0.0),  # index mcp
        (-0.8, 2.0, 0.0),  # index pip
        (-0.8, 3.0, 0.0),  # index dip
        (-0.8, 4.0, 0.0),  # index tip
        (0.0, 1.0, 0.0),  # middle mcp
        (0.0, 2.1, 0.0),  # middle pip
        (0.0, 3.2, 0.0),  # middle dip
        (0.0, 4.3, 0.0),  # middle tip
        (0.8, 1.0, 0.0),  # ring mcp
        (0.8, 2.0, 0.0),  # ring pip
        (0.8, 3.0, 0.0),  # ring dip
        (0.8, 4.0, 0.0),  # ring tip
        (1.6, 1.0, 0.0),  # little mcp
        (1.6, 1.9, 0.0),  # little pip
        (1.6, 2.8, 0.0),  # little dip
        (1.6, 3.7, 0.0),  # little tip
    ]

    if thumb_folded:
        xyz[2] = (-1.2, 0.8, 0.0)
        xyz[3] = (-0.8, 0.55, -0.20)
        xyz[4] = (-0.25, 0.45, -0.40)

    if curled:
        curled_coords = {
            6: (-0.8, 1.8, 0.0),
            7: (-0.1, 1.75, -0.25),
            8: (0.35, 1.1, -0.45),
            10: (0.0, 1.9, 0.0),
            11: (0.75, 1.85, -0.20),
            12: (1.15, 1.15, -0.45),
            14: (0.8, 1.8, 0.0),
            15: (1.45, 1.7, -0.18),
            16: (1.75, 1.05, -0.38),
            18: (1.6, 1.65, 0.0),
            19: (2.15, 1.55, -0.12),
            20: (2.35, 0.95, -0.28),
        }
        for idx, point in curled_coords.items():
            xyz[idx] = point

    xy = [(x, y) for x, y, _ in xyz]
    return xy, xyz


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
    assert feat["gesture_stable"] is None
    assert set(feat["finger_curl"].keys()) == {"thumb", "index", "middle", "ring", "little"}
    assert len(feat["landmarks_2d"]) == 21
    assert len(feat["landmarks_3d"]) == 21
    for value in feat["finger_curl"].values():
        assert 0.0 <= value <= 1.0
    assert 0.0 <= feat["pinch_distance_norm"]
    assert 0.0 <= feat["hand_open_ratio"]


def test_empty_feature_structure():
    feat = empty_features(123.0)

    assert feat["detected"] is False
    assert feat["handedness"] is None
    assert feat["confidence"] is None
    assert feat["gesture_raw"] == "unknown"
    assert feat["gesture_stable"] == "unknown"
    assert feat["pinch_distance_norm"] is None
    assert feat["landmarks_2d"] == []
    assert feat["landmarks_3d"] == []


def test_invalidate_control_features_preserves_detection_but_clears_control_fields():
    lms = [(0.5 + i * 0.001, 0.5 + i * 0.001) for i in range(21)]
    feat = extract_hand_features(lms, "Right", 0.9, 123.0)
    degraded = invalidate_control_features(feat)

    assert degraded["detected"] is True
    assert degraded["handedness"] == "Right"
    assert degraded["confidence"] == 0.9
    assert degraded["gesture_raw"] is None
    assert degraded["gesture_stable"] is None
    assert degraded["pinch_distance_norm"] is None
    assert degraded["hand_open_ratio"] is None
    assert degraded["landmarks_2d"] == feat["landmarks_2d"]
    assert degraded["landmarks_3d"] == feat["landmarks_3d"]
    assert degraded["finger_curl"] == {
        "thumb": None,
        "index": None,
        "middle": None,
        "ring": None,
        "little": None,
    }


def test_extract_hand_features_returns_empty_payload_for_incomplete_landmarks():
    feat = extract_hand_features([(0.5, 0.5)] * 3, "Right", 0.9, 123.0)

    assert feat == empty_features(123.0)


def test_extract_hand_features_falls_back_to_zero_z_when_xyz_is_missing_or_misaligned():
    lms = [(0.5 + i * 0.001, 0.5 + i * 0.001) for i in range(21)]
    feat = extract_hand_features(lms, "Right", 0.9, 123.0, landmarks_xyz=[(0.0, 0.0, 0.0)] * 10)

    assert len(feat["landmarks_3d"]) == 21
    assert all(point[2] == 0.0 for point in feat["landmarks_3d"])


def test_extract_hand_features_falls_back_to_zero_z_when_xyz_points_are_malformed():
    lms = [(0.5 + i * 0.001, 0.5 + i * 0.001) for i in range(21)]
    malformed_xyz = [(0.0, 0.0, 0.0)] * 20 + [(0.0, 0.0)]

    feat = extract_hand_features(lms, "Right", 0.9, 123.0, landmarks_xyz=malformed_xyz)

    assert len(feat["landmarks_3d"]) == 21
    assert all(point[2] == 0.0 for point in feat["landmarks_3d"])


def test_finger_curl_uses_joint_geometry_and_z_to_separate_open_and_fist():
    open_xy, open_xyz = _make_hand_pose(curled=False, thumb_folded=False)
    fist_xy, fist_xyz = _make_hand_pose(curled=True, thumb_folded=True)

    open_feat = extract_hand_features(open_xy, "Right", 0.9, 123.0, landmarks_xyz=open_xyz)
    fist_feat = extract_hand_features(fist_xy, "Right", 0.9, 124.0, landmarks_xyz=fist_xyz)

    for finger in ["index", "middle", "ring", "little"]:
        assert open_feat["finger_curl"][finger] < 0.05
        assert fist_feat["finger_curl"][finger] > open_feat["finger_curl"][finger] + 0.20

    assert open_feat["finger_curl"]["thumb"] < 0.05
    assert fist_feat["finger_curl"]["thumb"] > open_feat["finger_curl"]["thumb"] + 0.15
