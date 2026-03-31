from perception.mediapipe_hand import MediaPipeHandDetector


def test_mediapipe_detector_init_close():
    detector = MediaPipeHandDetector(max_num_hands=1)
    detector.close()
