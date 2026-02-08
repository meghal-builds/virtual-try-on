import cv2
import numpy as np
import math
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation


def detect_pose_from_image(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        return None

    height, width, _ = image.shape

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        # ==============================
        # SHOULDER POINTS
        # ==============================
        left_sh_lm = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_shoulder = [
            int(left_sh_lm.x * width),
            int(left_sh_lm.y * height)
        ]

        right_shoulder = [
            int(right_sh_lm.x * width),
            int(right_sh_lm.y * height)
        ]
        

        # ==============================
        # HIP POINTS
        # ==============================
        left_hip_lm = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        left_hip = [
            int(left_hip_lm.x * width),
            int(left_hip_lm.y * height)
        ]

        right_hip = [
            int(right_hip_lm.x * width),
            int(right_hip_lm.y * height)
        ]


        # Fix flipped shoulders
        if left_shoulder[0] > right_shoulder[0]:
            left_shoulder, right_shoulder = right_shoulder, left_shoulder
            left_hip, right_hip = right_hip, left_hip

        # ==============================
        # DISTANCES
        # ==============================
        shoulder_distance = int(
            math.sqrt(
                (right_shoulder[0] - left_shoulder[0]) ** 2 +
                (right_shoulder[1] - left_shoulder[1]) ** 2
            )
        )

        hip_distance = int(
            math.sqrt(
                (right_hip[0] - left_hip[0]) ** 2 +
                (right_hip[1] - left_hip[1]) ** 2
            )
        )

        # ==============================
        # TORSO HEIGHT
        # ==============================
        torso_height = abs(left_hip[1] - left_shoulder[1])

        # ==============================
        # SHOULDER ANGLE
        # ==============================
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        shoulder_angle = math.degrees(math.atan2(dy, dx))

        # ==============================
        # MIDPOINT
        # ==============================
        mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
        mid_y = (left_shoulder[1] + right_shoulder[1]) // 2

    # ==============================
    # SELFIE SEGMENTATION
    # ==============================
    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        results_seg = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = (results_seg.segmentation_mask > 0.5).astype(np.uint8)

    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "shoulder_distance": shoulder_distance,
        "hip_distance": hip_distance,
        "torso_height": torso_height,
        "shoulder_angle": shoulder_angle,
        "shoulder_midpoint": [mid_x, mid_y],
        "mask": mask.tolist()
    }
