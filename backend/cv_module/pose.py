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

        # Extract landmarks
        left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        left_shoulder = [int(left.x * width), int(left.y * height)]
        right_shoulder = [int(right.x * width), int(right.y * height)]
        left_hip_point = [int(left_hip.x * width), int(left_hip.y * height)]

        # Shoulder distance
        shoulder_distance = int(
            math.sqrt(
                (right_shoulder[0] - left_shoulder[0]) ** 2 +
                (right_shoulder[1] - left_shoulder[1]) ** 2
            )
        )

        # Shoulder angle (for garment rotation)
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        shoulder_angle = -math.degrees(math.atan2(dy, dx))

        # Midpoint
        mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
        mid_y = (left_shoulder[1] + right_shoulder[1]) // 2

        # Torso height
        torso_height = abs(left_hip_point[1] - left_shoulder[1])

    # -------- SELFIE SEGMENTATION --------
    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        results_seg = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = results_seg.segmentation_mask

        # Convert to binary mask
        mask = (mask > 0.5).astype(np.uint8)

    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "shoulder_distance": shoulder_distance,
        "shoulder_midpoint": [mid_x, mid_y],
        "torso_height": torso_height,
        "shoulder_angle": shoulder_angle,
        "mask": mask.tolist()
    }
