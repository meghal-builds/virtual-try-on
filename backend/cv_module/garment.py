import cv2
import numpy as np


def apply_garment(base_image_path, garment_image_path, pose_data):
    base_img = cv2.imread(base_image_path)
    garment_img = cv2.imread(garment_image_path, cv2.IMREAD_UNCHANGED)

    if base_img is None or garment_img is None:
        return None

    if garment_img.shape[2] < 4:
        return None

    h_g, w_g = garment_img.shape[:2]

    # ==============================
    # 1️⃣ GARMENT CORNER POINTS
    # ==============================
    src_points = np.float32([
        [0, 0],          # top-left
        [w_g, 0],        # top-right
        [0, h_g],        # bottom-left
        [w_g, h_g]       # bottom-right
    ])

    # ==============================
    # 2️⃣ BODY TARGET POINTS
    # ==============================
    left_sh = pose_data["left_shoulder"]
    right_sh = pose_data["right_shoulder"]
    left_hip = pose_data["left_hip"]
    right_hip = pose_data["right_hip"]

    dst_points = np.float32([
        left_sh,
        right_sh,
        left_hip,
        right_hip
    ])

    # ==============================
    # 3️⃣ PERSPECTIVE TRANSFORM
    # ==============================
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    warped = cv2.warpPerspective(
        garment_img,
        matrix,
        (base_img.shape[1], base_img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # ==============================
    # 4️⃣ ALPHA BLEND
    # ==============================
    alpha = warped[:, :, 3] / 255.0
    alpha = np.dstack([alpha, alpha, alpha])

    garment_rgb = warped[:, :, :3]

    mask = np.array(pose_data["mask"], dtype=np.uint8)
    mask = np.dstack([mask, mask, mask])

    blended = alpha * garment_rgb + (1 - alpha) * base_img

    final = mask * blended + (1 - mask) * base_img

    return final.astype(np.uint8)
