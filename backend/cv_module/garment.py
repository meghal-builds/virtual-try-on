import cv2
import numpy as np


def apply_garment(base_image_path, garment_image_path, pose_data):
    base_img = cv2.imread(base_image_path)
    garment_img = cv2.imread(garment_image_path, cv2.IMREAD_UNCHANGED)

    if base_img is None or garment_img is None:
        return None

    if garment_img.shape[2] < 4: 
        return None

    shoulder_distance = pose_data["shoulder_distance"]
    torso_height = pose_data["torso_height"]
    midpoint = pose_data["shoulder_midpoint"]
    mask = np.array(pose_data["mask"], dtype=np.uint8)

    angle = pose_data["shoulder_angle"]

    # Rotate garment
    (h, w) = garment_img.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    # compute new bounding dimensions
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # adjust rotation matrix
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_garment = cv2.warpAffine(
        garment_img,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    garment_img = rotated_garment


    garment_height, garment_width = garment_img.shape[:2]

    # Scaling
    width_scale = (shoulder_distance * 1.45) / garment_width
    height_scale = (torso_height * 1.2) / garment_height
    scale_factor = (width_scale + height_scale) / 2

    new_width = int(garment_width * scale_factor)
    new_height = int(garment_height * scale_factor)

    resized_garment = cv2.resize(garment_img, (new_width, new_height))

    x_offset = midpoint[0] - new_width // 2
    y_offset = midpoint[1] - int(new_height * 0.35)

    # Clipping
    y1 = max(y_offset, 0)
    y2 = min(y_offset + new_height, base_img.shape[0])
    x1 = max(x_offset, 0)
    x2 = min(x_offset + new_width, base_img.shape[1])

    garment_y1 = max(0, -y_offset)
    garment_y2 = garment_y1 + (y2 - y1)

    garment_x1 = max(0, -x_offset)
    garment_x2 = garment_x1 + (x2 - x1)

    base_region = base_img[y1:y2, x1:x2]
    garment_region = resized_garment[garment_y1:garment_y2, garment_x1:garment_x2]

    alpha = garment_region[:, :, 3] / 255.0
    alpha = np.dstack([alpha, alpha, alpha])

    garment_rgb = garment_region[:, :, :3]

    blended = (alpha * garment_rgb + (1 - alpha) * base_region).astype(np.uint8)

    # -------- APPLY SEGMENTATION MASK --------
    body_region = mask[y1:y2, x1:x2]
    body_region = np.dstack([body_region, body_region, body_region])

    final = body_region * blended + (1 - body_region) * base_region

    base_img[y1:y2, x1:x2] = final.astype(np.uint8)

    return base_img
