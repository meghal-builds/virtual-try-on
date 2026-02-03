import random

def detect_pose_from_image(image_path: str):
    # Temporary dummy data (until Het implements real MediaPipe logic)
    
    left_shoulder = [random.randint(100, 200), random.randint(100, 200)]
    right_shoulder = [random.randint(200, 300), random.randint(100, 200)]
    
    shoulder_distance = abs(right_shoulder[0] - left_shoulder[0])
    
    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "shoulder_distance": shoulder_distance
    }
