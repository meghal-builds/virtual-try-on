from cv_module.pose import detect_pose_from_image
from cv_module.garment import apply_garment
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
import cv2

app = FastAPI()

from fastapi.staticfiles import StaticFiles

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

UPLOAD_FOLDER = "uploads"

# Create uploads folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.get("/")
def home():
    return {"message": "Virtual Try-On Backend Running"}


@app.get("/test")
def test():
    return {"status": "working"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={
        "message": "Image uploaded successfully",
        "filename": file.filename,
        "path": file_path
    })

@app.post("/detect-pose")
def detect_pose(filename: str):
    
    image_path = f"uploads/{filename}"
    
    result = detect_pose_from_image(image_path)
    
    return {
        "message": "Pose detected successfully",
        "data": result
    }
@app.post("/try-on")
def try_on(filename: str):
    base_image_path = f"uploads/{filename}"
    garment_image_path = "uploads/shirt.png"  # your garment image

    pose_data = detect_pose_from_image(base_image_path)

    if pose_data is None:
        return {"error": "Pose detection failed"}

    output_image = apply_garment(base_image_path, garment_image_path, pose_data)

    if output_image is None:
        return {"error": "Garment application failed"}

    output_path = f"uploads/output_{filename}"
    cv2.imwrite(output_path, output_image)

    return {
        "message": "Try-on successful",
        "output_image": f"/uploads/output_{filename}"
    }

