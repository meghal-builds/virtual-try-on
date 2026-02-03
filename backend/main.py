from cv_module.pose import detect_pose_from_image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil

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

