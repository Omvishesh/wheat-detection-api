import os
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image
import sys

# Mask R-CNN imports
from mrcnn.config import Config
from mrcnn import model as modellib

# Root directory
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory for logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Configuration for inference
class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "object"

    # GPU settings
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + healthy + defective

    # Detection thresholds
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

# Initialize FastAPI
app = FastAPI(title="Wheat Seed Segmentation API",
              description="API for analyzing wheat seeds and counting healthy vs defective seeds")

# Model loading - do this at startup
inference_config = InferenceConfig()
model = None

@app.on_event("startup")
async def load_model():
    global model

    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                             config=inference_config,
                             model_dir=DEFAULT_LOGS_DIR)

    # Load weights
    model_path = 'logs/object20250403T2351/mask_rcnn_object_0010.h5'
    print(f"Loading weights from {model_path}")
    model.load_weights(model_path, by_name=True)
    print("Model loaded successfully")

@app.get("/")
async def root():
    return {"message": "Wheat Seed Segmentation API is running. Use /analyze-seeds endpoint to upload an image."}

@app.post("/analyze-seeds")
async def analyze_seeds(file: UploadFile = File(...)):
    # Read and process the uploaded image
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    image_array = np.array(image)

    # Check if image is in the right format
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        # Convert grayscale to RGB if needed
        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
        else:
            return JSONResponse(status_code=400,
                               content={"error": "Please upload a valid RGB or grayscale image"})

    # Run detection
    results = model.detect([image_array], verbose=0)
    r = results[0]  # We only processed one image

    # Count objects by class
    class_names = {1: "healthy", 2: "defective"}

    # Initialize counters
    healthy_count = 0
    defective_count = 0

    # Count instances by class
    for class_id in r['class_ids']:
        if class_id == 1:  # healthy
            healthy_count += 1
        elif class_id == 2:  # defective
            defective_count += 1

    # Prepare response
    response = {
        "total_seeds": len(r['class_ids']),
        "healthy_seeds": healthy_count,
        "defective_seeds": defective_count
    }

    return response

if __name__ == "__main__":
    uvicorn.run("main:app",port=8000, reload=True)