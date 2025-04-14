import os
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import uvicorn
from io import BytesIO
from PIL import Image
import sys
from starlette.status import HTTP_403_FORBIDDEN

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

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "default-development-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Model loading - do this at startup
inference_config = InferenceConfig()
model = None

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API key"
        )
    return api_key

@app.on_event("startup")
async def load_model():
    global model

    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                             config=inference_config,
                             model_dir=DEFAULT_LOGS_DIR)

    # Load weights - use environment variable or default path
    model_path = os.getenv('MODEL_PATH', 'logs/object20250403T2351/mask_rcnn_object_0010.h5')
    print(f"Loading weights from {model_path}")
    model.load_weights(model_path, by_name=True)
    print("Model loaded successfully")

@app.get("/")
async def root():
    return {"message": "Wheat Seed Segmentation API is running. Use /analyze-seeds endpoint to upload an image."}

@app.post("/analyze-seeds", dependencies=[Depends(get_api_key)])
async def analyze_seeds(file: UploadFile = File(...)):
    try:
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
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during processing: {str(e)}"}
        )

# Set port for Render compatibility
port = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port)
