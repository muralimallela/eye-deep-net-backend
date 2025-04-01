import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image

# Load model
try:
    model = load_model("extension_model.h5")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Constants
IMG_HEIGHT, IMG_WIDTH = 32, 32
LABELS = ['DR', 'MH', 'Normal', 'ODC']  

def preprocess_image(image):
    """Preprocess an image for model prediction."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  
    image = image.astype(np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint for image prediction."""
    try:
        # Check if file is an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        # Model prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        return {"predicted_class": LABELS[predicted_class]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
