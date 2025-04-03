import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File, HTTPException
import io
from PIL import Image


try:
    model = load_model("extension_model.keras")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")


app = FastAPI()


IMG_HEIGHT, IMG_WIDTH = 32, 32
LABELS = ['DR', 'MH', 'Normal', 'ODC']  

def preprocess_image(image):
    """Preprocess an image for model prediction."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  
    image = image.astype(np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def predict_image(image_path):
    """Predict fish species from image path and display output."""
    image = cv2.imread(image_path)
    img_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    img_array = np.array(img_resized).reshape(1, IMG_HEIGHT, IMG_WIDTH, 3) 
    img_array = img_array.astype('float32') / 255 
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    img_display = cv2.imread(image_path)
    img_display = cv2.resize(img_display, (400, 300))
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    cv2.putText(img_display, f'Predicted As: {LABELS[predicted_class]}', 
                (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return LABELS[predicted_class]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint for image prediction."""
    try:
       
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed_image = preprocess_image(image)

    
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        return {"predicted_class": LABELS[predicted_class]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


