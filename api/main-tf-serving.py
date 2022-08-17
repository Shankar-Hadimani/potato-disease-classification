from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io  import BytesIO
from PIL import Image
import tensorflow as tf
import requests


app = FastAPI()

END_POINT="http://localhost:8501/v1/models/potatoes-disease-classification_model:predict"
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app.get("/ping")
async def ping():
    """
    This is just a health check call
    """
    return "Hello... It's a health-check and I'm alive."



def read_file_as_image(data) -> np.ndarray:
    """
    Read data in bytes and open it as Image from PILLOW(PIL)
    then convert it to array
    """
    image = Image.open(BytesIO(data))
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    file -  is an argument
            UploadFile is a datatype in FastAPI
            File is default value in Fast API
    """
    
    """
    ### since model takes multiple images, 
    ### so to have batch of images, we need to rea dit as 2d array
    ### using np.expand_dims
    """
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    json_data = {
        "instances":img_batch.tolist()

    }
    
    response = requests.post(END_POINT, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        'class':predicted_class,
        'confidence': float(confidence) }
    



if __name__=='__main__':
    uvicorn.run(app, host='localhost', port=8000)
    