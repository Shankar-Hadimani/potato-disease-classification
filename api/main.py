from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io  import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_LOCATION = str(Path(__file__).absolute().parent) + "/../saved_models/0"
MODEL = tf.keras.models.load_model(MODEL_LOCATION)
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
    image = read_file_as_image(await file.read())

    """
    ### since model takes multiple images, 
    ### so to have batch of images, we need to rea dit as 2d array
    ### using np.expand_dims
    """
    img_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(img_batch)
    index = np.argmax(prediction[0])  ### zeroth prediction since it's from image batch [[]]

    predicted_class = CLASS_NAMES[index]
    confidence = np.max(prediction[0])

    return {
        'class':predicted_class,
        'confidence': float(confidence) }
    



if __name__=='__main__':
    uvicorn.run(app, host='localhost', port=8000)


