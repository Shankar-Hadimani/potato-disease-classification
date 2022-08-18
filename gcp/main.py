from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "gcs-potato-disease-classification-tf-model"
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

MODEL = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def predict(request):
    
    global model

    ## execute this only once.
    if model is None:
        download_blob(
        BUCKET_NAME,
        "models/potatoes.h5",
        "/tmp/potatoes.h5"
        )

        model = tf.keras.models.load_model("/tmp/potatoes.h5")
    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256,256)))
    image = image/255
    image_batch = tf.expand_dims(image, axis=0)
    predictions = model.predict(image_batch)
 
    CLASS_NAME = CLASS_NAMES[np.argmax(predictions[0])] ### select first as it's image batch[[...]]
    confidence = round(100 * (np.max(predictions[0])),2)

    return {"class": CLASS_NAME,"confidence": confidence }