## Prerequisite:

### Download the TensorFlow Serving Docker image and repo
```
docker pull tensorflow/serving
```


# Potato Disease Classification

## Setup for Python:

```
conda create -n tensorflow python=3.10
pip install -r api/requirements.txt
```

3. Install Tensorflow Serving ([Setup instructions](https://www.tensorflow.org/tfx/serving/setup))



## Training the Model

1. Download the data from [kaggle](https://www.kaggle.com/arjuntejaswi/plant-village).
2. Only keep folders related to Potatoes.
3. Run Azure Databricks Notebook.
4. Open `training/potato-disease-training.ipynb` in Azure Databricks Notebook.
5. update the path to dataset.
6. Run all the Cells one by one.
7. Copy the model generated and save it with the version number in the `models` folder in DBFS Filestore.
8. Download saved models and copy it into saved_models folder within this project structure

## Running the API

### Using FastAPI

1. Get inside `api` folder

```bash
cd api
```

2. Run the FastAPI Server using uvicorn

```bash
uvicorn main:app --reload --host 0.0.0.0
```

3. Your API is now running at `0.0.0.0:8000`

### Using FastAPI & TF Serve

1. Get inside `api` folder

```bash
cd api
```

2. Copy the `models.config.example` as `models.config` and update the paths in file.
3. Run the TF Serve (Update config file path below)

```bash
docker run -t --rm -p 8501:8501 -v C:/<subfolders>/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease-classification/models.config
```

4. Run the FastAPI Server using uvicorn from main.py or main-tf-serving.py in vscode
   OR you can run it from command prompt as shown below,

```bash
cd api
uvicorn main-tf-serving:app --reload --host 0.0.0.0
```

5. Your API is now running at `0.0.0.0:8000`


## Setup for ReactJS

Install Nodejs and NPM ((https://nodejs.org/en/#home-downloadhead))

```bash
cd frontend
npm install --from-lock-json
npm audit fix
```

4. Copy `.env.example` as `.env`.

5. Change API url in `.env`.

## Running the Frontend

1. Get inside `api` folder

```bash
cd frontend
```

2. update `REACT_APP_API_URL` in`.env`
3. Run the frontend

```bash
npm run start
```

## Creating the TF Lite Model

1. Run Azure Databricks Notebook.
2. Open `training/tf-lite-converter.ipynb` in Azure Databricks Notebook.
3. update the path to dataset.
4. Run all the Cells one by one.
5. Model would be saved in `tf-lite-models` folder.

## Deploying the TF Lite on GCP

1. Create a [GCP account]
2. Create a [Project on GCP]
3. Create a [GCP bucket]
4. Upload the potatoes.h5 model from `gcp/saved_model_h5` directory into the bucket as `models/potatoes.h5`.
5. Install Google Cloud SDK ([Setup instructions]
6. Authenticate with Google Cloud SDK.

```bash
gcloud auth login
```

7. Run the deployment script.

```bash
cd gcp
gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project project_id
```

8. Your model is now deployed.
9. Use Postman to test the GCF using the [Trigger URL](https://cloud.google.com/functions/docs/calling/http).



## Deploying the TF Model (.h5) on GCP

1. Create a [GCP account](https://console.cloud.google.com/freetrial/signup/tos?_ga=2.25841725.1677013893.1627213171-706917375.1627193643&_gac=1.124122488.1627227734.Cj0KCQjwl_SHBhCQARIsAFIFRVVUZFV7wUg-DVxSlsnlIwSGWxib-owC-s9k6rjWVaF4y7kp1aUv5eQaAj2kEALw_wcB).
2. Create a [Project on GCP](https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project) (Keep note of the project id).
3. Create a [GCP bucket](https://console.cloud.google.com/storage/browser/).
4. Upload the tf .h5 model generate in the bucket in the path `models/potato-model.h5`.
5. Install Google Cloud SDK ([Setup instructions](https://cloud.google.com/sdk/docs/quickstarts)).
6. Authenticate with Google Cloud SDK.

```bash
gcloud auth login
```

7. Run the deployment script.

```bash
cd gcp
gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project project_id
```

8. Model is now deployed.
9. Use Postman to test the Google Cloud Functions

courtesy: https://github.com/codebasics
