# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/cv-quality-inspection. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/product-quality-inspection.

# COMMAND ----------

# MAGIC %md
# MAGIC #Using the model for inference in production
# MAGIC In the previous notebook we have trained our deep learning model and deployed it using the model registry. Here we will see how we can use the model for inference.
# MAGIC
# MAGIC In the first step we will need to download the model from MLflow repository

# COMMAND ----------

import os
import torch
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

model_name = "cv_pcb_classification"


local_path = ModelsArtifactRepository(
    f"models:/{model_name}/Production"
).download_artifacts(
    ""
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Classifying PCB images
# MAGIC We will now create the UDF function that will be used to classify the PCB images

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator
from io import BytesIO
from PIL import Image
from torchvision.models import ViT_B_16_Weights
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_model = torch.load(
    local_path + "data/model.pth", map_location=torch.device(device)
)

weights = ViT_B_16_Weights.DEFAULT
feature_extractor = weights.transforms()

feature_extractor_b = sc.broadcast(feature_extractor)
model_b = sc.broadcast(loaded_model)

@pandas_udf("struct<score: float, label: int, labelName: string>")
def apply_vit(images_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:

    model = model_b.value
    feature_extractor = feature_extractor_b.value
    model = model.to(torch.device("cuda"))
    model.eval()
    id2label = {0: "normal", 1: "anomaly"}
    with torch.set_grad_enabled(False):
        for images in images_iter:
            pil_images = torch.stack(
                [
                    feature_extractor(Image.open(BytesIO(b)).convert("RGB"))
                    for b in images
                ]
            )
            pil_images = pil_images.to(torch.device(device))
            outputs = model(pil_images)
            preds = torch.max(outputs, 1)[1].tolist()
            probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1].tolist()
            yield pd.DataFrame(
                [
                    {"score": prob, "label": pred, "labelName": id2label[pred]}
                    for pred, prob in zip(preds, probs)
                ]
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set batch size
# MAGIC Let us set the batch size to 64 using the `maxRecordsPerBatch` parameter, for when data partitions in Spark are converted to Arrow record batches

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 64)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction table
# MAGIC We can now compute a new table with the predictions done for every image

# COMMAND ----------

spark.sql("drop table IF EXISTS circuit_board_prediction")
spark.table("circuit_board_gold").withColumn(
    "prediction", apply_vit("content")
).write.saveAsTable("circuit_board_prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display images that are wrongly labelled
# MAGIC Display images with a wrong label using simple SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   circuit_board_prediction
# MAGIC where
# MAGIC   labelName != prediction.labelName

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy it to our rest REST serverless real-time inference endpoints
# MAGIC
# MAGIC Let us deploy the model to a REST serverless real-time inference endpoint.
# MAGIC
# MAGIC But first let us create a wrapper model to be able to accept base64 images as input and publish it to MLflow

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
import base64
from PIL import Image
import io
import mlflow
from io import BytesIO

from torchvision.models import ViT_B_16_Weights



class CVModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        # instantiate model in evaluation mode
        model.to(torch.device("cpu"))
        self.model = model.eval()
        weights = ViT_B_16_Weights.DEFAULT
        self.feature_extractor = weights.transforms()

    def predict(self, context, images):
        with torch.set_grad_enabled(False):
          id2label = {0: "normal", 1: "anomaly"}
          pil_images = torch.stack(
              [
                  self.feature_extractor(
                      Image.open(BytesIO(base64.b64decode(row[0]))).convert("RGB")
                  )
                  for _, row in images.iterrows()
              ]
          )
          pil_images = pil_images.to(torch.device("cpu"))
          outputs = self.model(pil_images)
          preds = torch.max(outputs, 1)[1]
          probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1]
          labels = [id2label[pred] for pred in preds.tolist()]

          return pd.DataFrame( data=dict(
            score=probs,
            label=preds,
            labelName=labels)
          )

# COMMAND ----------

loaded_model = torch.load(
    local_path + "data/model.pth", map_location=torch.device("cpu")
)
wrapper = CVModelWrapper(loaded_model)
images = spark.table("circuit_board_gold").take(25)

b64image1 = base64.b64encode(images[0]["content"]).decode("ascii")
b64image2 = base64.b64encode(images[1]["content"]).decode("ascii")
b64image3 = base64.b64encode(images[3]["content"]).decode("ascii")
b64image4 = base64.b64encode(images[4]["content"]).decode("ascii")
b64image24 = base64.b64encode(images[24]["content"]).decode("ascii")

df_input = pd.DataFrame(
    [b64image1, b64image2, b64image3, b64image4, b64image24], columns=["data"]
)
df = wrapper.predict("", df_input)
display(df)

# COMMAND ----------

username = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
mlflow.set_experiment("/Users/{}/pcbqi".format(username))
model_name = "cv_pcb_classification_rt"
with mlflow.start_run(run_name=model_name) as run:
    mlflowModel = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapper,
        input_example=df_input,
        registered_model_name=model_name,
    )

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
latest_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
client.transition_model_version_stage(
    name=model_name, version=latest_version, stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now deploy this new model to our serverless real-time serving endpoint.
# MAGIC
# MAGIC Use the example and click "send request".
# MAGIC
# MAGIC <img width="1000px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/serving.png">

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Conclusion
# MAGIC
# MAGIC That is it! We have built an end-toend pipeline to incrementally ingest our dataset, clean it, and train a deep learning model. The production-grade pipeline and model is now deployed and ready for use.
# MAGIC
# MAGIC Databricks Lakehouse accelerate your team and simplify the go-to production:
# MAGIC
# MAGIC * Unique ingestion and data preparation capabilities with autoloader making Data Engineering accessible to all
# MAGIC * Ability to support all use-cases ingest and process structured and non structured dataset
# MAGIC * Advanced ML capabilities for ML training
# MAGIC * MLOps coverage to let your Data Scientist team focus on what matters (improving your business) and not operational task
# MAGIC * Support for all type of production deployment to cover all your use case, without external tools
# MAGIC * Security and compliance covered all along, from data security (table ACL) to model governance
# MAGIC
# MAGIC
# MAGIC As a result, teams using Databricks are able to deploy in production advanced ML projects in a matter of weeks, from ingestion to model deployment, drastically accelerating business.
