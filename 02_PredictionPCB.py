# Databricks notebook source
# DBTITLE 1,Download the model from the MLflowRepository
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

# DBTITLE 1,Create the UDF fonction to classify PCB images
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

# DBTITLE 1,Set the batch size to 64
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 64)

# COMMAND ----------

# DBTITLE 1,Compute a new table with the prediction for every images
spark.sql("drop table IF EXISTS circuit_board_prediction")
spark.table("circuit_board_gold").withColumn(
    "prediction", apply_vit("content")
).write.saveAsTable("circuit_board_prediction")

# COMMAND ----------

# DBTITLE 1,Display images with a wrong label
# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   circuit_board_prediction
# MAGIC where
# MAGIC   labelName != prediction.labelName

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Conclusion
# MAGIC 
# MAGIC That's it, we have build an end 2 end pipeline to incrementally ingest our dataset, clean it and train a Deep Learning model. The model is now deployed and ready for production-grade usage.
# MAGIC 
# MAGIC Databricks Lakehouse accelerate your team and simplify the go-to production:
# MAGIC 
# MAGIC 
# MAGIC * Unique ingestion and data preparation capabilities with autoloader making Data Engineering accessible to all
# MAGIC * Ability to support all use-cases ingest and process structured and non structured dataset
# MAGIC * Advanced ML capabilities for ML training
# MAGIC * MLOps coverage to let your Data Scientist team focus on what matters (improving your business) and not operational task
# MAGIC * Support for all type of production deployment to cover all your use case, without external tools
# MAGIC * Security and compliance covered all along, from data security (table ACL) to model governance
# MAGIC 
# MAGIC 
# MAGIC As result, Teams using Databricks are able to deploy in production advanced ML projects in a matter of weeks, from ingestion to model deployment, drastically accelerating business.
