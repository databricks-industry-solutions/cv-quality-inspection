# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/cv-quality-inspection. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/product-quality-inspection.

# COMMAND ----------

# MAGIC %md
# MAGIC # Product Quality Inspection of Printed Circuit Board (PCB) using Computer Vision and Real-time Serverless Inference
# MAGIC
# MAGIC <div style="float:right">
# MAGIC <img width="500px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/PCB1.png">
# MAGIC </div>
# MAGIC
# MAGIC In this solution accelerator, we will show you how Databricks can help you to deploy an end-to-end pipeline for product quality inspection. The model is deployed using Databricks [Serverless Real-time Inference](https://docs.databricks.com/archive/serverless-inference-preview/serverless-real-time-inference.html).
# MAGIC
# MAGIC We will use the [Visual Anomaly (VisA)](https://registry.opendata.aws/visa/) detection dataset, and build a pipeline to detect anomalies in our PCB images. 
# MAGIC
# MAGIC ## Why image quality inspection?
# MAGIC
# MAGIC Image quality inspection is a common challenge in the context of manufacturing. It is key to delivering Smart Manufacturing.
# MAGIC
# MAGIC ## Implementing a production-grade pipeline
# MAGIC
# MAGIC The image classification problem has been eased in recent years with pre-trained deep learning models, transfer learning, and higher-level frameworks. While a data science team can quickly deploy such a model, a real challenge remains in the implementation of a production-grade, end-to-end pipeline, consuming images and requiring MLOps/governance, and ultimately delivering results.
# MAGIC
# MAGIC Databricks Lakehouse is designed to make this overall process simple, letting Data Scientist focus on the core use-case.
# MAGIC
# MAGIC In order to build the quality inspection model, we use Torchvision. However, the same architecture may be used with other libraries. The Torchvision library is part of the PyTorch project, a popular framework for deep learning. Torchvision comes with model architectures, popular datasets, and image transformations. 
# MAGIC
# MAGIC
# MAGIC The first step in building the pipeline is data ingestion. Databricks enables the loading of any source of data, even images (unstructured data). This is stored in a table with the content of the image and also the associated label in a efficient and a distributed way.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality inspection image pipeline
# MAGIC
# MAGIC This is the pipeline we will be building. We ingest 2 datasets, namely:
# MAGIC
# MAGIC * The raw satellite images (jpg) containing PCB
# MAGIC * The label, the type of anomalies saved as CSV files
# MAGIC
# MAGIC We will first focus on building a data pipeline to incrementally load this data and create a final Gold table.
# MAGIC
# MAGIC This table will then be used to train a ML Classification model to learn to detect anomalies in our images in real time!
# MAGIC
# MAGIC <img width="1000px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/pipeline.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the dataset from https://registry.opendata.aws/visa/
# MAGIC
# MAGIC We will use `bash` commands to download the dataset from [https://registry.opendata.aws/visa/](https://registry.opendata.aws/visa/)
# MAGIC
# MAGIC As the data are on AWS S3, we need to install the AWS CLI library (`awscli`).

# COMMAND ----------

# MAGIC %pip install awscli

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /tmp/data
# MAGIC aws s3 cp --no-progress --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar /tmp

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /tmp/data
# MAGIC tar xf /tmp/VisA_20220922.tar --no-same-owner -C /tmp/data/ 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let us view some PCB images
# MAGIC
# MAGIC We can display images with `matplotlib` in a native python way.
# MAGIC
# MAGIC Let us investigate what a normal image looks like, and then one with an anomaly.

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=300):
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.imshow(img, interpolation="nearest", aspect="auto")


display_image("/tmp/data/pcb1/Data/Images/Normal/0000.JPG")
display_image("/tmp/data/pcb1/Data/Images/Anomaly/000.JPG")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let us now move the data to DBFS. 
# MAGIC
# MAGIC Quick reminder: the Databricks File System (DBFS) is a distributed file system mounted into a Databricks workspace and available on Databricks clusters. DBFS is an abstraction on top of scalable object storage that maps Unix-like filesystem calls to native cloud storage API calls. 

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf /dbfs/pcb1
# MAGIC mkdir -p /dbfs/pcb1/labels 
# MAGIC cp -r /tmp/data/pcb1/Data/Images/ /dbfs/pcb1/
# MAGIC cp /tmp/data/pcb1/image_anno.csv /dbfs/pcb1/labels/

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS circuit_board;
# MAGIC DROP TABLE IF EXISTS circuit_board_gold;
# MAGIC DROP TABLE IF EXISTS circuit_board_label;

# COMMAND ----------

cloud_storage_path="/pcb1"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load CSV label files with Auto Loader
# MAGIC CSV files can easily be loaded using Databricks [Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html)

# COMMAND ----------

from pyspark.sql.functions import substring_index, col

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", True)
    .option("cloudFiles.schemaLocation", f"{cloud_storage_path}/circuit_board_label_schema")
    .load(f"{cloud_storage_path}/labels/")
    .withColumn("filename", substring_index(col("image"), "/", -1))
    .select("filename", "label")
    .withColumnRenamed("label", "labelDetail")
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"{cloud_storage_path}/circuit_board_label_checkpoint")
    .toTable("circuit_board_label")
    .awaitTermination()
)
display(spark.table("circuit_board_label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load binary files with Auto Loader
# MAGIC
# MAGIC We can now use the Auto Loader to load images, and spark function to create the label column.
# MAGIC We can also very easily display the content of the images and the labels as a table.

# COMMAND ----------

from pyspark.sql.functions import substring_index, col, when

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.JPG")
    .option("recursiveFileLookup", "true")
    .option("cloudFiles.schemaLocation", f"{cloud_storage_path}/circuit_board_schema")
    .load(f"{cloud_storage_path}/Images/")
    .withColumn("filename", substring_index(col("path"), "/", -1))
    .withColumn(
        "labelName",
        when(col("path").contains("Anomaly"), "anomaly").otherwise("normal"),
    )
    .withColumn("label", when(col("labelName").eqNullSafe("anomaly"), 1).otherwise(0))
    .select("filename", "content", "label", "labelName")
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"{cloud_storage_path}/circuit_board_checkpoint")
    .toTable("circuit_board")
    .awaitTermination()
)
display(spark.table("circuit_board"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let us now merge the labels and the images tables

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE circuit_board_gold as (select cb.*, labelDetail from circuit_board cb inner join circuit_board_label cbl on cb.filename = cbl.filename);

# COMMAND ----------

# MAGIC %md
# MAGIC ## We can auto optimize our image tables
# MAGIC Auto optimize consists of two complementary features: optimized writes and auto compaction.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE circuit_board_gold SET TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true);
# MAGIC ALTER TABLE circuit_board SET TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can do any SQL command to this table.

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   circuit_board_gold
# MAGIC limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Our dataset is ready for our Data Science team
# MAGIC
# MAGIC That's it! We have now deployed a production-ready ingestion pipeline.
# MAGIC
# MAGIC Our images are incrementally ingested and joined with our label dataset.
# MAGIC
# MAGIC Let's see how this data can be used by a Data Scientist to [build the model]($./01_ImageClassificationPytorch) required for boat detection.
