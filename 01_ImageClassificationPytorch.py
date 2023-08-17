# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/cv-quality-inspection. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/product-quality-inspection.

# COMMAND ----------

# MAGIC %md
# MAGIC # Implementing and deploying our pytorch model
# MAGIC
# MAGIC Our next step as a Data Scientist is to implement an ML model to run image classification.
# MAGIC
# MAGIC We will re-use the gold table built in our previous data pipeline as a training dataset.
# MAGIC
# MAGIC Building such a model is greatly simplified by the use of [torchvision](https://pytorch.org/vision/stable/index.html).
# MAGIC
# MAGIC ## MLOps steps
# MAGIC
# MAGIC While building an image classification model can be easily done, deploying models in production is much harder.
# MAGIC
# MAGIC Databricks simplifies this process and accelerate the time-to-value journey with the help of MLFlow by providing
# MAGIC
# MAGIC * Auto experimentation tracking to keep track of progress
# MAGIC * Simple, distributed hyperparameter tuning with hyperopt to get the best model
# MAGIC * Model packaging in MLFlow, abstracting our ML framework
# MAGIC * Model registry for governance
# MAGIC * Batch or real time serving (1 click deployment)

# COMMAND ----------

model_name = "cv_pcb_classification"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable the GPU
# MAGIC With Deep Learning it makes sense to use GPUs for training.

# COMMAND ----------

import torch


# Check GPU availability
if not torch.cuda.is_available():  # is gpu
    raise Exception(
        "Please use a GPU-cluster for model training, CPU instances will be too slow"
    )

# COMMAND ----------

# del
import sys

print(
    "You are running a Databricks {0} cluster leveraging Python {1}".format(
        spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"),
        sys.version.split(" ")[0],
    )
)

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

from PIL import Image

import torchvision
import torch

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import horovod.torch as hvd
from sparkdl import HorovodRunner

import mlflow

import pyspark.sql.functions as f

import numpy as np
from functools import partial
import io
import uuid

username = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
mlflow.set_experiment("/Users/{}/pcbqi".format(username))

petastorm_path = (
    f"file:///dbfs/tmp/petastorm/{str(uuid.uuid4())}/cache"  # location where to store petastorm cache files
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Split data as train/test dataset
# MAGIC
# MAGIC Like for any ML model, we start by splitting the images in a training/test dataset

# COMMAND ----------

# retrieve images of interest
images = spark.table("circuit_board_gold").select(
    "content", "label", "filename"
)  # path will be used as a unique identifier in next steps

# retrieve stratified sample of images
images_train = images.sampleBy(
    "label", fractions={0: 0.8, 1: 0.8}
)  # 80% sample from each class to training
images_test = images.join(
    images_train, on="filename", how="leftanti"
)  # remainder to testing

# drop any unnecessary fields
images_train = images_train.drop("filename").repartition(
    sc.defaultParallelism
)  # drop path identifier
images_test = images_test.drop("filename").repartition(sc.defaultParallelism)

# verify sampling
display(
    images_train.withColumn("eval_set", f.lit("train"))
    .union(images_test.withColumn("eval_set", f.lit("test")))
    .groupBy("eval_set", "label")
    .agg(f.count("*").alias("instances"))
    .orderBy("eval_set", "label")
)

# COMMAND ----------

images_test.display()

# COMMAND ----------

images_train.display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Delta table for DL with petastorm
# MAGIC Our data is currently stored as a Delta table and available as a Spark dataframe. However, pytorch is expecting a specific type of data.
# MAGIC
# MAGIC To solve that, we will use Petastorm and the Spark converter to automatically send data to our model from the table. The converter will incrementally load the data using local cache for faster processing. Please see the associated [documentation](https://docs.databricks.com/applications/machine-learning/load-data/petastorm.html) for more details.

# COMMAND ----------

try:
    dbutils.fs.rm(petastorm_path, True)
except:
    pass

# COMMAND ----------

# configure destination for petastore cache
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, petastorm_path)

# determine rough bytes in dataset
bytes_in_train = (
    images_train.withColumn("bytes", f.lit(4) + f.length("content"))
    .groupBy()
    .agg(f.sum("bytes").alias("bytes"))
    .collect()[0]["bytes"]
)
bytes_in_test = (
    images_test.withColumn("bytes", f.lit(4) + f.length("content"))
    .groupBy()
    .agg(f.sum("bytes").alias("bytes"))
    .collect()[0]["bytes"]
)

# cache images data
converter_train = make_spark_converter(
    images_train,
    parquet_row_group_size_bytes=int(bytes_in_train / sc.defaultParallelism),
)
converter_test = make_spark_converter(
    images_test, parquet_row_group_size_bytes=int(bytes_in_test / sc.defaultParallelism)
)

# COMMAND ----------

NUM_CLASSES = 2  # two classes in labels (0 or 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Torchvision
# MAGIC Torchvision provides us with pre-trained models that we can reuse

# COMMAND ----------

from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)


def get_model():
    # access pretrained model
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # add a new classifier layer for transfer learning
    num_ftrs = model.heads.head.in_features

    # parameters of newly constructed modules have requires_grad=True by default
    model.heads.head = torch.nn.Linear(num_ftrs, NUM_CLASSES)

    return model, weights

# COMMAND ----------

model, weights = get_model()
transforms = weights.transforms()
print(model.heads)
print(transforms)

# COMMAND ----------

# define logic for the transformation of images
def transform_row(is_train, batch_pd):

    # apply pipeline to images
    batch_pd["features"] = batch_pd["content"].map(
        lambda x: np.ascontiguousarray(
            transforms(Image.open(io.BytesIO(x)).convert("RGB")).numpy()
        )
    )

    # transform labels (our evaluation metric expects values to be float32)
    # -----------------------------------------------------------
    batch_pd["label"] = batch_pd["label"].astype("float32")
    # -----------------------------------------------------------

    return batch_pd[["features", "label"]]


# define function to retrieve transformation spec
def get_transform_spec(is_train=True):

    spec = TransformSpec(
        partial(transform_row, is_train),  # function to call to retrieve/transform row
        edit_fields=[  # schema of rows returned by function
            ("features", np.float32, (3, 224, 224), False),
            ("label", np.float32, (), False),
        ],
        selected_fields=["features", "label"],  # fields in schema to send to model
    )

    return spec

# COMMAND ----------

# access petastorm cache and transform data using spec
with converter_train.make_torch_dataloader(
    transform_spec=get_transform_spec(is_train=True), batch_size=1
) as train_dataloader:

    # retrieve records from cache
    for i in iter(train_dataloader):
        print(i)
        break

# COMMAND ----------

BATCH_SIZE = 32  # process 32 images at a time
NUM_EPOCHS = 15  # iterate over all images 5 times

# COMMAND ----------

from sklearn.metrics import f1_score


def train_one_epoch(
    model,
    criterion,
    optimizer,
    scheduler,
    train_dataloader_iter,
    steps_per_epoch,
    epoch,
    device,
):

    model.train()  # set model to training mode

    # statistics
    running_loss = 0.0
    running_corrects = 0
    running_size = 0

    # iterate over the data for one epoch.
    for step in range(steps_per_epoch):

        # retrieve next batch from petastorm
        pd_batch = next(train_dataloader_iter)

        # seperate input features and labels
        inputs, labels = pd_batch["features"].to(device), pd_batch["label"].to(device)

        # track history in training
        with torch.set_grad_enabled(True):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=0)[:, 1]
            loss = criterion(probs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        running_size += inputs.size(0)

    scheduler.step()

    epoch_loss = running_loss / steps_per_epoch
    epoch_acc = running_corrects.double() / running_size

    print("Train Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def evaluate(
    model, criterion, test_dataloader_iter, test_steps, device, metric_agg_fn=None
):

    model.eval()  # set model to evaluate mode

    # statistics
    running_loss = 0.0
    running_corrects = 0
    running_size = 0
    f1_scores = 0

    # iterate over all the validation data.
    for step in range(test_steps):

        pd_batch = next(test_dataloader_iter)
        inputs, labels = pd_batch["features"].to(device), pd_batch["label"].to(device)

        # do not track history in evaluation to save memory
        with torch.set_grad_enabled(False):

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            loss = criterion(probs, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        running_size += inputs.size(0)
        f1_scores += f1_score(labels.cpu().data, preds.cpu())

    # average the losses across observations for each minibatch.
    epoch_loss = running_loss / test_steps
    epoc_f1 = f1_scores / test_steps
    epoch_acc = running_corrects.double() / running_size

    # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
    if metric_agg_fn is not None:
        epoch_loss = metric_agg_fn(epoch_loss, "avg_loss")
        epoch_acc = metric_agg_fn(epoch_acc, "avg_acc")
        epoc_f1 = metric_agg_fn(epoc_f1, "avg_f1")

    print(
        "Testing Loss: {:.4f} Acc: {:.4f} F1: {:.4f}".format(
            epoch_loss, epoch_acc, epoc_f1
        )
    )
    return epoch_loss, epoch_acc, epoc_f1

# COMMAND ----------

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def train_and_evaluate(lr=0.001):

    # determine if gpu available for compute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model
    model, _ = get_model()

    # assign model to process on identified processor device
    model = model.to(device)

    # optimize for binary cross entropy
    criterion = torch.nn.BCELoss()

    # only parameters of final layer are being optimized.
    filtered_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(filtered_params, lr=lr)

    # decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    # access data in petastorm cache
    with converter_train.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=True), batch_size=BATCH_SIZE
    ) as train_dataloader, converter_test.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=False), batch_size=BATCH_SIZE
    ) as val_dataloader:

        # define iterator for data access and number of cycles required
        train_dataloader_iter = iter(train_dataloader)
        steps_per_epoch = len(converter_train) // BATCH_SIZE

        val_dataloader_iter = iter(val_dataloader)
        validation_steps = max(1, len(converter_test) // BATCH_SIZE)

        # for each epoch
        for epoch in range(NUM_EPOCHS):

            print("Epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
            print("-" * 10)

            # train
            train_loss, train_acc = train_one_epoch(
                model,
                criterion,
                optimizer,
                exp_lr_scheduler,
                train_dataloader_iter,
                steps_per_epoch,
                epoch,
                device,
            )
            # evaluate
            val_loss, val_acc, val_f1 = evaluate(
                model, criterion, val_dataloader_iter, validation_steps, device
            )

    # correct a type issue with acc
    if type(val_acc) == torch.Tensor:
        val_acc = val_acc.item()

    return model, val_loss, val_acc, val_f1  # extract value from tensor


# model, loss, acc, f1 = train_and_evaluate(**{'lr':0.00001})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameters tuning with Hyperopt
# MAGIC
# MAGIC Our model is now ready. Tuning such a model can be complex. We have the choices between architectures, encoders, and hyperparameters like the learning rate.
# MAGIC
# MAGIC Let us use Hyperopt to find the best set of hyperparameters for us. Note that Hyperopt can also work in a distributed manner, training multiple models in parallel on multiple instances to speed-up the training process.

# COMMAND ----------

# define hyperparameter search space
search_space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1.2e-5)),
}


# define training function to return results as expected by hyperopt
def train_fn(params):

    # train model against a provided set of hyperparameter values
    model, loss, acc, f1 = train_and_evaluate(**params)

    # log this iteration to mlflow for greater transparency
    mlflow.log_metric("accuracy", acc)

    mlflow.log_metric("f1", f1)

    mlflow.pytorch.log_model(model, "model")
    # return results from this iteration
    return {"loss": loss, "status": STATUS_OK}


# determine degree of parallelism to employ
if torch.cuda.is_available():  # is gpu
    nbrWorkers = sc.getConf().get("spark.databricks.clusterUsageTags.clusterWorkers")
    if nbrWorkers is None:  # gcp
        nbrWorkers = sc.getConf().get(
            "spark.databricks.clusterUsageTags.clusterTargetWorkers"
        )
    parallelism = int(nbrWorkers)
    if parallelism == 0:  # single node cluster
        parallelism = 1
else:  # is cpu
    parallelism = sc.defaultParallelism

# perform distributed hyperparameter tuning
with mlflow.start_run(run_name=model_name) as run:

    argmin = fmin(
        fn=train_fn,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,  # total number of hyperparameter runs (this would typically be much higher)
        trials=SparkTrials(parallelism=parallelism),
    )  # number of hyperparameter runs to run in parallel

# COMMAND ----------

argmin = {"lr": 1.1747777342914114e-5}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed deep learning with Horovod
# MAGIC We can now train our model with more epochs. To accelerate the run we can distribute the training accros multiple nodes on our Spark cluster.
# MAGIC
# MAGIC See the documentation of [Horovod](https://docs.databricks.com/machine-learning/train-model/distributed-training/horovod-runner.html) for more details.

# COMMAND ----------

# define function for model evaluation
def metric_average_hvd(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


# define function for distributed training & evaluation
def train_and_evaluate_hvd(lr=0.001):

    # Step 1: Initialize Horovod
    hvd.init()

    # Step 2: Align the Horovod processes to specific CPU cores or GPUs

    # determine devices to use for training
    if torch.cuda.is_available():  # gpu
        torch.cuda.set_device(hvd.local_rank())
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")  # cpu

    # retrieve model and associate with device
    model, _ = get_model()
    model = model.to(device)
    criterion = torch.nn.BCELoss()

    # Step 3: Scale the learning rate based on the number of Horovod processes
    filtered_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(filtered_params, lr=lr * hvd.size())

    # Step 4: Wrap the optimizer for distribution
    optimizer_hvd = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_hvd, step_size=7, gamma=0.1
    )

    # Step 5: Initialize state variables associated with the Horovod processes
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # open up access to the petastorm cache
    with converter_train.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=True),
        cur_shard=hvd.rank(),
        shard_count=hvd.size(),
        batch_size=BATCH_SIZE,
    ) as train_dataloader, converter_test.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=False),
        cur_shard=hvd.rank(),
        shard_count=hvd.size(),
        batch_size=BATCH_SIZE,
    ) as test_dataloader:

        # each core/gpu will handle a batch
        train_dataloader_iter = iter(train_dataloader)
        train_steps = len(converter_train) // (BATCH_SIZE * hvd.size())
        test_dataloader_iter = iter(test_dataloader)
        test_steps = max(1, len(converter_test) // (BATCH_SIZE * hvd.size()))

        # iterate over dataset
        for epoch in range(NUM_EPOCHS):

            # print epoch info
            print("Epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
            print("-" * 10)

            # train model
            train_loss, train_acc = train_one_epoch(
                model,
                criterion,
                optimizer_hvd,
                exp_lr_scheduler,
                train_dataloader_iter,
                train_steps,
                epoch,
                device,
            )

            # evaluate model
            test_loss, test_acc, f1_acc = evaluate(
                model,
                criterion,
                test_dataloader_iter,
                test_steps,
                device,
                metric_agg_fn=metric_average_hvd,
            )

    return test_loss, test_acc, f1_acc, model

# COMMAND ----------

# determine parallelism available to horovod
if torch.cuda.is_available():  # is gpu
    nbrWorkers = sc.getConf().get("spark.databricks.clusterUsageTags.clusterWorkers")
    if nbrWorkers is None:  # gcp
        nbrWorkers = sc.getConf().get(
            "spark.databricks.clusterUsageTags.clusterTargetWorkers"
        )
    parallelism = int(nbrWorkers)
    if parallelism == 0:  # single node cluster
        parallelism = 1
else:
    parallelism = 2  # setting the parallelism low at 2 for the small data size; otherwise feel free to set to sc.defaultParallelism

# initialize runtime environment for horovod
hr = HorovodRunner(np=parallelism)

# run distributed training
with mlflow.start_run(run_name=model_name) as run:

    # train and evaluate model
    loss, acc, f1, model = hr.run(
        train_and_evaluate_hvd, **argmin
    )  # argmin contains tuned hyperparameters

    # log model in mlflow
    mlflow.log_params(argmin)
    mlflow.log_metrics({"loss": loss, "accuracy": acc, "f1": f1})
    mlflow.pytorch.log_model(model, "model")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Deploying our model in production
# MAGIC
# MAGIC Our model is now trained. All we have to do is get the best model (based on the `f1` metric) and deploy it in MLFlow registry.
# MAGIC
# MAGIC We can do that using the UI or with a couple of API calls:

# COMMAND ----------

# get the best model from the registry
best_model = mlflow.search_runs(
    filter_string=f'attributes.status = "FINISHED"',
    order_by=["metrics.f1 DESC"],
    max_results=1,
).iloc[0]
model_registered = mlflow.register_model(
    "runs:/" + best_model.run_id + "/model", model_name
)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
print("registering model version " + model_registered.version + " as production model")
client.transition_model_version_stage(
    name=model_name,
    version=model_registered.version,
    stage="Production",
    archive_existing_versions=True,
)

# COMMAND ----------

try:
    dbutils.fs.rm(petastorm_path, True)
except:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Our model is now deployed and flagged as production-ready!
# MAGIC
# MAGIC We have now deployed our model to our model registry. This will provide model governance, and will simplify and accelerate all downstream pipeline developements.
# MAGIC
# MAGIC The model is now ready to be used in any data pipeline (DLT, batch or real time with Databricks Model Serving). 
# MAGIC
# MAGIC Let us now see how we can use it to [run inferences]($./02_PredictionPCB) at scale.
