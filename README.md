![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

# Product Quality Inspection of Printed Circuit Board (PCB) using Computer Vision and Real-time Serverless Inference


In this solution accelerator, we will show you how Databricks can help you to deploy an end-to-end pipeline for product quality inspection. The model is deployed using Databricks [Serverless Real-time Inference](https://docs.databricks.com/archive/serverless-inference-preview/serverless-real-time-inference.html).

We will use the [Visual Anomaly (VisA)](https://registry.opendata.aws/visa/) detection dataset, and build a pipeline to detect anomalies in our PCB images. 

## Why image quality inspection?

Image quality inspection is a common challenge in the context of manufacturing. It is key to delivering Smart Manufacturing.

## Implementing a production-grade pipeline

The image classification problem has been eased in recent years with pre-trained deep learning models, transfer learning, and higher-level frameworks. While a data science team can quickly deploy such a model, a real challenge remains in the implementation of a production-grade, end-to-end pipeline, consuming images and requiring MLOps/governance, and ultimately delivering results.

Databricks Lakehouse is designed to make this overall process simple, letting Data Scientist focus on the core use-case.

In order to build the quality inspection model, we use Torchvision. However, the same architecture may be used with other libraries. The Torchvision library is part of the PyTorch project, a popular framework for deep learning. Torchvision comes with model architectures, popular datasets, and image transformations. 


The first step in building the pipeline is data ingestion. Databricks enables the loading of any source of data, even images (unstructured data). This is stored in a table with the content of the image and also the associated label in a efficient and a distributed way.

This is the pipeline we will be building. We ingest 2 datasets, namely:

* The raw satellite images (jpg) containing PCB
* The label, the type of anomalies saved as CSV files

We will first focus on building a data pipeline to incrementally load this data and create a final Gold table.

This table will then be used to train a ML Classification model to learn to detect anomalies in our images in real time!


___


<florent.brosse@databricks.com>


___


<img width="1000px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/pipeline.png">

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| aws-cli                                 | Universal Command Line Interface for Amazon Web Services      | Apache 2.0        | https://github.com/aws/aws-cli/                      |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
