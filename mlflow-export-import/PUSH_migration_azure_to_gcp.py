# Databricks notebook source
# Test GCS connection
gcs_data_file = 'gs://mlflow-model-dump/iowa_daily.csv'
df = spark.read.format("csv").load(gcs_data_file)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL: Build a sklearn model
# MAGIC If needed, build a sci-kit learn classification model below. 

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# set an experiment name; default experiment name will be the notebook
# TODO: replace with your own experiment name
experiment_name = '/Users/jon.cheung@databricks.com/mlflow-export-import'
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

# Log the model
mlflow.autolog()

with mlflow.start_run():
  X, y = datasets.load_iris(return_X_y=True, as_frame=True)
  clf = RandomForestClassifier(max_depth=7)
  clf.fit(X, y)

# COMMAND ----------

# This is a helper function to verify the latest version of your model and load the model if you wish
def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % MODEL_NAME)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])



# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG if not exists `jon_cheung`;
# MAGIC CREATE SCHEMA IF NOT EXISTS `mlflow-export-import`;

# COMMAND ----------

# Register UC model
mlflow.set_registry_uri("databricks-uc") # this is for unity catalog models
## TODO: Update this with your own CATALOG.SCHEMA.MODEL
catalog = "jon_cheung"
schema = "mlflow-export-import"
model = 'iris'

# Register te model 
MODEL_NAME = f'{catalog}.{schema}.{model}' 
autolog_run = mlflow.last_active_run()
model_uri = "runs:/{}/model".format(autolog_run.info.run_id)
mlflow.register_model(model_uri, MODEL_NAME)

latest_version = get_latest_model_version(model_name = MODEL_NAME)
model_version_uri = "models:/{model_name}/{latest_version}".format(model_name=MODEL_NAME,
                                                                   latest_version=latest_version)
model_version_uri


# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFLOW-EXPORT-IMPORT
# MAGIC 1. Export model from Unity Catalog --> DBFS
# MAGIC 2. Move model from DBFS --> GCS
# MAGIC

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip install git+https:///github.com/mlflow/mlflow-export-import/#egg=mlflow-export-import
# MAGIC pip install setuptools
# MAGIC pip install mlflow

# COMMAND ----------

from mlflow_export_import.model.export_model import export_model

# dump model into dbfs
model_dir = 'FileStore/jon.cheung@databricks.com/mlflow-export-models'
export_model(
    model_name = 'jon_cheung.mlflow-export-import.iris', 
    output_dir = f"/dbfs/{model_dir}/iris")


# COMMAND ----------

# check to make sure your model is in dbfs
dbutils.fs.ls(f"{model_dir}")

# COMMAND ----------

# upload dbfs exported models to gcs
bucket_name = "mlflow-model-dump"
dbutils.fs.mkdirs(f"gs://{bucket_name}/models")
dbutils.fs.cp({model_dir}, f"gs://{bucket_name}/models", recurse=True)
