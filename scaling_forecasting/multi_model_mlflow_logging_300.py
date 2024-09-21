# Databricks notebook source
# MAGIC %md ### This notebook implements an example, custom pyfunc model that applies multiple regression models
# MAGIC Built using Databricks ML Runtime 15.4

# COMMAND ----------

import joblib
import urllib.request
import json
import os
import requests

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col
import pyspark.sql.functions as func
import mlflow
from databricks.sdk import WorkspaceClient

from helpers import get_current_user, get_or_create_experiment

# COMMAND ----------

spark.conf.set('spark.sql.adaptive.enabled', 'false')
mlflow.autolog(disable=True)

# COMMAND ----------

current_user = get_current_user().lower().replace('-','_')
experiment_location = f'/Shared/{current_user}'
mlflow.set_experiment(experiment_location)

# COMMAND ----------

# MAGIC %md ### Create groups of regression features
# MAGIC
# MAGIC A regression model will be fit for each group; vary the number of groups to align with your actualy data volumes.

# COMMAND ----------

# Configure the data volumes
groups = 300
n_features_per_group = 10
n_samples_per_group = 100

# DBFS directory to store model artifacts
regression_model_dbfs_directory = 'dbfs:/Users/jon.cheung@databricks.com/customers/fastenal/regression_models'
regression_model_dbfs_directory_python = f"/{regression_model_dbfs_directory.replace(':', '')}"
# If Unity Catalog is enabled
output_delta_location = 'main.jon_cheung_models.mlc_multiple_regression_models'
# If Unity Catalog is not enabled
# output_delta_location = 'default.mlc_multiple_regression_models'

# Create the directory of delete and recreate if exists
try: 
  dbutils.fs.rm(regression_model_dbfs_directory, recurse=True)
  dbutils.fs.mkdirs(regression_model_dbfs_directory)
except:
  dbutils.fs.mkdirs(regression_model_dbfs_directory)

dbutils.fs.ls(regression_model_dbfs_directory)

# COMMAND ----------

def create_groups(groups=groups):

  groups = [[f'group_{str(n+1).zfill(2)}'] for n in range(groups)]

  schema = StructType()
  schema.add('group_name', StringType())

  return spark.createDataFrame(groups, schema=schema)

# COMMAND ----------

groups = create_groups()
display(groups)

# COMMAND ----------

# MAGIC %md ### Create group-level features
# MAGIC
# MAGIC This is accomplished using a PandasUDF to distribute the computation at the group level

# COMMAND ----------

def get_feature_col_names(n_features_per_group=n_features_per_group):
  return [f"features_{n}" for n in range(n_features_per_group)]


def configure_features_udf(n_features_per_group=n_features_per_group, n_samples_per_group=n_samples_per_group):

  def create_group_features(group_data: pd.DataFrame) -> pd.DataFrame:

    features, target = make_regression(n_samples=n_samples_per_group, n_features=n_features_per_group)
    feature_names = get_feature_col_names()
    df = pd.DataFrame(features, columns=feature_names)

    df['target'] = target.tolist()

    group_name = group_data["group_name"].loc[0]
    df['group_name'] = group_name

    col_order = ['group_name'] + feature_names + ['target']

    return df[col_order]
  
  return create_group_features


spark_schema = StructType()
spark_schema.add('group_name', StringType())
for feature_name in get_feature_col_names():
  spark_schema.add(feature_name, FloatType())
spark_schema.add('target', FloatType())

# COMMAND ----------

udf = configure_features_udf()
features = groups.groupBy('group_name').applyInPandas(udf, spark_schema)
display(features)

# COMMAND ----------

# MAGIC %md #### Use a PandasUDF to fit linear regressions to different groups of data
# MAGIC Fitted models are saved to the DBFS directory

# COMMAND ----------

def config_models_udf(dbfs_file_path=regression_model_dbfs_directory_python):

  def fit_group_models(group_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a regression model for the group and save it in pickle format to DBFS.
    """

    group_name = group_data["group_name"].loc[0] 
    group_features = group_data.drop(["group_name", "target"], axis=1)
    group_target = group_data['target']
    group_model = LinearRegression().fit(group_features.to_numpy(), np.array(group_target))
    saved_model_name = f"{regression_model_dbfs_directory_python}/{group_name}.pkl"
    joblib.dump(group_model, saved_model_name)

    df = pd.DataFrame([group_name], columns=['group_name'])
    df['model_storage_location'] = saved_model_name

    return df

  return fit_group_models

spark_schema = StructType()
spark_schema.add('group_name', StringType()) 
spark_schema.add('model_storage_location', StringType())

# COMMAND ----------

# MAGIC %md #### Fit the group-level models

# COMMAND ----------

num_cpu_cores = 64
spark.conf.set("spark.sql.shuffle.partitions", num_cpu_cores)

fit_models_udf = config_models_udf()

fitted_models = features.groupBy('group_name').applyInPandas(fit_models_udf, schema=spark_schema)

fitted_models.write.mode('overwrite').format('delta').saveAsTable(output_delta_location)
display(spark.table(output_delta_location))

# COMMAND ----------

# MAGIC %md #### View the pickled regression models in DBFS

# COMMAND ----------

dbutils.fs.ls(regression_model_dbfs_directory)

# COMMAND ----------

# MAGIC %md ### Log a custom pyfunc model
# MAGIC This model will load models stored as artifacts and apply the appropriate models to a group. It is basically a wrapper around the pickled model artifacts.

# COMMAND ----------

# Assemble a dictionary of {'artifact name': 'artifact dbfs path'}

artifacts = {}
model_artifacts = dbutils.fs.ls(regression_model_dbfs_directory)
for artifact in model_artifacts:
  path = artifact.path
  path = f"/{path.replace(':', '')}"
  group_name = path.split('/')[-1].split('.')[0]
  artifacts[group_name] = path

artifacts

# COMMAND ----------

class ModelLoader(mlflow.pyfunc.PythonModel):
  """
  A custom MLflow model that applies sku-level forecasting
  models to each sku contained within an inference dataset.
  """

  def load_context(self, context):
    """
    Load all the models into a Python dictionary
    """
    import joblib

    self.group_models = {}
    for group_name, artifact_path in context.artifacts.items():
      self.group_models[group_name] = joblib.load(artifact_path)
  
  def predict(self, context, group_data):
    """
    Apply the appropirate model to each row.
    """
    predictions = []
    for index, row in group_data.iterrows():
      group_name = (row['group_name'])
      group_features = row.drop(["group_name", "target"]).to_numpy()

      model = self.group_models.get(group_name)

      prediction = model.predict([group_features])
      predictions.append(prediction)
    return pd.DataFrame(predictions, columns=['predictions'])
  

with mlflow.start_run(run_name='group_regressions') as run:
  run_id = run.info.run_id

  input_example = features.toPandas()
  input_example = input_example.sample(5)
  input_example.reset_index(inplace=True, drop=True)

  mlflow.pyfunc.log_model(artifact_path="model", 
                          python_model=ModelLoader(), 
                          artifacts=artifacts,
                          input_example=input_example,
                          registered_model_name='mlc_multiple_regression')

# COMMAND ----------

# Test the model using batch inference
logged_model = f'runs:/{run_id}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

features_pd = features.toPandas()
predictions = loaded_model.predict(features_pd)

# COMMAND ----------

display(predictions)
