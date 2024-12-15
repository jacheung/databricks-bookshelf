# Databricks notebook source
# MAGIC %md 
# MAGIC ## Parallelized Prophet Modeling with Ray
# MAGIC
# MAGIC Prophet is a simple, yet powerful, additive forecasting model. To the former, it's implementation is intuitive and requires editing a few parameters and, to the latter, it provides an algorithmically efficient way to identify time-related patterns in the data. These two aspects make Prophet an ideal starting, and possibly end-state, point for a forecasting model. 
# MAGIC
# MAGIC However, in real-world production use-cases we must overcome scaling challenges in model training and inference. These scaling challenges can happen at the model level or the data level. At the model level, model's become too large to fit on a single machine and therefore need to broken up intelligently to be trained on multiple machines. On the data level, the volume of data is enormous and this requires distributed computing to get a model into production in a timely manner. With Prophet, we run into bottlenecks on the data side. 
# MAGIC
# MAGIC This notebook will be about parallelizing training and inference of Prophet when we have a large volume of data. In this scenario we'll be forecasting the demand for 10 unique stores and 50 unique items. This totals to 500 forecasting models. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup:
# MAGIC * DBR 15.4 LTS
# MAGIC * Multi-node cluster
# MAGIC   * driver: 32GB 8-cores 
# MAGIC   * workers x8: 64GB 16-cores each
# MAGIC * Enable autoscaling; min 2 workers and max 8 workers
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet skforecast
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from skforecast.datasets import fetch_dataset
data = fetch_dataset(name="store_sales")

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

to_plot = 3
storeId = np.random.choice(data['store'].unique(), to_plot)
itemId = np.random.choice(data['item'].unique(), to_plot)

plt.figure(figsize=(15, 5))
for x in range(to_plot):  
  selected_data = data.loc[(data['store'] == storeId[x]) & (data['item'] == itemId[x])]
  plt.plot(selected_data['sales'], label=f"store {storeId[x]} - item {itemId[x]}")
plt.ylabel('Sales Amount')
plt.legend()
plt.show()

# COMMAND ----------

# data prep for prophet forecast
data_prophet = data.copy()
data_prophet.reset_index(drop=False, inplace=True)
data_prophet.rename(columns={'date':'ds', 'sales':'y'}, inplace=True)

import itertools
# since we are forecasting for each item per each store, let's see the total number of combinations
combinations = list(itertools.product(data_prophet['store'].unique(), 
                                      data_prophet['item'].unique()))

print(f'The total number of combinations are {len(combinations)}')


# COMMAND ----------

import pandas as pd
from prophet import Prophet
import time
t = time.time()

# this is our train function for Prophet
def train_and_inference_prophet(train_data:pd.DataFrame, 
                                store_id:int, 
                                item_id:int, 
                                horizon: int):
  selected_data = train_data.loc[(train_data['store'] == store_id) & (train_data['item'] == item_id)]
  m = Prophet(daily_seasonality=True)
  m.fit(train_data)
  future = m.make_future_dataframe(periods=horizon)
  forecast = m.predict(future)
  return forecast

# Serial forecasting of all 500 models. Each model takes about 5 minutes so this could take 40 hours... We'll run 1 combination just to see. 
combinations_reduced = combinations[:1]
forecasts = []
forecasts = [train_and_inference_prophet(train_data=data_prophet, 
                                                store_id=combo[0], 
                                                item_id=combo[1], 
                                                horizon=14
                                                ) 
                     for combo in combinations_reduced]

elapsed = time.time() - t
print(elapsed)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Parallelize Prophet Forecasting
# MAGIC We will parallelize our forecasting by leveraging Ray Core. 
# MAGIC
# MAGIC Simply, we will create worker nodes that mirror the configuration of our multi-node cluster. Each worker node in our c cluster has 16 CPUs and we will allocate 2 CPUs per actor. Each actor is runs one iteration of our Ray Core task. 
# MAGIC
# MAGIC Below is an image of how we're parallelizing. In short, we're training 64 models in parallel.
# MAGIC ![](images/prophet_parallelization.jpg)
# MAGIC
# MAGIC

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

# The recommended configuration for a Ray cluster is as follows:
# - set the num_cpus_per_node to the CPU count per worker node ( ith this configuration, each Apache Spark worker node launches one Ray worker node that will fully utilize the resources of each Apache Spark worker node.)
# - set min_worker_nodes to the number of Ray worker nodes you want to launch on each node.
# - set max_worker_nodes to the total amount of worker nodes (this and `min_worker_nodes` together enable autoscaling)
setup_ray_cluster(
  min_worker_nodes=2,
  max_worker_nodes=8,
  num_cpus_per_node=16,
  num_gpus_worker_node=0,
  collect_log_to_path="/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs"
)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

import pandas as pd
from prophet import Prophet
import pickle
import mlflow
import os
from mlflow.utils.databricks_utils import get_databricks_env_vars

experiment_name = '/Users/jon.cheung@databricks.com/ray_prophet'
mlflow.set_experiment(experiment_name)
mlflow_db_creds = get_databricks_env_vars("databricks")

# Here we transform our training code to one that works with Ray. We simply add a @ray.remote decorator to the function along with some mlflow logging parameters for a nested child runs
@ray.remote
def train_and_inference_prophet(train_data:pd.DataFrame, 
                                store_id:int, 
                                item_id:int, 
                                horizon:int,
                                parent_run_id:str
                                ):
        selected_data = train_data.loc[(train_data['store'] == store_id) & (train_data['item'] == item_id)]
        # Set mlflow credentials and active MLflow experiment within each Ray task
        os.environ.update(mlflow_db_creds)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name = f"store_{store_id}_item_{item_id}",
                              parent_run_id=parent_run_id):
                m = Prophet(daily_seasonality=True)
                m.fit(train_data)
                future = m.make_future_dataframe(periods=horizon)
                forecast = m.predict(future)
                mlflow.prophet.log_model(pr_model=m,
                                         artifact_path="prophet_model")
        return forecast

# Here, the call to the train_and_inference_prophet function creates an object reference. By default, Ray will use a single-CPU per task. Since, Prophet is a bit more compute intensive, we'll increase the number of CPUs to 2.
# Using 8 workers (each with 64GB memory and 16 cores; i.e. m5.2xlarge on Azure), we can parallelize our training and inference to 64 tasks in parallel. 
# Instead of 41 hours for 500 models, our parallelized method takes a little over one hour. 

with mlflow.start_run(run_name="prophet_models_241212") as parent_run: 
        # Start parent run on the main driver process
        forecasts_obj_ref = [train_and_inference_prophet
                        .options(num_cpus=2)
                        .remote(train_data=data_prophet, 
                                store_id=combo[0],
                                item_id=combo[1], 
                                horizon=14,
                                parent_run_id=parent_run.info.run_id
                                ) 
                        for combo in combinations]

        # We need to call ray.get() on the referenced object to create a blocking call. 
        # Blocking call is one which will not return until the action it performs is complete.
        forecasts = ray.get(forecasts_obj_ref)

# COMMAND ----------

shutdown_ray_cluster()
