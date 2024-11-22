# Databricks notebook source
# MAGIC %run ./generate_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deep learning scaling
# MAGIC * 4 deep learning models
# MAGIC * 4 Ray worker nodes to match number of models
# MAGIC * maxing out CPUs per worker node

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster


# The recommended configuration for each Ray worker node is as follows: Minimum 4 CPU cores per Ray worker node. Minimum 10GB heap memory for each Ray worker node.
# currently using a 96 workern node cluster
setup_ray_cluster(
  min_worker_nodes=6,
  max_worker_nodes=6,
  num_cpus_per_node=4,
  num_gpus_worker_node=0,
  collect_log_to_path="/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs"
)




# COMMAND ----------

import ray
import random
import time
from fractions import Fraction

ray.init(ignore_reinit_error=True)

@ray.remote
def pi4_sample(sample_count):
    """pi4_sample runs sample_count experiments, and returns the
    fraction of time it was inside the circle.
    """
    in_count = 0
    for i in range(sample_count):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            in_count += 1
    return Fraction(in_count, sample_count)

SAMPLE_COUNT = 1000 * 100000
start = time.time()
future = pi4_sample.remote(sample_count=SAMPLE_COUNT)
pi4 = ray.get(future)
end = time.time()
dur = end - start
print(f'Running {SAMPLE_COUNT} tests took {dur} seconds')

pi = pi4 * 4
print(float(pi))


# COMMAND ----------

from ray.util.spark import shutdown_ray_cluster
import ray

shutdown_ray_cluster()
ray.shutdown()


# COMMAND ----------

import ray
import random
import time
from fractions import Fraction

global_models_list = [NBEATS(h=horizon, max_steps=50),
                      NHITS(h=horizon, max_steps=50)]

ray.init()

@ray.remote
def train_deep_learning_models():
    for model in range(global_models_list):
      with mlflow.run()
        
        nf = NeuralForecast(models=model, freq='M')
        nf.fit(df=Y_train_df)
        Y_hat_df = nf.predict(Y_test_df).reset_index()


# COMMAND ----------


