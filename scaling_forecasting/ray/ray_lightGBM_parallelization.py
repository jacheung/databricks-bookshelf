# Databricks notebook source
# MAGIC %md
# MAGIC # Parallelized Bayesian Hyperparameter Tuning for LightGBM with Ray 
# MAGIC
# MAGIC Boosting algorithms, like LightGBM, offers a simple, yet powerful, model to solve many regression and classification problems. However, they are prone to overfitting and require hyperparameter tuning with validation datasets to ensure they can be generalized to the real-world problems they are meant to solve. When it comes to hyperparameter tuning, traditional grid-search is inefficient (i.e. unnecessarily time-consuming). It offers little benefit over more efficient methods like Bayesian search, especially when the search space is large. To double-click on this, Bayesian search balances exploration and exploitation. It explores the search space and uses this as a prior to determine which area to search more in-depth for later trials. 
# MAGIC
# MAGIC This notebook outlines two powerful additions to LightGBM to improve (i.e. make more efficient) hyperparameter search. They are:
# MAGIC 1. Ray for parallelized search
# MAGIC 2. Optuna for Bayesian search

# COMMAND ----------

!pip install --quiet lightgbm scikit-learn bayesian-optimization==1.5.1 ray[default] ray[tune] optuna mlflow
dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Baseline: one LightGBM model
# MAGIC This short code-snippet below builds one LightGBM model using a specific set of hyperparameters. It'll provide a starting point for us before we parallelize. Ensure you understand what's going on here before we move onto the next steps. 

# COMMAND ----------

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

# Create a pseudo-dataset to test
data, labels = make_regression(n_samples=10000000, 
                                   n_features=100, 
                                   n_informative=10, 
                                   n_targets=1)
# Perform train test split
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

# Build input matrices for LightGBM
train_set = lgb.Dataset(train_x, label=train_y)
test_set = lgb.Dataset(test_x, label=test_y)

# LightGBM hyperparameter configs
config = {'objective':'regression',
          'metric': 'rmse',
          'num_leaves':31,
          'learning_rate':0.05,
          'n_estimators':1000,
          'num_threads': 16, 
          'random_state':42}
# Train the classifier
results = {}
gbm = lgb.train(config,
                train_set,
                valid_sets=[train_set, test_set],
                valid_names=["train", "validation"],
                callbacks = [lgb.record_evaluation(results)]
                )

# Plot tarin and validation metric across time 
lgb.plot_metric(results)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Parallelize hyperparameter tuning for LightGBM
# MAGIC To parallelize hyperparameter tuning we will perform two steps:
# MAGIC - 2a. instantiate a Ray cluster - a Ray cluster is composed of multi-nodes for computing. Since this is Ray on Spark, we can assign `min/max worker_nodes` equal to (or less than) the number of worker nodes in the Spark cluster and `num_cpus_per_node` to the number of CPUs allocated per worker in the Spark cluster. 
# MAGIC - 2b. Use Ray Tune to define and search the hyperparameter space. 
# MAGIC
# MAGIC ### 2a. Instantiate a Ray cluster
# MAGIC
# MAGIC The recommended configuration for a Ray cluster is as follows:
# MAGIC - set the `num_cpus_per_node` to the CPU count per worker node (with this configuration, each Apache Spark worker node launches one Ray worker node that will fully utilize the resources of each Apache Spark worker node.)
# MAGIC - set `min_worker_nodes` to the number of Ray worker nodes you want to launch on each node.
# MAGIC - set `max_worker_nodes` to the total amount of worker nodes (this and `min_worker_nodes` together enable autoscaling)

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

# The below configuration mirrors my Spark worker cluster set up. Change this to match your cluster configuration. 
setup_ray_cluster(
  min_worker_nodes=2,
  max_worker_nodes=8,
  num_cpus_per_node=16,
  num_gpus_worker_node=0,
  collect_log_to_path="/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Use Ray Tune to parallelize hyperparameter search
# MAGIC
# MAGIC ![](images/xgboost_ray_tune.jpg)

# COMMAND ----------

import os
import numpy as np
import lightgbm as lgb
import mlflow
from mlflow.utils.databricks_utils import get_databricks_env_vars
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch

# Set the number of trials to run
num_samples = 48

# Set mlflow experiment name
experiment_name = '/Users/jon.cheung@databricks.com/ray-lgbm-foot-locker'
mlflow.set_experiment(experiment_name)
mlflow_db_creds = get_databricks_env_vars("databricks")

# Define a training function to parallelize
def train_global_forecaster(config: dict,
                            experiment_name: str,
                            parent_run_id: str,
                            mlflow_credentials: dict,
                            ):
    """
    This objective function trains a LGBM model given a set of sampled hyperparameters. There is no returned value but a metric that is sent back to the driver node to update the progress of the HPO run.

    config: dict, defining the sampled hyperparameters to train the model on.
    **The below three parameters are used for nesting each HPO run as a child run**
    experiment_name: str, the name of the mlflow experiment to log to. This is inherited from the driver node that initiates the mlflow parent run.
    parent_run_id: str, the ID of the parent run. This is inherited from the driver node that initiates the mlflow parent run.
    mlflow_credentials: dict, the credentials for logging to mlflow. This is inherited from the driver node. 
    """
    # Set mlflow credentials and active MLflow experiment within each Ray task
    os.environ.update(mlflow_db_creds)
    mlflow.set_experiment(experiment_name)

    # Write code to import your dataset here
    data, labels = make_regression(n_samples=10000000, 
                                   n_features=55, 
                                   n_informative=25, 
                                   n_targets=1)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    # Build input matrices for LightGBM
    train_set = lgb.Dataset(train_x, label=train_y)
    test_set = lgb.Dataset(test_x, label=test_y)
    with mlflow.start_run(run_name='lgbm_forecast_model_hpo',
                          parent_run_id=parent_run_id):
        # Train the classifier
        results = {}
        gbm = lgb.train(config,
                        train_set,
                        valid_sets=[test_set],
                        valid_names=["validation"],
                        callbacks = [lgb.record_evaluation(results)]
                        )
        # get RMSE of validation set for last iteration
        rmse = results['validation']['rmse'][-1]

        # write mlflow metrics
        mlflow.log_params(config)
        mlflow.log_metrics({'validation_rmse': rmse})

    # Return evaluation results back to driver node
    train.report({"rmse": rmse, "done": True})

# By default, Ray Tune uses 1 CPU/trial. LightGBM leverages hyper-threading so we will utilize all CPUs in a node per instance. Since I've set up my nodes to have 64 CPUs each, I'll set the "cpu" parameter to 64. Feel free to tune this down if you're seeing that you're not utilizing all the CPUs in the cluster. 
trainable_with_resources = tune.with_resources(train_global_forecaster, 
                                               {"cpu": 16})

# Define the hyperparameter search space.
param_space = {
    "objective": "regression_l1",
    # "objective": tune.choice(["regression_l1", "tweedie"]),
    "max_depth": 7,
    "min_data_in_leaf": 500, 
    "boosting": tune.choice(['dart', 'gbdt']), 
    "extra_trees": True,
    "learning_rate": tune.uniform(0.01, 0.3),
    "bagging_fraction": tune.uniform(0.6, 0.8),
    "bagging_freq": tune.randint(1, 5),
    "metrics": "rmse",
    "num_threads": 64,    
    "n_estimators": tune.randint(100, 1000),
    "num_leaves": tune.randint(10, 100),
    "early_stopping_round": tune.randint(3, 20),
}

# Set up search algorithm. Here we use Optuna and set the sampler to a Bayesian one (i.e. TPES)
optuna = OptunaSearch(metric="rmse", 
                      mode="min")

with mlflow.start_run(run_name ='parallelized_64_cores') as parent_run:
    tuner = tune.Tuner(
        ray.tune.with_parameters(
            trainable_with_resources,
            experiment_name=experiment_name,
            parent_run_id = parent_run.info.run_id,
            mlflow_credentials=mlflow_db_creds),
        tune_config=tune.TuneConfig(num_samples=num_samples,
                                    search_alg=optuna),
        param_space=param_space
        )
    results = tuner.fit()

results.get_best_result(metric="rmse", 
                        mode="min").config
