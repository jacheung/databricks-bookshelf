# Databricks notebook source
!pip install --quiet lightgbm scikit-learn bayesian-optimization==1.5.1 ray[tune] optuna
dbutils.library.restartPython()


# COMMAND ----------

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100000, n_features=100, n_informative=10, n_targets=1)



# COMMAND ----------

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

# Create a pseudo-dataset to test
data, labels = make_regression(n_samples=100000, 
                                   n_features=100, 
                                   n_informative=10, 
                                   n_targets=1)
# Perform train test split
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

# Build input matrices for XGBoost
train_set = lgb.Dataset(train_x, label=train_y)
test_set = lgb.Dataset(test_x, label=test_y)

# LightGBM hyperparameter configs
config = {'objective':'regression',
          'metric': 'rmse',
          'num_leaves':31,
          'learning_rate':0.05,
          'n_estimators':100,
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

ray.shutdown()

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

setup_ray_cluster(
  min_worker_nodes=4,
  max_worker_nodes=8,
  num_cpus_per_node=16,
  num_gpus_worker_node=0,
  collect_log_to_path="/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs"
)

# COMMAND ----------

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import  mean_squared_error
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow


from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch


# Set mlflow experiment name
experiment_name = '/Users/jon.cheung@databricks.com/ray-lgbm-bayesian'

# Define a training function to parallelize
def train_global_forecaster(config):
    # Write code to import your dataset here
    data, labels = make_regression(n_samples=100000, 
                                   n_features=100, 
                                   n_informative=10, 
                                   n_targets=1)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    # Build input matrices for XGBoost
    train_set = lgb.Dataset(train_x, label=train_y)
    test_set = lgb.Dataset(test_x, label=test_y)
    # Train the classifier
    results = {}
    gbm = lgb.train(config,
                    train_set,
                    valid_sets=[test_set],
                    valid_names=["validation"],
                    callbacks = [lgb.record_evaluation(results)]
                    )
    # get RMSE of last iteration
    rmse = results['validation']['rmse'][-1]
    # Return evaluation RMSE
    train.report({"rmse": rmse, "done": True})

# Define the hyperparameter search space and how to sample.
param_space = {
    "objective": "regression",
    "metrics": "rmse",
    "learning_rate": tune.uniform(0.01, 0.3),
    "n_estimators": tune.randint(100, 1000),
    "num_leaves": tune.randint(10, 100)
}

# By default, Ray Tune uses 1 CPU/trial. LightGBM leverages hyper-threading so we will utilize all CPUs in a node per instance. Since I've set up my nodes to have 16 CPUs each, I'll set the "cpu" parameter to 16. 
trainable_with_resources = tune.with_resources(train_global_forecaster, 
                                               {"cpu": 16})

# bayesopt = BayesOptSearch(metric="rmse", mode="min")
optuna = OptunaSearch(metric="rmse", mode="min")
# Run the hyperparameter search for 48 trials. 
tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(num_samples=48,
                                search_alg=optuna),
    run_config=train.RunConfig(name="mlflow",
                               callbacks=[
                                   MLflowLoggerCallback(tracking_uri='databricks',experiment_name=experiment_name,save_artifact=True,
                                                        )
                                   ],
                               ),
    param_space=param_space
    )
results = tuner.fit()
results.get_best_result(metric="rmse", 
                        mode="min").config
