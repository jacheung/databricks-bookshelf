# Databricks notebook source
# MAGIC %md 
# MAGIC ## Parallelizing XGBoost with Ray
# MAGIC
# MAGIC XGBoost is one of the most common and powerful boosting models out there; it can be used for both regression and classification problems. Amidst its powerful feature set, there are drawbacks with training time and the requirement to perform extensive hyperparameter search to reduce overfitting. To meet this compute demand, XGBoost natively leverages hyper-threading, allowing it to use all the CPU cores on a single-machine. However, what if hyper-threading is not enough?
# MAGIC
# MAGIC Here comes Ray to the rescue. Ray offers a distributed version of XGBoost to further reduce computing time. With drop-in replacements of `xgboost` native classes, `xgboost_ray` allows you to leverage multi-node clusters to further distribute your training. For example, instead of using 16 cores in a single-node, `xgboost_ray` allows you to scale that up to a multi-node cluster, leveraging 16 cores PER node to further reduce training time. 

# COMMAND ----------

# MAGIC %pip install --quiet skforecast scikit-lego xgboost_ray
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Create and prepare a dataset
# MAGIC We'll be using the M4 dataset which has 100k different time-series available. We'll be building one global time-series model. This means we'll be using one model to forecast and predict. For the data preparation, we will:
# MAGIC 1. use a radial basis function to encode the day of year
# MAGIC 2. one-hot encode the unique identifier

# COMMAND ----------

import pandas as pd
from skforecast.datasets import fetch_dataset
from sklearn.preprocessing import OneHotEncoder
from sklego.preprocessing import RepeatingBasisFunction

def create_m4_daily(n_series: int):
    y_df = fetch_dataset(name="m4_daily")
    _ids = [f"D{i}" for i in range(1, n_series+1)]
    X_df = (
        y_df.groupby("series_id")
        .filter(lambda x: x.series_id.iloc[0] in _ids)
        .groupby("series_id")
        .apply(transform_group)
        .reset_index(drop=True)
    )

    X_df = X_df.infer_objects()
    y = X_df.pop('y')
    X_df.fillna(0, inplace=True)
    return X_df, y

def transform_group(df):
    unique_id = df['series_id'].iloc[0]
    if len(df) > 1020:
        df = df.iloc[-1020:]
    _start = pd.Timestamp("2020-01-01")
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")
    res_df = pd.DataFrame({'year': date_idx.year,
                          'day_of_year': date_idx.day_of_year,
                          'series_id': unique_id,
                          'y': df.value.values})

    # Radial Basis Function encode the time variable "day of year" as a 
    encoding_periods = 12
    rbf_daily = RepeatingBasisFunction(
          n_periods   = encoding_periods,
          column      = 'day_of_year',
          input_range = (1, 365),
          remainder="passthrough"
          )
    res_df = rbf_daily.fit_transform(res_df)
    column_names=[f"rbf_daily_{x}" for x in range(encoding_periods)] + ['year', 'series_id', 'y']
    rbf_df = pd.DataFrame(data=res_df,
                          columns=column_names)
    
    # One-hot encode the series_id
    encoder = OneHotEncoder(sparse_output=False, 
                            handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(df[['series_id']])
    one_hot_df = pd.DataFrame(one_hot_encoded, 
                              columns=encoder.get_feature_names_out(['series_id']))

    output_df = pd.concat([rbf_df, one_hot_df], axis=1).drop('series_id', axis=1)

    return output_df


# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Train a single XGBoost model
# MAGIC We'll test our dataset for an XGBoost Regression model. Here we'll instantiate a simple configuration for the hyperparameters and train-test split. XGBoost here uses one node and hyper-threading, utilizing all available CPU cores to train the model. We'll scale this to a multi-node cluster using Ray in the next section.

# COMMAND ----------

import xgboost as xgb
from sklearn.model_selection import train_test_split

config = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.7
}

# parameter to define how many time-series datasets to train on. Max is 100k. We'll use a small subset for this testing round. 
data, labels = create_m4_daily(n_series=10)
# Split into train and test set
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
# Build input matrices for XGBoost
train_set = xgb.DMatrix(train_x, label=train_y)
test_set = xgb.DMatrix(test_x, label=test_y)
# Train the classifier
results = {}
xgb.train(
    config,
    train_set,
    evals=[(test_set, "eval")],
    evals_result=results,
    verbose_eval=False,
)
# Return prediction accuracy
rmse = results["eval"]["rmse"][-1]


# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3. Parallelize hyperparameter tuning for XGBoost
# MAGIC To parallelize hyperparameter tuning we will perform two steps:
# MAGIC - 3a. instantiate a Ray cluster - a Ray cluster just means utilizing multi-nodes for computing. Since this is Ray on Spark, we can assign worker_nodes equal to the number of worker nodes in the cluster and num_cpus_per_node to the number of CPUs allocated per worker. 
# MAGIC - 3b. Utilize Ray Tune to define the hyperparameter space and search. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Instantiate a Ray cluster
# MAGIC
# MAGIC The recommended configuration for a Ray cluster is as follows:
# MAGIC - set the num_cpus_per_node to the CPU count per worker node ( ith this configuration, each Apache Spark worker node launches one Ray worker node that will fully utilize the resources of each Apache Spark worker node.)
# MAGIC - set min_worker_nodes to the number of Ray worker nodes you want to launch on each node.
# MAGIC - set max_worker_nodes to the total amount of worker nodes (this and `min_worker_nodes` together enable autoscaling)

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

# MAGIC %md
# MAGIC ### 3b. Use Ray Tune to parallelize hyperparameter search

# COMMAND ----------

from sklearn.model_selection import train_test_split
import xgboost as xgb
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

# Set mlflow experiment name
experiment_name = '/Users/jon.cheung@databricks.com/ray-xgb-hyperparameter-tuning'

# Define a training function to parallelize
def train_global_forecaster(config):
    # Load a slightly larger dataset
    data, labels = create_m4_daily(n_series=100)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier
    results = {}
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        evals_result=results,
        verbose_eval=False,
    )
    # Return evaluation RMSE
    rmse = results["eval"]["rmse"][-1]
    train.report({"rmse": rmse, "done": True})

# Define the hyperparameter search space and how to sample.
param_space = {
    "objective": "reg:squarederror",
    "learning_rate": tune.loguniform(0.01, 0.3),
    "n_estimators": tune.randint(100, 1000),
    "max_depth": tune.randint(6, 12),
    "min_child_weight": tune.choice([1, 2, 3]),
    "subsample": tune.uniform(0.5, 1.0)
}

# By default, Ray Tune uses 1 CPU/trial. XGBoost tends to be compute expensive and leverages hyper-threading so we will utilize all CPUs in a node. Since each of my nodes have 16 CPUs, I'll set the "cpu" parameter to 16. 
trainable_with_resources = tune.with_resources(train_global_forecaster, 
                                               {"cpu": 16})

# Run the hyperparameter search for 8 trials. 
tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(num_samples=48),
    run_config=train.RunConfig(
        name="mlflow",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri='databricks',
                experiment_name=experiment_name,
                save_artifact=True,
                )
            ],
        ),
    param_space=param_space
)
results = tuner.fit()
results.get_best_result(metric="rmse", 
                        mode="min").config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distribute training of final model
# MAGIC After hyperparameter tuning, we now have our best configuration. However, we can't just grab the best hyperparameter tuned model because that model was only trained using a subset of the data. We will now use the full dataset and the best configuration to train the final model. 
# MAGIC
# MAGIC We will use `ray_xgboost` to train our final model, assuming the final dataset is massive and will require multi-node distributed training. In our toy dataset, 810k rows of data x 1k features trained in 5 minutes on 2 actors with 16 CPU each.

# COMMAND ----------

from xgboost_ray import RayDMatrix, RayParams, RayXGBRegressor, predict
import mlflow
from mlflow.models import infer_signature

mlflow.set_experiment(experiment_name='/Users/jon.cheung@databricks.com/ray-xgb')
registered_model_name = "jon_cheung.mmf_m4.xgboost_ray_distributed_training"

# Prepare our dataset by using RayDMatrix
data, labels = create_m4_daily(n_series=500)
train_set = RayDMatrix(data, 
                       labels)

# Load in best parameters from our hyperparameter tuning
final_config = results.get_best_result(metric="rmse", 
                                       mode="min").config

with mlflow.start_run(run_name='final_model_241214'):
        """
        Defining num actors and cpus_per_actor. In short because of hyper-threading by XGBoost, it makes sense to allocate actors to the number of nodes and cpus_per_actor to the number of cpus you have per node. 
        - Example: I'm using a multi-node cluster with 8 nodes and 16cpus per node. 
        - So I'll allocate num_actors <= 8 and 16 cpus_per_actor. 
        See this doc --> https://xgboost.readthedocs.io/en/stable/tutorials/ray.html#setting-the-number-of-cpus-per-actor
        """
        clf = RayXGBRegressor()
        clf.fit(X=train_set, 
                y=None,
                ray_params=RayParams(num_actors=2,
                                     cpus_per_actor=16))
        
        # Create a signature for model input and output enforcement
        # We'll run a set of predictions to get the RayDMatrix schema enforcement on inference.
        predictions = clf.predict(train_set)
        signature = infer_signature(data, predictions)

        # Log and register our model to mlflow
        mlflow.xgboost.log_model(xgb_model=clf,
                                 artifact_path="model",
                                 registered_model_name=registered_model_name,
                                 signature=signature)
        mlflow.log_params(final_config)

# 810k rows x 1014 features = 5 minute training time on 2 actors w/ 16 CPUs each. 


# COMMAND ----------

# MAGIC %md 
# MAGIC ## 5. Distributed inference for XGBoost
# MAGIC
# MAGIC Since we are using `ray_xgboost` we can't just load the model directly from `mlflow` using `mlflow.pyfunc.load_model`. The signature enforcement from mlflow collides with the RayDMatrix requirement. 
# MAGIC
# MAGIC Instead, we will download the pickled model, load it using `xgb` native library and then pass that model into `xgboost_ray` for distributed inference. 

# COMMAND ----------

import xgboost as xgb
from mlflow import MlflowClient
import mlflow
from xgboost_ray import RayDMatrix, RayParams, predict

# Get latest registered version of model and 
client = MlflowClient()
registered_model_name = "jon_cheung.mmf_m4.xgboost_ray_distributed_training"
rms = client.search_model_versions(f"name='{registered_model_name}'")
# Download model pickle to local and load it using the xgboost library
fname = f'runs:/{rms[0].run_id}/model/model.xgb'
pickle_dir = mlflow.artifacts.download_artifacts(fname)
bst = xgb.Booster(model_file=pickle_dir)


# Generate 500 time series to match what was trained on. If this were a batch inference use-case, load your dataset here.
X, y = create_m4_daily(n_series=500)
test_X = RayDMatrix(data=X,
                    labels=None)

# Perform batch inference on 300k rows of data
predict(model=bst,
        data=test_X,
        ray_params=RayParams(num_actors=4,
                             cpus_per_actor=16))

