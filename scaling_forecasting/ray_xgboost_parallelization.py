# Databricks notebook source
# MAGIC %pip install --quiet skforecast scikit-lego xgboost_ray
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from skforecast.datasets import fetch_dataset
from sklearn.preprocessing import OneHotEncoder
from sklego.preprocessing import RepeatingBasisFunction

n = 10

def create_m4_daily():
    y_df = fetch_dataset(name="m4_daily")
    _ids = [f"D{i}" for i in range(1, n+1)]
    y_df = (
        y_df.groupby("series_id")
        .filter(lambda x: x.series_id.iloc[0] in _ids)
        .groupby("series_id")
        .apply(transform_group)
        .reset_index(drop=True)
    )
    return y_df

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
    output_df.fillna(0, inplace=True)

    return output_df

df = create_m4_daily()

# COMMAND ----------

df

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

shutdown_ray_cluster()

# COMMAND ----------

from xgboost_ray import RayDMatrix, RayParams, RayXGBRegressor

train_y = df.pop('y')
train_x = df

train_set = RayDMatrix(train_x, 
                       train_y)


clf = RayXGBRegressor()

# COMMAND ----------

from xgboost_ray import RayDMatrix, RayParams, RayXGBRegressor

train_y = df.pop('y')
train_x = df

train_set = RayDMatrix(train_x, 
                       train_y)


clf = RayXGBRegressor()

# scikit-learn API will automatically convert the data
# to RayDMatrix format as needed.
# You can also pass X as a RayDMatrix, in which case
# y will be ignored.

clf.fit(X=train_set, 
        y=None,
        ray_params=RayParams(num_actors=6,
                             cpus_per_actor=16))



# evals_result = {}
# bst = train(
#     {
#         "objective": "binary:logistic",
#         "eval_metric": ["logloss", "error"],
#     },
#     train_set,
#     evals_result=evals_result,
#     evals=[(train_set, "train")],
#     verbose_eval=False,
#     ray_params=RayParams(num_actors=32, cpus_per_actor=2))

# bst.save_model("model.xgb")
# print("Final training error: {:.4f}".format(
#     evals_result["train"]["error"][-1]))

# COMMAND ----------

# steps:
# 1) encode time using RBF
# 2) distributed data parallel using Ray 
# 3) output XGBOost model and do inference.
