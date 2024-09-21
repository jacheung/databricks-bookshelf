# Databricks notebook source
# MAGIC %md ## Reg_SPD workflow

# COMMAND ----------

# pip install azureml-sdk azureml-mlflow azureml-core simplejson mlxtend pandas-profiling spark-df-profiling skforecast prophet

# COMMAND ----------

import os
import argparse
import sys
import joblib
import pickle
import logging
import random
from random import randrange
import time
import json
from itertools import chain
import math
import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import  PoissonRegressor, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import pyspark.sql.functions as func
from pyspark.sql.window import Window
from pyspark.sql.functions import col
from pyspark import TaskContext

from datetime import date

# COMMAND ----------

spark.conf.set('spark.sql.adaptive.enabled', 'false')
spark.conf.set('spark.sql.shuffle.partitions', 2000)

# COMMAND ----------

#%sql
#OPTIMIZE demand_forecast_parent.L6_000014 ZORDER BY (p_id);

# COMMAND ----------

# MAGIC %md #### Create features

# COMMAND ----------

#Week Flags
num_weeks = 52
week_features = []
for week in range(num_weeks):
  week_features.append(f"wk{week + 1}")
  
#Christmas Days
todays_date = date.today()
year = todays_date.year
christmas_excl = []
for christmas in range(2013, year):
  christmas_excl.append(f"{christmas+2}-12-25")

# COMMAND ----------

#Pids Filter
pids_df = (spark.table('demand_forecast_prod.fcst_pids').select('p_id'))
                                                      
# Collect distributed data to driver node
pids = pids_df.collect()

# Covert to Python list
pid_list = [row.p_id for row in pids]

# COMMAND ----------

# Time: 10 minutes (likely bc of the window function)
features_columns = [
                'p_id', 'ut_id', 'cldr_day_of_wk_id', 'day_dt', 'ntl_log_bse_pr_am', 'ntl_log_bse_promo_dct_pr_am', 'fcl_per_of_yr_id',

                'adv_circ_flg', 'adv_mid_wk_flg', 'adv_super_evnt_flg', 'adv_dgtl_circ_flg',
  
  			    'ny', 'superbowlsat', 'dayb4valentine', 'valentine', 'easter', 'easterwk',	
			    'momsat', 'mom', 'memorial', 'memwkend', 'dadsat', 'dad', 'julyfour', 'labor',	
			    'laborwkend', 'laborfri', 'laborsat', 'laborsun', 'columbus', 'mcc_blk_out',
			    'prehalo_fri', 'prehalo_sat', 'prehalo_sun',
			    'halloweeneve', 'halloweenfri', 'halloween', 'veterans', 
			    'fridayb4turkey', 'saturdayb4turkey', 'turkeywed', 'turkey', 'blackfri', 	
			    'redsat', 'dectwothree', 'dectwosix', 'sunprexmas', 'xmaseve', 'sunpny', 'satpny', 'nyeve',
  
                'mkt_bskt_ut_qt', 'ntl_log_mkt_bskt_ut_qt'] + week_features

record_cnt_window = Window.partitionBy('p_id', 'ut_id', 'cldr_day_of_wk_id').orderBy('p_id')

# This is all of L6's
features = (spark.table('demand_forecast_prod.L6').select(features_columns)
                                                  .filter(col('p_id').isin (pid_list))
                                                  .fillna(value = 0)
                                                  .withColumn('count', func.count('*').over(record_cnt_window))
                                                  .filter(col('day_dt').isin (christmas_excl)== False)
                                                  .filter(col('count') >= 1))

# COMMAND ----------

# MAGIC %md #### Define PandasUDF

# COMMAND ----------

# Works product, store, day of week level
# def configure_model_spd_udf(model_type, model, target_column, features_columns, timestamp_column, id_columns, oos_weeks):
  
#   def model_udf(group_data):
    
#     start_time = datetime.datetime.now()
    
#     # Capture metadata and set additional parameters
#     product = group_data['p_id'][0]
#     store = group_data['ut_id'][0]
#     day = group_data['cldr_day_of_wk_id'][0]
    
    
#     model_name = model_type+"_store"+str(store) +"_day"+str(day)+"_"+str(product)

#     group_data = group_data.set_index(timestamp_column).sort_index(ascending=True)
                                                                     
#     # Splits train and test based on oos_weeks
#     max_date = group_data.index.max()
#     min_date = group_data.index.min()

#     test_size = oos_weeks * 7
#     split_date_min = max_date - datetime.timedelta(test_size+1)
#     split_date_max = max_date - datetime.timedelta(test_size)

#     train = group_data.loc[min_date:split_date_min]
#     test = group_data.loc[split_date_max:max_date]
    
#     train_cnt = train.shape[0]
#     test_cnt = test.shape[0]

#     training_failed = 0
    
#     try:
#       model.fit(train[feature_columns], train[target_column])
    
#       insample_predictions = model.predict(train[feature_columns])
#       outsample_predictions = model.predict(test[feature_columns])

#       digits = 3
#       train_mse = round(mean_squared_error(insample_predictions, train[target_column]), digits)
#       train_rmse = np.sqrt(train_mse)
#       train_mae = mean_absolute_error(insample_predictions, train[target_column])
#       train_r_squared = r2_score(insample_predictions, train[target_column])
#       train_mape = mean_absolute_percentage_error(insample_predictions, train[target_column])
#       train_wmape = mean_absolute_percentage_error(insample_predictions, train[target_column], sample_weight=train[target_column])

#       test_mse = round(mean_squared_error(outsample_predictions, test[target_column]), digits)
#       test_rmse = np.sqrt(test_mse)
#       test_mae = mean_absolute_error(outsample_predictions, test[target_column])
#       test_r_squared = r2_score(outsample_predictions, test[target_column])
#       test_mape = mean_absolute_percentage_error(outsample_predictions, test[target_column])
#       test_wmape = mean_absolute_percentage_error(outsample_predictions, test[target_column]) #TODO: Add weighting 

#       stage_id = TaskContext().stageId()
#       task_id = TaskContext().taskAttemptId()

#       end_time = datetime.datetime.now()
#       elapsed = end_time-start_time
#       seconds = round(elapsed.total_seconds(), 3)

#       return pd.DataFrame({'p_id':              [product],
#                           'ut_id':             [store], 
#                           'cldr_day_of_wk_id': [day], 
#                           'stage_id':          [stage_id], 
#                           'task_id':           [task_id],
#                           'model_type':        [model_type],
#                           'model_name':        [model_name],
#                           'start_time':        [str(start_time)], 
#                           'end_time':          [str(end_time)], 
#                           'seconds':           [seconds], 
#                           'train_mse':         [train_mse], 
#                           'train_rmse':        [train_rmse], 
#                           'train_mae':         [train_mae],
#                           'train_r_squared':   [train_r_squared], 
#                           'train_mape':        [train_mape], 
#                           'train_wmape':       [train_wmape], 
#                           'test_mse':          [test_mse], 
#                           'test_rmse':         [test_rmse], 
#                           'test_mae':          [test_mae],
#                           'test_r_squared':    [test_r_squared], 
#                           'test_mape':         [test_mape], 
#                           'test_wmape':        [test_wmape], 
#                           'train_rows':        [train_cnt],
#                           'test_rows':         [test_cnt],
#                           'max_date':          [max_date],
#                           'min_date':          [min_date],
#                           'split_date_min':    [split_date_min],
#                           'split_date_max':    [split_date_max],
#                           'training_failed':   [training_failed],
#                           'model':             [pickle.dumps(model)]
#                          })
    
#     except:
#       training_failed = 1
      
#       end_time = datetime.datetime.now()
#       elapsed = end_time-start_time
#       seconds = round(elapsed.total_seconds(), 3)

#       stage_id = TaskContext().stageId()
#       task_id = TaskContext().taskAttemptId()
      
#       return pd.DataFrame({'p_id':              [product],
#                            'ut_id':             [store], 
#                            'cldr_day_of_wk_id': [day], 
#                            'stage_id':          [stage_id], 
#                            'task_id':           [task_id],
#                            'model_type':        [model_type],
#                            'model_name':        [model_name],
#                            'start_time':        [str(start_time)], 
#                            'end_time':          [str(end_time)], 
#                            'seconds':           [seconds], 
#                            'train_mse':         [0.0], 
#                            'train_rmse':        [0.0], 
#                            'train_mae':         [0.0],
#                            'train_r_squared':   [0.0], 
#                            'train_mape':        [0.0], 
#                            'train_wmape':       [0.0], 
#                            'test_mse':          [0.0], 
#                            'test_rmse':         [0.0], 
#                            'test_mae':          [0.0],
#                            'test_r_squared':    [0.0], 
#                            'test_mape':         [0.0], 
#                            'test_wmape':        [0.0], 
#                            'train_rows':        [train_cnt],
#                            'test_rows':         [test_cnt],
#                            'max_date':          [max_date],
#                            'min_date':          [min_date],
#                            'split_date_min':    [split_date_min],
#                            'split_date_max':    [split_date_max],
#                            'training_failed':   [training_failed],
#                            'model':             [b""]
#                           })
      
      
      

#   return model_udf

# COMMAND ----------

from pyspark.sql.functions import pandas_udf

def configure_model_spd_udf(model_type, model, target_column, features_columns, timestamp_column, id_columns, oos_weeks):
  
  def model_udf_loop(group_data: pd.DataFrame) -> pd.DataFrame:
    
    start_time = datetime.datetime.now()
    return_models = []
    stage_id = TaskContext().stageId()
    task_id = TaskContext().taskAttemptId()
    
    for day, group in group_data.groupby("cldr_day_of_wk_id"):
    
      try:
        # Capture metadata and set additional parameters
        product = group['p_id'].iloc[0]
        store = group['ut_id'].iloc[0]
        model_name = f"{model_type}_store{store}_day{day}_{product}"

        group = group.set_index(timestamp_column).sort_index(ascending=True)

        # Splits train and test based on oos_weeks
        max_date = group.index.max()
        min_date = group.index.min()

        test_size = oos_weeks * 7
        split_date_min = max_date - datetime.timedelta(test_size+1)
        split_date_max = max_date - datetime.timedelta(test_size)

        # What's the advantage of a train/test split here?
        train = group.loc[min_date:split_date_min]
        test = group.loc[split_date_max:max_date]

        train_cnt = train.shape[0]
        test_cnt = test.shape[0]

        training_failed = 0

      
        model.fit(train[feature_columns], train[target_column])

        insample_predictions = model.predict(train[feature_columns])
        outsample_predictions = model.predict(test[feature_columns])

        digits = 3
        train_mse = round(mean_squared_error(insample_predictions, train[target_column]), digits)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(insample_predictions, train[target_column])
        train_r_squared = r2_score(insample_predictions, train[target_column])
        train_mape = mean_absolute_percentage_error(insample_predictions, train[target_column])
        train_wmape = mean_absolute_percentage_error(insample_predictions, train[target_column])

        test_mse = round(mean_squared_error(outsample_predictions, test[target_column]), digits)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(outsample_predictions, test[target_column])
        test_r_squared = r2_score(outsample_predictions, test[target_column])
        test_mape = mean_absolute_percentage_error(outsample_predictions, test[target_column])
        test_wmape = mean_absolute_percentage_error(outsample_predictions, test[target_column])

        end_time = datetime.datetime.now()
        elapsed = end_time-start_time
        seconds = round(elapsed.total_seconds(), 3)

        return_models.append({
                            'p_id':              product,
                            'ut_id':             store, 
                            'cldr_day_of_wk_id': day, 
                            'stage_id':          stage_id, 
                            'task_id':           task_id,
                            'model_type':        model_type,
                            'model_name':        model_name,
                            'start_time':        str(start_time), 
                            'end_time':          str(end_time), 
                            'seconds':           seconds, 
                            'train_mse':         train_mse, 
                            'train_rmse':        train_rmse, 
                            'train_mae':         train_mae,
                            'train_r_squared':   train_r_squared, 
                            'train_mape':        train_mape, 
                            'train_wmape':       train_wmape, 
                            'test_mse':          test_mse, 
                            'test_rmse':         test_rmse, 
                            'test_mae':          test_mae,
                            'test_r_squared':    test_r_squared, 
                            'test_mape':         test_mape, 
                            'test_wmape':        test_wmape, 
                            'train_rows':        train_cnt,
                            'test_rows':         test_cnt,
                            'max_date':          max_date,
                            'min_date':          min_date,
                            'split_date_min':    split_date_min,
                            'split_date_max':    split_date_max,
                            'training_failed':   training_failed,
                            'error_if_failed':   '',
                            'model':             pickle.dumps(model)
                           })
  

      except Exception as err:
        training_failed = 1

        end_time = datetime.datetime.now()
        elapsed = end_time-start_time
        seconds = round(elapsed.total_seconds(), 3)

        return_models.append({
                           'p_id':              group_data['p_id'][0],
                           'ut_id':             group_data['ut_id'][0], 
                           'cldr_day_of_wk_id': day, 
                           'stage_id':          stage_id, 
                           'task_id':           task_id,
                           'model_type':        model_type,
                           'model_name':        model_name,
                           'start_time':        str(start_time), 
                           'end_time':          str(end_time), 
                           'seconds':           seconds, 
                           'train_mse':         0.0, 
                           'train_rmse':        0.0, 
                           'train_mae':         0.0,
                           'train_r_squared':   0.0, 
                           'train_mape':        0.0, 
                           'train_wmape':       0.0, 
                           'test_mse':          0.0, 
                           'test_rmse':         0.0, 
                           'test_mae':          0.0,
                           'test_r_squared':    0.0, 
                           'test_mape':         0.0, 
                           'test_wmape':        0.0, 
                           'train_rows':        train_cnt,
                           'test_rows':         test_cnt,
                           'max_date':          max_date,
                           'min_date':          min_date,
                           'split_date_min':    split_date_min,
                           'split_date_max':    split_date_max,
                           'training_failed':   training_failed,
                           'error_if_failed':   str(err),
                           'model':             b""
                          })
    return pd.DataFrame(return_models)


  return model_udf_loop

# COMMAND ----------

#Model Type
model_type = "Reg_SPD"
model = LinearRegression() # TODO: is this correct to define the model in this scope?

#Create Model Directory
#dbutils.fs.mkdirs(f"mnt/DS/Merch/DmdFcst/Batch/Model_Dir/{model_type}")

target_column = 'ntl_log_mkt_bskt_ut_qt'
id_columns = ['p_id','ut_id']
# id_columns = ['p_id','ut_id','cldr_day_of_wk_id']
timestamp_column= 'day_dt'
non_feature_columns = id_columns + [target_column, timestamp_column, 'mkt_bskt_ut_qt','count']
feature_columns =  [column for column in features_columns if column not in non_feature_columns]
oos_weeks = 6

udf_train =  configure_model_spd_udf(model_type, model, target_column, feature_columns, timestamp_column, id_columns, oos_weeks)

schema = """p_id int, 
           ut_id int, 
           cldr_day_of_wk_id int, 
           stage_id int, 
           task_id int,
           model_type string,
           model_name string,
           start_time string,
           end_time string,
           seconds float,
           train_mse float,
           train_rmse float,
           train_mae float,
           train_r_squared float,
           train_mape float,
           train_wmape float,
           test_mse float,
           test_rmse float,
           test_mae float,
           test_r_squared float,
           test_mape float,
           test_wmape float,
           train_rows int,
           test_rows int,
           max_date date, 
           min_date date, 
           split_date_max date, 
           split_date_min date, 
           training_failed int,
           error_if_failed string,
           model binary
           """

results = features.groupBy(id_columns).applyInPandas(udf_train, schema=schema)
results.write.mode('overwrite').format("delta").option("overwriteSchema", "true").saveAsTable("demand_forecast_prod.demand_forecast_spdl_results_mgm_test_arrow_batch_size")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- OPTIMIZE demand_forecast_prod.demand_forecast_spdl_results_mgm_test
# MAGIC OPTIMIZE demand_forecast_prod.demand_forecast_spdl_results_mgm_test_arrow_batch_size
# MAGIC ZORDER BY (p_id, ut_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- VACUUM demand_forecast_prod.demand_forecast_spdl_results_mgm_test
# MAGIC VACUUM demand_forecast_prod.demand_forecast_spdl_results_mgm_test_arrow_batch_size
