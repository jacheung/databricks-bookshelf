# Databricks notebook source
# MAGIC %pip install -U "statsforecast[fugue]==1.7.6"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Forecasting using StatsForecast with SparkDF with Intermittent Forecasting Models

# COMMAND ----------

from statsforecast import StatsForecast
from statsforecast.distributed.fugue import FugueBackend
from statsforecast.models import AutoARIMA
from statsforecast.models import (
    Naive,
    CrostonClassic, 
    CrostonOptimized,
    CrostonSBA,
    IMAPA,
    ADIDA
)
from statsforecast.utils import AirPassengersDF

# COMMAND ----------


# generate spark dataframe from the pandas dataframe
sdf = spark.createDataFrame(AirPassengersDF)

# COMMAND ----------

# instantiate the StatsForecast class with the models you want to forecast with
sf = StatsForecast(
    models=[ 
      Naive(), # include Naive to make the comparison easier
      CrostonClassic(),
      CrostonOptimized(),
      CrostonSBA(),
      IMAPA(),
      ADIDA(),
     ], 
    fallback_model=Naive(),
    freq='D'
)

#
y_pred = sf.forecast(
    df=sdf,
    h=28
  )
 
# display the forecasting results of the PySpark DataFrame
y_pred.display()

# COMMAND ----------

# 5-fold CV
y_pred_cv = sf.cross_validation(df=sdf,
                                  h=28,
                                  step_size=5,
                                  n_windows=5)

y_pred_cv.display()

              

# COMMAND ----------

# MAGIC %md 
# MAGIC # Forecasting using Fugue Backend

# COMMAND ----------

backend = FugueBackend(spark, {"fugue.spark.use_pandas_udf":True})


# COMMAND ----------

from statsforecast.models import ETS
init = time()
ets_forecasts = backend.forecast(
    "s3://m5-benchmarks/data/train/m5-target.parquet", 
    [ETS(season_length=7, model='ZAA')], 
    freq="D", 
    h=28, 
).toPandas()
end = time()
print(f'Minutes taken by StatsForecast on a Spark cluster: {(end - init) / 60}')

