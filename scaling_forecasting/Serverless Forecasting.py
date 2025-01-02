# Databricks notebook source
# MAGIC %pip install skforecast

# COMMAND ----------

from skforecast.datasets import fetch_dataset
data = fetch_dataset(name="store_sales")

# COMMAND ----------

data.reset_index()

# COMMAND ----------

df_spark = spark.createDataFrame(data.reset_index())

# Specify the Unity Catalog location
catalog = "jon_cheung"
schema = "sales_forecasting"
table_name = "store_x_item"

# Write the Spark DataFrame to Unity Catalog as a Delta table
df_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.{table_name}")


