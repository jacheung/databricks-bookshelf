# Databricks notebook source
dbutils.widgets.text("my_param", "value1", "Parameter Value")

# COMMAND ----------

# Retrieve the widget value
my_param = dbutils.widgets.get("my_param")

# Print the received value
print(f"Received parameter: {my_param}")

# COMMAND ----------

# Use the parameter in your logic
if my_param == "value1":
    print("No parameter passed; using default settings.")
else:
    print(f"Using parameter: {my_param} for processing.")
