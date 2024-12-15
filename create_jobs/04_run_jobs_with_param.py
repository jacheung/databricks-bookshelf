# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./03_example_lambda_with_param

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs
import time

w = WorkspaceClient()

# COMMAND ----------

notebook_params = {
    "my_param": my_param
}

job_id = "519790006359434"
# Trigger the job
run = w.jobs.run_now(
    job_id=job_id,
    notebook_params=notebook_params
)

# Get and display the Run ID
print(f"Run ID: {run.run_id}")

# COMMAND ----------

print(f"View the job at {w.config.host}/#job/{job_id}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](plots/run_with_param.png)
