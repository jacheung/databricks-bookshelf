# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook shows examples creating a multi-taask Job using SDK with a Job compute.
# MAGIC
# MAGIC Read more
# MAGIC https://databricks-sdk-py.readthedocs.io/en/latest/dbdataclasses/jobs.html#databricks.sdk.service.jobs.Task

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Databricks SDK

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Path Existance
# MAGIC
# MAGIC This path includes the notebook you will run

# COMMAND ----------

import os

# Define the relative path
relative_path = "./00_example_lambda"

# Get the absolute path relative to the current working directory
current_dir = os.getcwd()  # This will give you the current working directory
notebook_path = os.path.abspath(os.path.join(current_dir, relative_path))

# Check if the path exists in Databricks
if os.path.exists(notebook_path):
    print(f"Path exists: {notebook_path}")
else:
    print(f"Path does not exist: {notebook_path}")

# COMMAND ----------

notebook_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Multi-task Job with Job Compute
# MAGIC
# MAGIC A Job compute could be created when creating a Job.  

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs
import time

w = WorkspaceClient()

# COMMAND ----------

'''
Create a multi-task that
- uses a Job Cluster
- runs two notebook tasks (provding a dummy path you can use)
- The second task is dependent on the first

The job does NOT need to run successfully, it actually won't with the dummy notebook path

To modify the difficulty you can use an existing cluster (maybe the one you just created) or expand hint for a cluster definition
'''



# Select DBR and node type for our cluster
latest_lts = w.clusters.select_spark_version(latest=True, long_term_support=True)
node_type_id = w.clusters.select_node_type(local_disk=True)

# Define the Job cluster
cluster_def = jobs.JobCluster(
    "job-cluster",
    new_cluster=compute.ClusterSpec(
        node_type_id=node_type_id,
        spark_version=latest_lts,
        num_workers=2,
    ),
)

first_task = jobs.Task(
    description="first-task",
    job_cluster_key="job-cluster",
    notebook_task=jobs.NotebookTask(notebook_path=notebook_path),
    task_key="task1",
    timeout_seconds=0,
)

second_task = jobs.Task(
    description="test",
    job_cluster_key="job-cluster",
    notebook_task=jobs.NotebookTask(notebook_path=notebook_path),
    task_key="task2",
    depends_on=[jobs.TaskDependency(task_key="task1")],
    timeout_seconds=0,
)

created_job = w.jobs.create(
    name=f"multi_task_job_with_job_cluster",
    job_clusters=[cluster_def],
    tasks=[first_task, second_task],
)

# COMMAND ----------

print(f"View the job at {w.config.host}/#job/{created_job.job_id}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](plots/multi_task_job_with_job_cluster.png)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run a Job

# COMMAND ----------

run = w.jobs.run_now(job_id=created_job.job_id)
print(f"Run ID: {run.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC run with parameters

# COMMAND ----------

run = w.jobs.run_now(
    job_id=created_job.job_id,
    notebook_params={
        "param1": "value1",
        "param2": "value2"
    }
)
print(f"Run ID: {run.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete the Job

# COMMAND ----------

# delete the job
# w.jobs.delete(job_id=created_job.job_id)
