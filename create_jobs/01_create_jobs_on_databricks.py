# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook shows examples of creating Jobs with serverless or classical compute.
# MAGIC
# MAGIC Read more
# MAGIC https://docs.databricks.com/en/dev-tools/sdk-python.html#code-examples

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
relative_path = "./example_lambda"

# Get the absolute path relative to the current working directory
current_dir = os.getcwd()  # This will give you the current working directory
notebook_path = os.path.abspath(os.path.join(current_dir, relative_path))

# Check if the path exists in Databricks
if os.path.exists(notebook_path):
    print(f"Path exists: {notebook_path}")
else:
    print(f"Path does not exist: {notebook_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Job that uses serverless compute

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask

w = WorkspaceClient()

job = w.jobs.create(
    name="create_serverless_job",
    tasks=[
        Task(
            task_key="workday_job",
            notebook_task=NotebookTask(notebook_path=notebook_path),
        )
    ],
)

print(f"View the job at {w.config.host}/#job/{job.job_id}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](plots/job_with_serverless_cluster.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Job with All-purpose Compute
# MAGIC
# MAGIC When you use a classical compute to run a job, first create a cluster then create the job. 
# MAGIC
# MAGIC Note that it is not recommended to run Job with All-purpose cluster. Just showing the capability to create Job and run it with any clusters. See [details](https://docs.databricks.com/en/jobs/compute.html#should-all-purpose-compute-ever-be-used-for-jobs). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a cluster 

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print("Attempting to create cluster. Please wait...")

c = w.clusters.create_and_wait(
  cluster_name             = 'my_stateful_cluster',
  spark_version            = '12.2.x-scala2.12',
  node_type_id             = 'i3.xlarge',
  autotermination_minutes = 15,
  num_workers              = 1
)

print(f"The cluster is now ready at " \
      f"{w.config.host}#setting/clusters/{c.cluster_id}/configuration\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](plots/created_cluster.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Print out the cluster IDs. We will use those ID to create job. 

# COMMAND ----------

clusters = w.clusters.list()
for cluster in clusters:
  if cluster.creator_user_name == 'jiayi.wu@databricks.com':
    print(f"Cluster Name: {cluster.cluster_name}, Cluster ID: {cluster.cluster_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a job with existing compute

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, Source

w = WorkspaceClient()

job_name            = "create_job"
description         = "a Job with existed stateful cluster"
existing_cluster_id = "1115-150914-58rv63bf"
task_key            = "workday_job"

print("Attempting to create the job. Please wait...\n")

j = w.jobs.create(
  name = job_name,
  tasks = [
    Task(
      description = description,
      existing_cluster_id = existing_cluster_id,
      notebook_task = NotebookTask(
        base_parameters = dict(""),
        notebook_path = notebook_path
      ),
      task_key = task_key
    )
  ]
)

print(f"View the job at {w.config.host}/#job/{j.job_id}\n")
