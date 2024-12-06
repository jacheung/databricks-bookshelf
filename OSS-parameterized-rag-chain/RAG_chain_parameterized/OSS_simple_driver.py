# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk mlflow-skinny mlflow mlflow[gateway] langchain langchain_core faiss-gpu-cu12 langchain_community transformers langchain_huggingface
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow


model_config = mlflow.models.ModelConfig(development_config='../rag_chain_config.yaml')

# COMMAND ----------

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name=model_config.get("faiss_embedding_model"))

new_vector_store = FAISS.load_local(
    model_config.get('faiss_dbfs_cache_dir'), embeddings, allow_dangerous_deserialization=True
)

docs = new_vector_store.similarity_search("money")
docs[0]

# COMMAND ----------

import os
# artifact_dict = {'vector_index': model_config.get("faiss_dbfs_cache_dir").replace('/dbfs','dbfs:')}

mlflow.set_tracking_uri('databricks')
mlflow.set_registry_uri('databricks')

# Log the model to MLflow
with mlflow.start_run(run_name="lol-non-uc"):
  logged_chain_info = mlflow.langchain.log_model(
          lc_model=os.path.join(os.getcwd(), 'simple_rag_chain'),  # Chain code file e.g., /path/to/the/chain.py 
          model_config='../rag_chain_config.yaml', # Chain configuration 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          registered_model_name=model_config.get("faiss_embedding_model"))
          # input_example=input_example,
          # example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
      )


# COMMAND ----------

# mlflow.pyfunc.get_model_dependencies(logged_chain_info.model_uri)

mlflow_model = mlflow.pyfunc.load_model(logged_chain_info.model_uri)


# COMMAND ----------

# MAGIC %pip install -r /local_disk0/repl_tmp_data/ReplId-19398-50e0d-4/tmpnw9_t5lh/chain/requirements.txt

# COMMAND ----------


