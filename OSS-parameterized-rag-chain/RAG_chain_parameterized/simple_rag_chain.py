# Databricks notebook source
# %pip install -U --quiet databricks-sdk==0.28.0 mlflow-skinny mlflow mlflow[gateway] databricks-vectorsearch langchain==0.2.1 langchain_core==0.2.5 faiss-gpu-cu12 langchain_community==0.2.4 transformers langchain_huggingface
# dbutils.library.restartPython()

%pip install -U --quiet databricks-sdk mlflow-skinny mlflow mlflow[gateway] langchain langchain_core faiss-gpu-cu12 langchain_community transformers langchain_huggingface
dbutils.library.restartPython()

# COMMAND ----------

import os
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatDatabricks
import mlflow

mlflow.set_registry_uri('databricks')

mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config='../rag_chain_config.yaml')

def load_retriever(persist_directory):
  embeddings = HuggingFaceEmbeddings(model_name=model_config.get("faiss_embedding_model"),
                                      cache_folder=model_config.get("faiss_embedding_model_dbfs_cache_dir"))
                                   
  faiss_index = FAISS.load_local(model_config.get("faiss_dbfs_cache_dir"),
                                      embeddings, 
                                      allow_dangerous_deserialization=True)
  return faiss_index.as_retriever()


prompt = hub.pull("rlm/rag-prompt-mistral")

llm = ChatDatabricks(endpoint=model_config.get("llm_model_serving_endpoint_name"))

with mlflow.start_run() as run:

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": load_retriever(model_config.get("faiss_dbfs_cache_dir")), "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)


    model_info = mlflow.langchain.log_model(
        rag_chain_with_source,
        artifact_path="retrieval_qa",
        loader_fn=load_retriever,
        persist_dir=model_config.get("faiss_dbfs_cache_dir"),
        registered_model_name=model_config.get("registered_model_name"),
        model_config='../rag_chain_config.yaml',# Chain configuration 
        )

# COMMAND ----------

from mlflow import MlflowClient

mlflow.set_registry_uri('databricks')
client=MlflowClient()
model_name=model_config.get('registered_model_name')
model_info=client.search_model_versions(f"name='{model_name}'")

mlflow_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_info[0].version}")

# COMMAND ----------

mlflow_model.predict('What can you tell me about the stock market?')

# COMMAND ----------

# import os
# import mlflow
# from langchain.vectorstores import FAISS
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain_core.output_parsers import StrOutputParser
# from langchain import hub
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.chat_models import ChatDatabricks

# embeddings = HuggingFaceEmbeddings(model_name=model_config.get("faiss_embedding_model"),
#                                    cache_folder=model_config.get("faiss_embedding_model_dbfs_cache_dir"))
                                   
# faiss_index = FAISS.load_local(model_config.get("faiss_dbfs_cache_dir"),
#                                 embeddings, 
#                                 allow_dangerous_deserialization=True)
# retriever = faiss_index.as_retriever(search_type="mmr", search_kwargs={"k": 1})

# prompt = hub.pull("rlm/rag-prompt-mistral")

# def format_docs(docs):
#     return "\n\n".join([d.page_content for d in docs])

# llm = ChatDatabricks(endpoint=model_config.get("llm_model_serving_endpoint_name"))

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()} 
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# mlflow.models.set_model(model=rag_chain)
