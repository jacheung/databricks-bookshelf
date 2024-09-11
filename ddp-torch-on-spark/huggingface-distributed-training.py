# Databricks notebook source
# MAGIC %md
# MAGIC https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html
# MAGIC
# MAGIC
# MAGIC HuggingFace tranformer training https://docs.databricks.com/_extras/notebooks/source/deep-learning/distributed-fine-tuning-hugging-face.html
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC The TorchDistributor has three main configurations
# MAGIC
# MAGIC 1. `num_processes` refers to the number of spark tasks to be run.
# MAGIC 2. `local_mode` refers to training on the driver node versus training on worker nodes. When training on a single node set local_mode=True
# MAGIC 3. `use_gpu` determines whether we will train using GPUs or not.

# COMMAND ----------

import torch
 
NUM_WORKERS = 4
 
def get_gpus_per_worker(_):
  import torch
  return torch.cuda.device_count()
 
NUM_GPUS_PER_WORKER = sc.parallelize(range(4), 4).map(get_gpus_per_worker).collect()[0]
USE_GPU = NUM_GPUS_PER_WORKER > 0
if USE_GPU:
  print(f'GPUs found. {NUM_GPUS_PER_WORKER} per worker. Enabling GPUs.')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Preprocess the IMDB dataset
# MAGIC
# MAGIC One key difference in the Hugging Face blog and this notebook is that this example uses **all** of the IMDB data, not just 3000 data points.

# COMMAND ----------

import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import load_dataset # import the huggingface Datasets option
import pandas as pd

# Initialize the tokenizer and collator for preprocessing the data.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load in the dataset from HuggingFace's Datasets
imdb = load_dataset("imdb")
train = pd.DataFrame(imdb["train"])
test = pd.DataFrame(imdb["test"])

# The IMDb Movie Reviews dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. The dataset contains an even number of positive and negative reviews. Only highly polarizing reviews are considered. `texts` is the review and `labels` is 0/1, 0 for negative and 1 for positive.
texts = train["text"].tolist()
labels = train["label"].tolist()

# Split the dataset for training and testing
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2)

# Encode the dataset
train_encodings = tokenizer(train_texts, truncation=True)
val_encodings = tokenizer(val_texts, truncation=True)

class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# PyTorch Dataset Object. 
# Each item in the list is a dictionary with the keys ('input_ids', 'attention_mask', 'labels')
# 'input_ids' is the vectorized text, 
# 'attention_mask' is a vector of 1s and 0s to indicate which tokens are in the text.
# 'labels' is the label for the text.
tokenized_train = ImdbDataset(train_encodings, train_labels)
tokenized_test = ImdbDataset(val_encodings, val_labels)

# COMMAND ----------

# MAGIC %md ## Set up the training function
# MAGIC
# MAGIC The TorchDistributor API has support for single node multi-GPU training as well as multi-node training. 
# MAGIC
# MAGIC When you wrap the single-node code in the `train()` function, Databricks recommends you include all the import statements inside the train() function to avoid library pickling issues. You can return any picklable object in `train_model()`, but that means you can't return `Trainer` since that can't be picklable without a process group. You can instead return the best checkpoint path and use that externally. 

# COMMAND ----------

import numpy as np
from datasets import load_metric

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}
 
output_dir = "/Users/jon.cheung@databricks.com/imdb/finetuning-sentiment-model-v1" # Save to DBFS (required)

def train_model():
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
      output_dir=output_dir,
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=2,
      weight_decay=0.01,
      save_strategy="epoch",
      report_to=[], # REMOVE MLFLOW INTEGRATION FOR NOW
      push_to_hub=False,  # DO NOT PUSH TO MODEL HUB FOR NOW,
      load_best_model_at_end=True, # RECOMMENDED
      metric_for_best_model="eval_loss", # RECOMMENDED
      evaluation_strategy="epoch" # RECOMMENDED
    )

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_test,
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.state.best_model_checkpoint

# It is recommended to create a separate local trainer from pretrained model instead of using the trainer used in distributed training
def test_model(ckpt_path):
  model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=2)
  local_trainer = Trainer(model=model,eval_dataset=tokenized_test,tokenizer=tokenizer,data_collator=data_collator,compute_metrics=compute_metrics)
  return local_trainer.evaluate()

def test_example(ckpt_path, inputs):
  from transformers import pipeline
  model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=2)
  p = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
  outputs = p(inputs)
  return ["Positive" if item["label"] == "LABEL_0" else "Negative" for item in outputs]

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test local training without GPUs. 

# COMMAND ----------

single_node_ckpt_path = train_model()

# COMMAND ----------

test_model(single_node_ckpt_path)

# COMMAND ----------

# MAGIC %md ## Run distributed training on a single node with multiple GPUs
# MAGIC
# MAGIC Distributor with `local_mode=True` will run the `train()` function directly on the driver node of the spark cluster.
# MAGIC
# MAGIC To configure how many GPUs to use in total for this run, pass `num_processes=N` to the Distributor where `N` is the number of GPUs you want to use on the driver node. Note that you don't need to actually make any changes to your training code.

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

NUM_PROCESSES = torch.cuda.device_count()
print(f"We're using {NUM_PROCESSES} GPUs")
single_node_multi_gpu_ckpt_path = TorchDistributor(num_processes=NUM_PROCESSES, local_mode=True, use_gpu=USE_GPU).run(train_model)

# COMMAND ----------

test_model(single_node_multi_gpu_ckpt_path)

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

# 1. Development workflow for notebooks
# Run PyTorch Training locally on GPU (using a PyTorch native function)
result = TorchDistributor(num_processes=2, 
local_mode=True, 
use_gpu=True).run(<function_or_script>,
<args>)

# 2. Migrate training from external repositories
# Run PyTorch Training on GPU (using a file with PyTorch code)
train_file = "/path/to/train.py"
args = ["--learning_rate=0.001", "--batch_size=16"]
distributor = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True)
distributor.run(train_file, *args)


