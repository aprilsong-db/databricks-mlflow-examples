# Databricks notebook source
# MAGIC %pip install mlflow==2.4.2

# COMMAND ----------

import mlflow

output_schema = "asong_dev.test"
output_table = "wikipedia_summaries"
number_articles = 1024


df = spark.read.parquet("/databricks-datasets/wikipedia-datasets/data-001/en_wikipedia/articles-only-parquet").select("title", "text")
display(df)

# COMMAND ----------

sample_imbalanced = df.limit(number_articles)
sample = sample_imbalanced.repartition(32).persist()
sample.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Using a pipeline in a Pandas UDF

# COMMAND ----------

# MAGIC %md
# MAGIC The next command loads the `transformers` pipeline for summarization using the `facebook/bart-large-cnn` model. 
# MAGIC
# MAGIC [Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) conveniently wrap best practices for certain tasks, bundling together tokenizers and models. They can also help with batching data sent to the GPU, so that you can perform inference on multiple items at a time. Setting the `device` to 0 causes the pipeline to use the GPU for processing. You can use this setting reliably even if you have multiple GPUs on each machine in your Spark cluster. Spark automatically reassigns GPUs to the workers.
# MAGIC
# MAGIC You can also directly load tokenizers and models if needed; you would just need to reference and invoke them directly in the UDF.

# COMMAND ----------

display(sample.limit(5))

# COMMAND ----------

from transformers import pipeline
import torch
import pandas as pd
from pyspark.sql.functions import pandas_udf
from tqdm.auto import tqdm

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@pandas_udf('string')
def summarize_batch_udf(texts: pd.Series) -> pd.Series:
  pipe = tqdm(summarizer(texts.to_list(), truncation=True, batch_size=8), total=len(texts), miniters=10)
  summaries = [summary['summary_text'] for summary in pipe]
  return pd.Series(summaries)

summaries = sample.select(sample.title, sample.text, summarize_batch_udf(sample.text).alias("summary"))
display(summaries.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # MLFlow Open AI support with function-based flavor
# MAGIC
# MAGIC
# MAGIC The OpenAI Python library provides convenient access to the OpenAI API from applications written in the Python language. It includes a predefined set of classes that map to OpenAI API resources. Usage of these provided classes will dynamically initialize connection, passing of data to, and retrieval of responses from a wide range of model versions and endpoints of the OpenAI API.
# MAGIC
# MAGIC The MLflow OpenAI flavor supports:
# MAGIC - Automatic signature schema detection
# MAGIC - Parallelized API requests for faster inference.
# MAGIC - Automatic API request retry on transient errors such as a rate limit error.
# MAGIC
# MAGIC Shown below is an example of logging the `openai.ChatCompletion` model and loading it back for inference:- -

# COMMAND ----------

import os
import mlflow
import openai

#databricks secrets put --scope database_secrets_asong --key <key>

# When the MLFLOW_OPENAI_SECRET_SCOPE environment variable is set, 
# `mlflow.openai.log_model` reads its value and saves it in `openai.yaml`
os.environ["MLFLOW_OPENAI_SECRET_SCOPE"] = "database_secrets_asong"

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        messages=[{
            "role": "user",
            "content": "Tell me a joke about {animal}.",
        }],
        artifact_path="model",
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)


# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow: logging pre-trained Hugging Face model
# MAGIC ## Save model to MLflow

# COMMAND ----------

import transformers
import mlflow 

model_architecture = "sshleifer/distilbart-cnn-12-6"
tokenizer=transformers.AutoTokenizer.from_pretrained(model_architecture)
model=transformers.BartForConditionalGeneration.from_pretrained(model_architecture)

summarizer = transformers.pipeline(
    task="summarization", 
    model=model, 
    tokenizer=tokenizer)

model_path="pipeline"
with mlflow.start_run() as run:
    mlflow.transformers.log_model(transformers_model=summarizer, 
                                  artifact_path=model_path, 
                                  input_example="Hi there!",
                                 )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model from MLflow

# COMMAND ----------

logged_model_uri = f"runs:/{run.info.run_id}/{model_path}"

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model_uri, result_type='string')

summaries = sample.select(sample.title, sample.text, loaded_model(sample.text).alias("summary"))
display(summaries.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Using `pyfunc` version of model for lightweight chatbot interfact

# COMMAND ----------

import transformers
import mlflow

chat_pipeline = transformers.pipeline(model="microsoft/DialoGPT-medium")

with mlflow.start_run():
  model_info = mlflow.transformers.log_model(
    transformers_model=chat_pipeline,
    artifact_path="chatbot",
    input_example="Hi there!"
  )

# Load as interactive pyfunc
chatbot = mlflow.pyfunc.load_model(model_info.model_uri)

# COMMAND ----------

chatbot.predict("What is the best way to get to Antarctica?")


# COMMAND ----------

chatbot.predict("What kind of boat should I use?")

# COMMAND ----------



# COMMAND ----------


