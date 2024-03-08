# Databricks notebook source
# MAGIC %md
# MAGIC #### Install MLCORE SDK

# COMMAND ----------

# %pip install /dbfs/FileStore/sdk/dev/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install Deep Checks, MLFlow, Pandas and Numpy to specific version

# COMMAND ----------

# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install pandas-gbq
# MAGIC %pip install protobuf==3.17.2
# MAGIC
# MAGIC # %pip install databricks-feature-store
# MAGIC %pip install deepchecks
# MAGIC # %pip install mlflow
# MAGIC %pip install pandas==1.0.5
# MAGIC %pip install numpy==1.19.1
# MAGIC %pip install matplotlib==3.3.2

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import libraries

# COMMAND ----------

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, model_evaluation
from io import StringIO
from pyspark.sql import functions as F
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

import json,time

import pandas as pd
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.cloud import bigquery
from google.oauth2.credentials import Credentials



credentials_path = "/dbfs/FileStore/application_cred_3.json"
project_id = "mlcore-gcp"
dataset_id = "mlcore_test"



with open(credentials_path, "r") as f:
    credentials_data = json.load(f)

oauth_credentials = Credentials.from_authorized_user_info(credentials_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read inputs parameters provided to the notebook.

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
model_data_path = dbutils.widgets.get("model_data_path")
feature_columns = dbutils.widgets.get("feature_columns").split(",")
target_columns = dbutils.widgets.get("target_columns").split(",")
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read the train data

# COMMAND ----------

feature_columns

# COMMAND ----------

target_columns

# COMMAND ----------

query = f'SELECT * FROM `{model_data_path}`'
data_to_check = pd.read_gbq(query, project_id=project_id, credentials=oauth_credentials)
data_to_check.drop(columns = ["id","date","timestamp"],inplace = True)
data_to_check = spark.createDataFrame(data_to_check)
data_to_check.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fetch the Train and Test data and select the categorical columns

# COMMAND ----------

trainDF = data_to_check.filter(F.col("dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE") == "train").select(feature_columns + target_columns)
testDF = data_to_check.filter(F.col("dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE") == "test").select(feature_columns + target_columns)

# COMMAND ----------

def detect_categorical_cols(df, threshold=5):
    """
    Get the Categorical columns with greater than threshold percentage of unique values.

    This function returns the Categorical columns with the unique values in the column
    greater than the threshold percentage.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
    threshold : int , default = 5
        threshold value in percentage

    Returns
    -------
    report_data : dict
        dictionary containing the Numeric column data.

    """
    df = df.toPandas()
    no_of_rows = df.shape[0]
    possible_cat_cols = (
        df.convert_dtypes()
        .select_dtypes(exclude=[np.datetime64, "float", "float64"])
        .columns.values.tolist()
    )
    temp_series = df[possible_cat_cols].apply(
        lambda col: (len(col.unique()) / no_of_rows) * 100 > threshold
    )
    cat_cols = temp_series[temp_series == False].index.tolist()
    return cat_cols

# COMMAND ----------

categorial_columns = detect_categorical_cols(trainDF.select(feature_columns))

# COMMAND ----------

pd_train = trainDF.toPandas()
pd_test = testDF.toPandas()

ds_train = Dataset(pd_train, label=target_columns[0], cat_features=categorial_columns)
ds_test = Dataset(pd_test, label=target_columns[0], cat_features=categorial_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Perform Data Integrity Test on Train and Test data.

# COMMAND ----------

train_res = data_integrity().run(ds_train)
test_res = data_integrity().run(ds_test)

# COMMAND ----------

from utils import utils

# COMMAND ----------

# DBTITLE 1,Show the report for train data
utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=train_res,
						   plot_name="Train_DeepCheck_Report",
						   lib='deepchecks',
						   folder_name='Test_Validation',
						   )

# COMMAND ----------

# DBTITLE 1,Show the report for test data
utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=test_res,
						   plot_name="Test_DeepCheck_Report",
						   lib='deepchecks',
						   folder_name='Test_Validation',
						   )
