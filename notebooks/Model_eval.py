# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.core-0.4.4-py3-none-any.whl --force-reinstall
# MAGIC # %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.eda-0.4.4-py3-none-any.whl
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.model_eval-0.4.4-py3-none-any.whl
# MAGIC
# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install pandas-gbq
# MAGIC %pip install protobuf==3.17.2
# MAGIC %pip install numpy==1.22
# MAGIC

# COMMAND ----------

import sklearn
import mlflow
from mlflow.tracking import MlflowClient
import warnings
import pandas as pd
from tigerml.model_eval import RegressionReport
from utils import utils
warnings.filterwarnings("ignore")

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

model_name = dbutils.widgets.get("model_name")
model_data_path = dbutils.widgets.get("model_data_path")
feature_columns = dbutils.widgets.get("feature_columns").split(",")
target_column = dbutils.widgets.get("target_columns")
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")

# COMMAND ----------

client = MlflowClient()
model_versions = client.get_latest_versions(model_name)
model_version = model_versions[0].version

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

query = f'SELECT * FROM `{model_data_path}`'
train_output_df = pd.read_gbq(query, project_id=project_id, credentials=oauth_credentials)
train_output_df.drop(columns = ["id","date","timestamp"],inplace = True)
train_output_df.display()

# COMMAND ----------

train_df = train_output_df[train_output_df['dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE'] == "train"]
test_df = train_output_df[train_output_df['dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE'] == "test"]

# COMMAND ----------

regOpt1 = RegressionReport(y_train=train_df[target_column], x_train=train_df[feature_columns], x_test=test_df[feature_columns], y_test=test_df[target_column],model=loaded_model)

# COMMAND ----------

utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=regOpt1,
						   plot_name="TigeML_Eval_Report",
						   lib = "tigerml_eval",
						   include_shap = True,
						   folder_name='Model_Evaluation',
						   )
