# Databricks notebook source
# MAGIC %md ## DATA QUALITY with Great Expectations and Evedently

# COMMAND ----------

# MAGIC %md <b>Imports
# MAGIC
# MAGIC Along with the imports required for the notebook to execute custom transformations, we have to import <b>MLCoreClient</b> from <b>MLCORE_SDK</b>, which provides helper methods to integrate the custom notebook with rest of the Data Prep or Data Prep Deployment flow.

# COMMAND ----------

# DBTITLE 1,Install packages
# %pip install google-auth
# %pip install google-cloud-storage
# %pip install azure-storage-blob
# %pip install pandas-gbq
# %pip install protobuf==3.17.2
# %pip install numpy==1.22

%pip install evidently
%pip install great-expectations
%pip install dataclasses
%pip install pandas==1.3.5
%pip install --force-reinstall typing-extensions==4.5.0

# COMMAND ----------

# DBTITLE 1,Imports
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset,DataQualityPreset
from evidently.metrics import *
from evidently.tests.base_test import generate_column_tests
import json
from functools import reduce
from pyspark.sql import DataFrame
import seaborn as sns
from pyspark.sql import functions as F, types as T
import pandas as pd
from delta.tables import DeltaTable
from datetime import datetime
from pyspark.sql import functions as F, DataFrame, Window
import requests
from requests.structures import CaseInsensitiveDict
import json   
import uuid
import calendar
import re
import time
from datetime import datetime
from pyspark.sql.functions import *
import numpy as np
from pyspark.sql.window import Window
from joblib import Parallel, delayed
from pyspark.sql.types import *
from delta.tables import *
from pyspark.sql import functions as F
from pyspark.sql import types as T
from tenacity import retry, stop_after_delay, stop_after_attempt, wait_exponential
import traceback
import uuid
import pandas as pd
print(pd.__version__)
import great_expectations as gx
from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler
from great_expectations.dataset.sparkdf_dataset import SparkDFDataset
from great_expectations.render.renderer import *
from great_expectations.render.view import DefaultJinjaPageView
from great_expectations.expectations.expectation import ExpectationConfiguration
import json
from utils import utils

# COMMAND ----------

input_table_path = dbutils.widgets.get("input_table_path")

# COMMAND ----------

data = """
{
  "expectation_configs": [
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "num_refill_req_l3m", "min_value": 0, "max_value": 8}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "num_refill_req_l3m"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "num_refill_req_l3m", "min_value": 6, "max_value": 12}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "num_refill_req_l3m", "min_value": 0, "max_value": 3}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "num_refill_req_l3m", "min_value": 3, "max_value": 5}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "num_refill_req_l3m", "min_value": 3, "max_value": 5}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "transport_issue_l1y", "min_value": 0, "max_value": 7}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "transport_issue_l1y"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "transport_issue_l1y", "min_value": 4, "max_value": 7}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "transport_issue_l1y", "min_value": 0, "max_value": 3}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "transport_issue_l1y", "min_value": 0, "max_value": 1}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "transport_issue_l1y", "min_value": 0, "max_value": 1}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "Competitor_in_mkt", "min_value": 0, "max_value": 12}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "Competitor_in_mkt"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "Competitor_in_mkt", "min_value": 8, "max_value": 13}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "Competitor_in_mkt", "min_value": 0, "max_value": 2}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "Competitor_in_mkt", "min_value": 2, "max_value": 4}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "Competitor_in_mkt", "min_value": 2, "max_value": 4}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "retail_shop_num", "min_value": 1500, "max_value": 12000}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "retail_shop_num"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "retail_shop_num", "min_value": 10500, "max_value": 12000}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "retail_shop_num", "min_value": 1500, "max_value": 2000}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "retail_shop_num", "min_value": 4500, "max_value": 5500}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "retail_shop_num", "min_value": 4500, "max_value": 5500}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "distributor_num", "min_value": 10, "max_value": 75}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "distributor_num"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "distributor_num", "min_value": 60, "max_value": 70}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "distributor_num", "min_value": 5, "max_value": 20}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "distributor_num", "min_value": 35, "max_value": 50}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "distributor_num", "min_value": 35, "max_value": 50}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "dist_from_hub", "min_value": 45, "max_value": 285}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "dist_from_hub"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "dist_from_hub", "min_value": 260, "max_value": 285}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "dist_from_hub", "min_value":45, "max_value": 65}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "dist_from_hub", "min_value": 150, "max_value": 180}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "dist_from_hub", "min_value": 150, "max_value": 180}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "workers_num", "min_value": 0, "max_value": 120}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "workers_num"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "workers_num", "min_value": 85, "max_value": 120}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "workers_num", "min_value": 0, "max_value": 15}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "workers_num", "min_value": 15, "max_value": 40}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "workers_num", "min_value": 15, "max_value": 40}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "storage_issue_reported_l3m", "min_value": 0, "max_value": 50}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "storage_issue_reported_l3m"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "storage_issue_reported_l3m", "min_value": 30, "max_value": 50}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "storage_issue_reported_l3m", "min_value": 0, "max_value": 5}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "storage_issue_reported_l3m", "min_value": 10, "max_value": 30}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "storage_issue_reported_l3m", "min_value": 10, "max_value": 30}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "wh_breakdown_l3m", "min_value": 0, "max_value": 15}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "wh_breakdown_l3m"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "wh_breakdown_l3m", "min_value": 3, "max_value": 9}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "wh_breakdown_l3m", "min_value": 0, "max_value": 3}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "wh_breakdown_l3m", "min_value": 2, "max_value": 5}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "wh_breakdown_l3m", "min_value": 2, "max_value": 5}},
    {"expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "govt_check_l3m", "min_value": 1, "max_value": 40}},
    {"expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "govt_check_l3m"}},
    {"expectation_type": "expect_column_max_to_be_between",
      "kwargs": {"column": "govt_check_l3m", "min_value": 25, "max_value": 45}},
    {"expectation_type": "expect_column_min_to_be_between",
      "kwargs": {"column": "govt_check_l3m", "min_value": 0, "max_value": 5}},
    {"expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {"column": "govt_check_l3m", "min_value": 10, "max_value": 25}},
    {"expectation_type": "expect_column_median_to_be_between",
      "kwargs": {"column": "govt_check_l3m", "min_value": 10, "max_value": 25}}
  ]
}
"""

# COMMAND ----------

# MAGIC %md <b> Sample Data

# COMMAND ----------

_,vault_scope = utils.get_env_vault_scope(dbutils)

encrypted_sa_details = dbutils.secrets.get(vault_scope,"gcp-service-account-encypted")
encryption_key = dbutils.secrets.get(vault_scope,"gcp-service-account-private-key")
bq_database_name = dbutils.secrets.get(vault_scope,"gcp-bq-database-name")
gcp_project_id = dbutils.secrets.get(vault_scope,"gcp-api-quota-project-id")
az_container_name = str(dbutils.secrets.get(scope=vault_scope, key='az-container-name'))

df = utils.df_read(
            spark = spark,
            data_path=input_table_path,
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type="bigquery")

# COMMAND ----------

pd_df = df.toPandas()
pd_df.drop(columns = ["id","date","timestamp"],inplace = True)
pd_df.display()

# COMMAND ----------

adult_ref = pd_df.sample(n=len(pd_df), replace=False)
adult_cur = pd_df.sample(n=len(pd_df), replace=False)
data_quality_report = Report(metrics=[
    DataQualityPreset(),])
    

# COMMAND ----------

# MAGIC %md <b> Great Expectations function to get data quality report

# COMMAND ----------

context = gx.get_context()
df_gx = df
validator = SparkDFDataset(df_gx)

# COMMAND ----------

data = json.loads(data)

# COMMAND ----------

expectation_configs = data['expectation_configs']

# COMMAND ----------

for config in expectation_configs:
    expectation_type = config["expectation_type"]
    kwargs = config["kwargs"]
    getattr(validator, expectation_type)(**kwargs)
validation_result = validator.validate()

# COMMAND ----------

document_model = ValidationResultsPageRenderer().render(validation_result)
html_content = DefaultJinjaPageView().render(document_model)
displayHTML(html_content)

# COMMAND ----------

utils.push_plots_to_mlcore(dbutils=dbutils,
                        figure_to_save=html_content,
                        plot_name="DQ_index",
                        lib='gx',
                        folder_name='data_quality',
                        )
