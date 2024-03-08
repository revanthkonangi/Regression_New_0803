# Databricks notebook source
# MAGIC %md ## DATA Drift TEMPLATE -- EVEDENTLY AI
# MAGIC

# COMMAND ----------

# %pip install google-auth
# %pip install google-cloud-storage
# %pip install azure-storage-blob
# %pip install pandas-gbq
# %pip install protobuf==3.17.2


%pip install evidently
%pip install dataclasses
%pip install pandas==1.3.5
%pip install numpy==1.21.4
%pip install pydantic==1.10.12

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset,DataQualityPreset
#from evidently.metrics import *
from evidently.metrics import DatasetMissingValuesMetric

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *
from utils import utils

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame
import seaborn as sns
from pyspark.sql import functions as F, types as T

# COMMAND ----------

input_table_path = dbutils.widgets.get("input_table_path")

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

adult_ref = pd_df.sample(n=500, replace=False)
adult_cur = pd_df.sample(n=500, replace=False)

# COMMAND ----------

data_drift_report = Report(metrics=[
    DataDriftPreset(),])

data_drift_report.run(reference_data=adult_ref, current_data=adult_cur)
x = data_drift_report.json()
data_drift_report.show(mode='inline')

# COMMAND ----------

utils.push_plots_to_mlcore(dbutils=dbutils,
                               figure_to_save=data_drift_report,
                               plot_name='Data_drift',
                               lib='evedently',
                               folder_name='data_drift',
                               )
