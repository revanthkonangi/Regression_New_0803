# Databricks notebook source
# MAGIC %md
# MAGIC # Data Engineering for Feature Table
# MAGIC
# MAGIC ## Notebook Purpose
# MAGIC This notebook focuses on performing data engineering tasks on the feature table, including reading data, creating/updating tables, and registering the table using MLCORE_SDK.
# MAGIC
# MAGIC ## Notebook Structure
# MAGIC The notebook is organized into the following sections:
# MAGIC
# MAGIC #### 1. Configuration
# MAGIC Before running the notebook, make sure to set up the `SolutionConfig` YAML file with the necessary parameters. This includes adjusting the file path and specifying any relevant settings. You no longer need to take user input directly; the configuration is read from the YAML file.
# MAGIC
# MAGIC #### 2. Reading and Database Setup
# MAGIC We read the features DataFrame from the DBFS path and ensure that the Hive database is created if it doesn't already exist. The Hive database is crucial for managing structured data.
# MAGIC
# MAGIC #### 3. Table Creation/Update
# MAGIC In this section, we create or update the features table within the Hive database. This table stores the processed data, making it accessible for further analysis and usage.
# MAGIC
# MAGIC #### 4. Initializing MLCORE_SDK
# MAGIC We initialize the `mlclient` from the MLCORE_SDK. This SDK provides essential functionality for managing and interacting with machine learning assets.
# MAGIC
# MAGIC #### 5. Registering the Table
# MAGIC We register the features table using the `log` function from the MLCORE_SDK. This step is essential for tracking and managing the table's lifecycle. We use the `operation_type = "register_table"` parameter to specify the registration action.
# MAGIC
# MAGIC #### Data Set
# MAGIC The data set is an example of `Pharma Sales`.
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC - Data engineering is a critical step in the data analysis process, ensuring that the data is in a usable format.
# MAGIC - The notebook follows a structured approach: data loading, feature table creation, and table registration.
# MAGIC - Registering the table in the UI streamlines further analysis and collaboration within the Databricks environment.

# COMMAND ----------

# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Input Environment
try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# DBTITLE 1,Load the Yaml Config
import yaml
from utils import utils

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configurations
# MAGIC
# MAGIC This notebook focuses on data engineering tasks, including the extraction of essential parameters and configurations for processing data.
# MAGIC
# MAGIC #### General Parameters
# MAGIC
# MAGIC We begin by loading general parameters from the SolutionConfig YAML file:
# MAGIC
# MAGIC - `sdk_session_id`: This parameter is obtained using the `env` variable to create a dynamic key.
# MAGIC - `env`: The environment is directly extracted from the SolutionConfig YAML file.
# MAGIC - `db_name`: The database name is also sourced from the SolutionConfig YAML file.
# MAGIC
# MAGIC #### Data Engineering Specific Parameters
# MAGIC
# MAGIC In this section, we configure parameters specific to data engineering:
# MAGIC
# MAGIC - `primary_keys`: These are the primary keys for the feature table, essential for data management and integrity.
# MAGIC - `features_table_name`: The name of the features table where processed data will be stored.
# MAGIC - `features_dbfs_path`: The DBFS (Databricks File System) path to the source data that needs to be processed.
# MAGIC
# MAGIC ##### This structured data engineering notebook streamlines the process of configuring and executing data engineering tasks, making it easier to manage and process data within the Databricks environment.

# COMMAND ----------

# DBTITLE 1,Configs
# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# DE SPECIFIC PARAMETERS
primary_keys = solution_config["data_engineering"]["data_engineering_ft"]["primary_keys"]
features_table_name = solution_config["data_engineering"]["data_engineering_ft"]["features_table_name"]
features_dbfs_path = solution_config["data_engineering"]["data_engineering_ft"]["features_dbfs_path"]

# COMMAND ----------

from MLCORE_SDK import mlclient
mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = "DE")

# COMMAND ----------

# DBTITLE 1,Load the features_df from the path
features_df = spark.read.load(features_dbfs_path)

# COMMAND ----------

features_df.display()

# COMMAND ----------

features_df = features_df.drop('date','id','timestamp')

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# DBTITLE 1,ADD A MONOTONICALLY INREASING COLUMN - "id"
from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
features_df = features_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
features_df = features_df.withColumn("date", F.lit(date))
features_df = features_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in features_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  features_df = features_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

# MAGIC %md <b> Write Table to Big Query

# COMMAND ----------

_,vault_scope = utils.get_env_vault_scope(dbutils)

encrypted_sa_details = dbutils.secrets.get(vault_scope,"gcp-service-account-encypted")
encryption_key = dbutils.secrets.get(vault_scope,"gcp-service-account-private-key")
bq_database_name = dbutils.secrets.get(vault_scope,"gcp-bq-database-name")
gcp_project_id = dbutils.secrets.get(vault_scope,"gcp-api-quota-project-id")
az_container_name = str(dbutils.secrets.get(scope=vault_scope, key='az-container-name'))

features_table_name = f"{features_table_name}_{env}_{sdk_session_id}"

utils.df_write(
    data_path=features_table_name,
    dataframe = features_df,
    mode = "overwrite",
    bucket_name=f"{az_container_name}_{env}",
    bq_database_name=bq_database_name,
    bq_project_id=gcp_project_id,
    encrypted_service_account=encrypted_sa_details,
    encryption_key=encryption_key,
    resource_type="bigquery")

# COMMAND ----------

stagemetrics.end()

# COMMAND ----------

stagemetrics.print_report()

# COMMAND ----------

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

# COMMAND ----------

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use MLCore SDK to register Features Tables

# COMMAND ----------

# DBTITLE 1,Import ML Client
from MLCORE_SDK import mlclient

# COMMAND ----------

# DBTITLE 1,Register features table
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = features_table_name,
    num_rows = features_df.count(),
    cols = features_df.columns,
    column_datatype = features_df.dtypes,
    table_schema = features_df.schema,
    primary_keys = primary_keys,
    table_path = f"{gcp_project_id}.{bq_database_name}.{features_table_name}",
    table_type="bigquery",
    table_sub_type="Source",
    env = "dev",
    verbose = True,
    compute_usage_metrics = compute_metrics)
