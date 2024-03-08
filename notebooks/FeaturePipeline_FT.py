# Databricks notebook source
# DBTITLE 1,Uncomment in case you want to register the data in feature store.
# %pip install databricks-feature-store 

# COMMAND ----------

# dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

try : 
    env = dbutils.widgets.get("env")
    task = dbutils.widgets.get("task")
except :
    env, task = "dev","fe"
print(f"Input environment : {env}")
print(f"Input task : {task}")

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
from utils import utils
from MLCORE_SDK import mlclient

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# JOB SPECIFIC PARAMETERS FOR FEATURE PIPELINE
if task.lower() == "fe":
    features_dbfs_path = solution_config["feature_pipelines"]["feature_pipelines_ft"]["features_dbfs_path"]
    transformed_features_table_name = solution_config["feature_pipelines"]["feature_pipelines_ft"]["transformed_features_table_name"]
    is_scheduled = solution_config["feature_pipelines"]["feature_pipelines_ft"]["is_scheduled"]
    batch_size = int(solution_config["feature_pipelines"]["feature_pipelines_ft"].get("batch_size",500))
    cron_job_schedule = solution_config["feature_pipelines"]["feature_pipelines_ft"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["feature_pipelines"]["feature_pipelines_ft"]["primary_keys"]
else:
    # JOB SPECIFIC PARAMETERS FOR DATA PREP DEPLOYMENT
    features_dbfs_path = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["features_dbfs_path"]
    transformed_features_table_name = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["transformed_features_table_name"]
    is_scheduled = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["is_scheduled"]
    batch_size = int(solution_config["data_prep_deployments"]["data_prep_deployment_ft"].get("batch_size",500))
    cron_job_schedule = solution_config["data_prep_deployments"]["data_prep_deployment_ft"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["primary_keys"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### FEATURE ENGINEERING

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### FEATURE ENGINEERING on Feature Data

# COMMAND ----------

# DBTITLE 1,Load the data
_,vault_scope = utils.get_env_vault_scope(dbutils)

encrypted_sa_details = dbutils.secrets.get(vault_scope,"gcp-service-account-encypted")
encryption_key = dbutils.secrets.get(vault_scope,"gcp-service-account-private-key")
bq_database_name = dbutils.secrets.get(vault_scope,"gcp-bq-database-name")
gcp_project_id = dbutils.secrets.get(vault_scope,"gcp-api-quota-project-id")
az_container_name = str(dbutils.secrets.get(scope=vault_scope, key='az-container-name'))

features_df = utils.df_read(
            spark = spark,
            data_path=f"{features_dbfs_path}_{env}_{sdk_session_id}",
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type="bigquery")

# COMMAND ----------

features_df.display()

# COMMAND ----------

from pyspark.sql import functions as F
import pickle

# COMMAND ----------

if is_scheduled:
  pickle_file_path = f"/mnt/FileStore/{db_name}"
  dbutils.fs.mkdirs(pickle_file_path)
  print(f"Created directory : {pickle_file_path}")
  pickle_file_path = f"/dbfs/{pickle_file_path}/{transformed_features_table_name}.pickle"

  try : 
    with open(pickle_file_path, "rb") as handle:
        obj_properties = pickle.load(handle)
        print(f"Instance loaded successfully")
  except Exception as e:
    print(f"Exception while loading cache : {e}")
    obj_properties = {}
  print(f"Existing Cache : {obj_properties}")

  if not obj_properties :
    start_marker = 1
  elif obj_properties and obj_properties.get("end_marker",0) == 0:
    start_marker = 1
  else :
    start_marker = obj_properties["end_marker"] + 1
  end_marker = start_marker + batch_size - 1
else :
  start_marker = 1
  end_marker = features_df.count()

  print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

# DBTITLE 1,Perform some feature engineering step. 
FT_DF = features_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

# DBTITLE 1,Exit the job if there is no new data
if not FT_DF.first():
  dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

if task.lower() != "fe":
    mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = task)

# COMMAND ----------

FT_DF = FT_DF.drop('date','timestamp')

# COMMAND ----------

FT_DF.display()

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
FT_DF = FT_DF.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
FT_DF = FT_DF.withColumn("date", F.lit(date))
FT_DF = FT_DF.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in FT_DF.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  FT_DF = FT_DF.withColumn("id", F.row_number().over(window))

# COMMAND ----------

transformed_features_table_name = f"{transformed_features_table_name}_{env}_{sdk_session_id}"


utils.df_write(
    data_path=transformed_features_table_name,
    dataframe = FT_DF,
    mode = "append",
    bucket_name=f"{az_container_name}_{env}",
    bq_database_name=bq_database_name,
    bq_project_id=gcp_project_id,
    encrypted_service_account=encrypted_sa_details,
    encryption_key=encryption_key,
    resource_type="bigquery")

print(f"Big Query path : {gcp_project_id}.{bq_database_name}.{transformed_features_table_name}",)

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
# MAGIC
# MAGIC ### REGISTER THE FEATURES ON MLCORE
# MAGIC

# COMMAND ----------

# DBTITLE 1,Register Features Transformed Table
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = transformed_features_table_name,
    num_rows = FT_DF.count(),
    cols = FT_DF.columns,
    column_datatype = FT_DF.dtypes,
    table_schema = FT_DF.schema,
    primary_keys = primary_keys,
    table_path = f"{gcp_project_id}.{bq_database_name}.{transformed_features_table_name}",
    table_type="bigquery",
    table_sub_type="Source",
    request_type = task,
    env = env,
    batch_size = batch_size,
    quartz_cron_expression = cron_job_schedule,
    verbose = True,
    compute_usage_metrics = compute_metrics)

# COMMAND ----------

if is_scheduled:
  obj_properties['end_marker'] = end_marker
  with open(pickle_file_path, "wb") as handle:
      pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Instance successfully saved successfully")

# COMMAND ----------

# from MLCORE_SDK.sdk.manage_sdk_session import get_session
# from MLCORE_SDK.helpers.auth_helper import get_container_name
# from MLCORE_SDK.helpers.mlc_job_helper import get_job_id_run_id

# res = get_session(sdk_session_id,dbutils=dbutils)
# project = res.json()["data"]["project_id"]
# version = res.json()["data"]["version"]

# az_container_name = get_container_name(dbutils)
# job_id, run_id = get_job_id_run_id(dbutils)
# job_run_str = str(job_id) + "/" + str(run_id)

# if not env:
#     env = dbutils.widgets.get("env")

# media_artifacts_path = f"/mnt/{az_container_name}/{env}/media_artifacts/{project}/{version}/{job_run_str}"

# COMMAND ----------

# try :
    
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Tiger_ML_Python_EDA", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#             "input_table_path" : transformed_features_table_name,
#             "media_artifacts_path" : media_artifacts_path,
#             })
# except Exception as e:
#     print(f"Exception while triggering EDA notebook : {e}")

# COMMAND ----------

# try :
    
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Anomaly_Detection", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#             "input_table_path" : transformed_features_table_name,
#             "media_artifacts_path" : media_artifacts_path,
#             "Target_column": "distributor_num"
#             })
# except Exception as e:
#     print(f"Exception while triggering Anomaly_Detection notebook : {e}")

# COMMAND ----------

# try :
    
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Data_Drift", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#             "input_table_path" : transformed_features_table_name,
#             "media_artifacts_path" : media_artifacts_path,
#             })
# except Exception as e:
#     print(f"Exception while triggering Data_Drift notebook : {e}")

# COMMAND ----------

# try :
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Data_Quality", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#             "input_table_path" :transformed_features_table_name,
#             "media_artifacts_path" : media_artifacts_path,
#             })
# except Exception as e:
#     print(f"Exception while triggering Data_Quality notebook : {e}")
