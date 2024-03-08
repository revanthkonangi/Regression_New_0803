# Databricks notebook source
# MAGIC %md
# MAGIC # Model Inference Notebook
# MAGIC
# MAGIC ## Notebook Purpose
# MAGIC This notebook focuses on utilizing a trained machine learning model for making predictions on new data. The notebook covers the steps from loading data to saving and registering the prediction results.
# MAGIC The main part of this notebook focuses on performing the inference process using the specified model and parameters. This includes:
# MAGIC - Loading the trained model with the specified version.
# MAGIC - Loading the transformed features data.
# MAGIC - Executing predictions on the inference data.
# MAGIC - Storing the inference results in the specified output table.
# MAGIC
# MAGIC ## Notebook Structure
# MAGIC The notebook is organized into the following sections:
# MAGIC
# MAGIC #### 1. Installing Dependencies and Importing Packages
# MAGIC Install the MLCORE SDK and import the required Python packages for this inference process.
# MAGIC
# MAGIC #### 2. Load Configurations
# MAGIC Before proceeding with model training, load the necessary configurations from a YAML file. These configurations should include hyperparameters, file paths, and other essential settings.
# MAGIC
# MAGIC #### 3. Adding Job Run to MLCore
# MAGIC Check the validity of the input and add the inference run to MLCore using `operation_type="job_run_add"` if it's valid.
# MAGIC
# MAGIC #### 4. Loading Inference Features Data
# MAGIC Load the new data for which predictions are to be made. This data should have the same format as the data used for training. Apply the specified start and end markers to the `features_df` DataFrame to filter the data within the desired time range for retraining. The start and end markers define the time period for which the data will be used in the retraining process.
# MAGIC
# MAGIC #### 5. Loading the Model
# MAGIC Load the trained machine learning model that you intend to use for predictions.
# MAGIC
# MAGIC #### 6. Making Predictions and Output Table
# MAGIC Apply the loaded model to the inference features data to make predictions. Create an output table containing the predictions.
# MAGIC
# MAGIC #### 7. Saving Output Predictions to Hive
# MAGIC Store the output predictions table in a Hive table. This facilitates further analysis and comparison.
# MAGIC
# MAGIC #### 8. Initializing MLCORE SDK and Registering Inference Artifacts
# MAGIC Initialize the `mlclient` from the MLCORE SDK. Use the `log` function with `operation_type="register_inference"` to register the inference artifacts in MLCore.
# MAGIC
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC - This notebook demonstrates the process of using a trained model for inference on new data.
# MAGIC - User inputs, data loading, prediction, and registration steps are covered.
# MAGIC - Leveraging the MLCORE SDK and Hive database enhances the management and tracking of inference artifacts.

# COMMAND ----------

# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Imports
from MLCORE_SDK import mlclient
import ast
from pyspark.sql import functions as F
from datetime import datetime
from delta.tables import *
import time
import pandas as pd
import mlflow
import pickle

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

import yaml
from utils import utils

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### General Parameters
# MAGIC
# MAGIC We begin by loading general parameters from the SolutionConfig YAML file:
# MAGIC
# MAGIC - `sdk_session_id`: This parameter is obtained using the `env` variable to create a dynamic key.
# MAGIC - `env`: The environment is directly extracted from the SolutionConfig YAML file.
# MAGIC - `db_name`: The database name is also sourced from the SolutionConfig YAML file.
# MAGIC - `use_latest`: A boolean flag indicating whether to use the latest model version. If set to `True`, the inference process will use the latest model version available.
# MAGIC
# MAGIC ### Job-Specific Parameters for Inference
# MAGIC
# MAGIC In this section, we configure job-specific parameters for the inference process:
# MAGIC
# MAGIC - `transformed_features_path`: The path to the transformed features data for inference.
# MAGIC - `ground_truth_path`: The path to the ground truth data (if available) for comparison.
# MAGIC - `output_table_name`: The name of the output table where inference results will be stored.
# MAGIC - `batch_size`: The batch size for processing inference data, with a default value of 500.
# MAGIC - `cron_job_schedule`: The cron job schedule for executing inference jobs, with a default value of "0 */10 * ? * *".
# MAGIC - `model_name`: The name of the machine learning model to be used for inference.
# MAGIC - `model_version`: The version of the machine learning model to be used for inference.
# MAGIC - `feature_columns`: The columns that represent features in the inference data.
# MAGIC - `date_column`: The date column is crucial for time-series inference.
# MAGIC
# MAGIC ### Model Version Selection
# MAGIC
# MAGIC The `use_latest` flag and the specified `model_version` parameter control which model version is used for inference. You can choose to use the latest version or a specific version depending on your requirements.
# MAGIC
# MAGIC #####This structured inference notebook streamlines the process of executing machine learning model predictions on new data, with flexibility in model version selection and scheduling.

# COMMAND ----------

# DBTITLE 1,Configs
# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']
use_latest = True

# JOB SPECIFIC PARAMETERS FOR INFERENCE
transformed_features_path = solution_config["inference"]["transformed_features_path"]
ground_truth_path = solution_config["inference"]["ground_truth_path"]
output_table_name = solution_config["inference"]["output_table_name"]
batch_size = int(solution_config["inference"].get("batch_size",500))
cron_job_schedule = solution_config["inference"].get("cron_job_schedule","0 */10 * ? * *")
model_name = solution_config["inference"]["model_name"]
model_version = solution_config["inference"]["model_version"]


primary_keys = solution_config["train"]["primary_keys"]
features = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]

# COMMAND ----------

# DBTITLE 1,Select the latest trained model
if use_latest:
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    x = client.get_latest_versions(model_name)
    model_version = x[0].version

# COMMAND ----------

# DBTITLE 1,Add the run to ML Core if it is valid.
from MLCORE_SDK import mlclient
mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = "inference")

# COMMAND ----------

pickle_file_path = f"/mnt/FileStore/{db_name}"
dbutils.fs.mkdirs(pickle_file_path)
print(f"Created directory : {pickle_file_path}")
pickle_file_path = f"/dbfs/{pickle_file_path}/{output_table_name}.pickle"

# COMMAND ----------

try : 
  with open(pickle_file_path, "rb") as handle:
      obj_properties = pickle.load(handle)
      print(f"Instance loaded successfully")
except Exception as e:
  print(f"Exception while loading cache : {e}")
  obj_properties = {}
print(f"Existing Cache : {obj_properties}")

# COMMAND ----------

if not obj_properties :
  start_marker = 1
elif obj_properties and obj_properties.get("end_marker",0) == 0:
  start_marker = 1
else :
  start_marker = obj_properties["end_marker"] + 1
end_marker = start_marker + batch_size - 1

print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Load Inference Features Data

# COMMAND ----------

_,vault_scope = utils.get_env_vault_scope(dbutils)

encrypted_sa_details = dbutils.secrets.get(vault_scope,"gcp-service-account-encypted")
encryption_key = dbutils.secrets.get(vault_scope,"gcp-service-account-private-key")
bq_database_name = dbutils.secrets.get(vault_scope,"gcp-bq-database-name")
gcp_project_id = dbutils.secrets.get(vault_scope,"gcp-api-quota-project-id")
az_container_name = str(dbutils.secrets.get(scope=vault_scope, key='az-container-name'))

features_dbfs_path = f"{transformed_features_path}_{env}_{sdk_session_id}"
ground_truth_path = f"{ground_truth_path}_{env}_{sdk_session_id}"

features_df = utils.df_read(
            spark = spark,
            data_path=features_dbfs_path,
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type="bigquery")

gt_data = utils.df_read(
            spark = spark,
            data_path=ground_truth_path,
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type="bigquery")

# COMMAND ----------

features_df.count()

# COMMAND ----------

features_df.display()

# COMMAND ----------

# DBTITLE 1,Apply the start and end marker to features_df
FT_DF = features_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

FT_DF.display()

# COMMAND ----------

# DBTITLE 1,Exit the notebook if no data available
if not FT_DF.first():
  dbutils.notebook.exit("No data is available for inference, hence exiting the notebook")

# COMMAND ----------

ground_truth = gt_data.toPandas()[primary_keys + target_columns]
tranformed_features_df = FT_DF.toPandas()
tranformed_features_df.dropna(inplace=True)
tranformed_features_df.shape

# COMMAND ----------

inference_df = tranformed_features_df[features]
display(inference_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Model

# COMMAND ----------

# DBTITLE 1,Loading the pre trained model
loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

predictions = loaded_model.predict(inference_df)
type(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Output Table

# COMMAND ----------

tranformed_features_df

# COMMAND ----------

tranformed_features_df["prediction"] = predictions
tranformed_features_df = pd.merge(tranformed_features_df,ground_truth, on=primary_keys[0], how='inner')
output_table = spark.createDataFrame(tranformed_features_df)

# COMMAND ----------

output_table.display()

# COMMAND ----------

output_table = output_table.drop('date','timestamp')

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

from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_table = output_table.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_table = output_table.withColumn("date", F.lit(date))
output_table = output_table.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in output_table.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  output_table = output_table.withColumn("id", F.row_number().over(window))

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_job_helper import get_job_id_run_id

job_id, run_id = get_job_id_run_id(dbutils)

output_table = output_table.withColumnRenamed(target_columns[0],"ground_truth_value")
output_table = output_table.withColumn("acceptance_status",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_time",F.lit(None).cast("long"))
output_table = output_table.withColumn("accepted_by_id",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_by_name",F.lit(None).cast("string"))
output_table = output_table.withColumn("moderated_value",F.lit(None).cast("double"))
output_table = output_table.withColumn("inference_job_id",F.lit(job_id).cast("string"))
output_table = output_table.withColumn("inference_run_id",F.lit(run_id).cast("string"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Save Output Table

# COMMAND ----------

output_table_name = f"{output_table_name}_{env}_{sdk_session_id}"

utils.df_write(
    data_path=output_table_name,
    dataframe = output_table,
    mode = "append",
    bucket_name=f"{az_container_name}_{env}",
    bq_database_name=bq_database_name,
    bq_project_id=gcp_project_id,
    encrypted_service_account=encrypted_sa_details,
    encryption_key=encryption_key,
    resource_type="bigquery")

print(f"Big Query path : {gcp_project_id}.{bq_database_name}.{output_table_name}",)

# COMMAND ----------

stagemetrics.end()

# COMMAND ----------

stagemetrics.print_report()

# COMMAND ----------

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory").collect()[0].asDict()

# COMMAND ----------

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# DBTITLE 1,Register Inference artifacts in MLCore
mlclient.log(operation_type = "register_inference",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    output_table_name=output_table_name,
    output_table_path=f"{gcp_project_id}.{bq_database_name}.{output_table_name}",
    feature_table_path=f"{gcp_project_id}.{bq_database_name}.{features_dbfs_path}",
    ground_truth_table_path=f"{gcp_project_id}.{bq_database_name}.{ground_truth_path}",
    model_name=model_name,
    model_version=model_version,
    num_rows=output_table.count(),
    cols=output_table.columns,
    table_type="bigquery",
    batch_size = str(batch_size),
    env = env,
    column_datatype = output_table.dtypes,
    table_schema = output_table.schema,
    verbose = True,
    compute_usage_metrics = compute_metrics)

# COMMAND ----------

# DBTITLE 1,Saving the end marker
obj_properties['end_marker'] = end_marker
with open(pickle_file_path, "wb") as handle:
    pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Instance successfully saved successfully")
