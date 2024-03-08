# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.core-0.4.4-py3-none-any.whl --force-reinstall
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.eda-0.4.4-py3-none-any.whl
# MAGIC %pip install numpy==1.22

# COMMAND ----------

# MAGIC %md ## EDA Python
# MAGIC

# COMMAND ----------

from tigerml.eda import EDAReport
from utils import utils

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

tigermleda = EDAReport(pd_df)

# COMMAND ----------

utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=tigermleda,
						   plot_name="Tiger_ML_EDA",
						   lib='tigerml',
						   ext='html',
						   folder_name='custom_reports',
						   )
