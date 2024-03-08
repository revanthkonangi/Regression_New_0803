# Databricks notebook source
# MAGIC %md ## Anomaly Detection
# MAGIC

# COMMAND ----------

# %pip install google-auth
# %pip install google-cloud-storage
# %pip install azure-storage-blob
# %pip install pandas-gbq
# %pip install protobuf==3.17.2
# %pip install numpy==1.22
# %pip install pandas==1.0.5

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from utils import utils

# COMMAND ----------

input_table_path = dbutils.widgets.get("input_table_path")
Target_column = dbutils.widgets.get("Target_column")
Target_column

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

random_state = np.random.RandomState(42)
model=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=random_state)
model.fit(pd_df[[Target_column]])
print(model.get_params())

# COMMAND ----------

pd_df['scores'] = model.decision_function(pd_df[[Target_column]])
pd_df['anomaly_score'] = model.predict(pd_df[[Target_column]])
pd_df['anomaly_score'] = pd_df['anomaly_score'].replace({-1: 1, 1: 0})
pd_df.display()

# COMMAND ----------

import plotly.express as px

# COMMAND ----------

pd_df = pd_df[:100]

# COMMAND ----------

fig = px.scatter(pd_df, x='index', y=Target_column, color='anomaly_score',
                     hover_name='index',
                     color_continuous_scale=['blue', 'red'],  # Customize colors
                     labels={Target_column: 'Predicted_Target','anomaly_score': 'Anomalies'},
                     title='Anomalies in Predictions')
    
# Add a line plot connecting the points
line_trace = px.line(pd_df, x='index', y=Target_column, line_shape='linear')
fig.add_trace(line_trace.data[0])

# Customize marker symbols for anomalies
fig.update_traces(marker=dict(size=20), selector=dict(mode='markers+text'))

# Remove the color bar
fig.update_layout(coloraxis_showscale=False)

# Show the plot
fig.show()

# COMMAND ----------

utils.push_plots_to_mlcore(dbutils=dbutils,
                               figure_to_save=fig,
                               plot_name="Anamoly_detection",
                               lib='plotly',
                               ext='html',
                               folder_name='anomaly_detection',
                               )
