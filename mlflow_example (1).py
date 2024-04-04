# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC Import a dataset of Airbnb listings and featurize the data.  We'll use this to train a model.

# COMMAND ----------

import boto3
import pandas as pd

# COMMAND ----------

s3 = boto3.client('s3')

# COMMAND ----------

bucket = "columbia-gr5069-main"
airbnb_data = "raw/airbnb/airbnb-cleaned-mlflow.csv"

obj_laps = s3.get_object(Bucket= bucket, Key= airbnb_data) 
df = pd.read_csv(obj_laps['Body'])

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Perform a train/test split.

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Log a basic experiment by doing the following:<br><br>
# MAGIC
# MAGIC 1. Start an experiment using `mlflow.start_run()` and passing it a name for the run
# MAGIC 2. Train your model
# MAGIC 3. Log the model using `mlflow.sklearn.log_model()`
# MAGIC 4. Log the model error using `mlflow.log_metric()`
# MAGIC 5. Print out the run id using `run.info.run_uuid`

# COMMAND ----------

pip install mlflow

# COMMAND ----------


import mlflow

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Basic RF Experiment") as run:
  # Create model, train it, and create predictions
  rf = RandomForestRegressor()
  rf.fit(X_train, y_train)
  predictions = rf.predict(X_test)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
  print("  mse: {}".format(mse))
  
  # Log metrics
  mlflow.log_metric("mse", mse)
  
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Parameters, Metrics, and Artifacts
# MAGIC
# MAGIC But wait, there's more!  In the last example, you logged the run name, an evaluation metric, and your model itself as an artifact.  Now let's log parameters, multiple metrics, and other artifacts including the feature importances.
# MAGIC
# MAGIC First, create a function to perform this.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> To log artifacts, we have to save them somewhere before MLflow can log them.  This code accomplishes that by using a temporary file that it then deletes.

# COMMAND ----------



# COMMAND ----------

def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  import os
  import matplotlib.pyplot as plt
  import mlflow.sklearn
  import seaborn as sns
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import tempfile

  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]

    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("  mse: {}".format(mse))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)  
    mlflow.log_metric("r2", r2)  
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file
    
    # Create plot
    fig, ax = plt.subplots()

    # sns.residplot(predictions, y_test, lowess=True)
    # plt.xlabel("Predicted values for Price ($)")
    # plt.ylabel("Residual")
    # plt.title("Residual Plot")

    # # Log residuals using a temporary file
    # temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
    # temp_name = temp.name
    # try:
    #   fig.savefig(temp_name)
    #   mlflow.log_artifact(temp_name, "residuals.png")
    # finally:
    #   temp.close() # Delete the temp file
      
    # display(fig)
    return run.info.run_uuid

# COMMAND ----------

# MAGIC %md
# MAGIC Run with new parameters.

# COMMAND ----------

params = {
  "n_estimators": 100,
  "max_depth": 5,
  "random_state": 42
}

log_rf(experimentID, "Second Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Check the UI to see how this appears.  Take a look at the artifact to see where the plot was saved.
# MAGIC
# MAGIC Now, run a third run.

# COMMAND ----------

params_1000_trees = {
  "n_estimators": 1000,
  "max_depth": 10,
  "random_state": 42
}

log_rf(experimentID, "Third Run", params_1000_trees, X_train, X_test, y_train, y_test)

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, current_date
from pyspark.sql.types import IntegerType 
from pyspark.sql.functions import col, avg, sum, count, when, isnan, lit, round
from pyspark.sql.functions import col, upper, when, substring, regexp_replace, translate


# COMMAND ----------

df = spark.read.csv('s3://columbia-gr5069-main/raw/pit_stops.csv', header = True)

# COMMAND ----------

pip install mlflow


# COMMAND ----------

import mlflow

# COMMAND ----------

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# 假设 df 是您的原始 DataFrame
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')  # 将 duration 转换为数值型，非数值转为 NaN

# 处理缺失值
df.dropna(subset=['duration'], inplace=True)  # 移除 duration 中的缺失值

# 对分类变量进行独热编码
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['raceId', 'driverId', 'stop']])

# 将编码后的特征转换为 DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(['raceId', 'driverId', 'stop']))

# 准备 X 和 y
X = encoded_df
y = df['duration']

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

from pyspark.sql.functions import regexp_extract, col
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# 查找和替换 '\\N' 值
X_train = X_train.replace('\\N', np.nan)
y_train = y_train.replace('\\N', np.nan)

X_test = X_test.replace('\\N', np.nan)
y_test = y_test.replace('\\N', np.nan)
# 处理缺失值
# 选项 1: 移除包含缺失值的行
X_train = X_train.dropna()
y_train = y_train.dropna()

X_test = X_test.dropna()
y_test = y_test.dropna()
# 选择 X_train 中的数值型列并转换为 float 类型
X_train_numeric = X_train.select_dtypes(include=[np.number])
X_train = X_train_numeric.astype(float)

X_test_numeric = X_test.select_dtypes(include=[np.number])
X_test = X_test_numeric.astype(float)
# 将 y_train 转换为 float 类型
y_train = y_train.astype(float)
y_test = y_test.astype(float)



# COMMAND ----------



# COMMAND ----------

def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  import os
  import matplotlib.pyplot as plt
  import mlflow.sklearn
  import seaborn as sns
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import tempfile
  import pandas as pd

  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]

    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("  mse: {}".format(mse))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Create feature importance
    importance = pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)),
                              columns=["Feature", "Importance"]
                             ).sort_values("Importance", ascending=False)

    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv", delete=False)
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close()  # Delete the temp file

    # Create plot for residuals
    fig, ax = plt.subplots()
    sns.residplot(x=predictions, y=y_test - predictions, lowess=True)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png", delete=False)
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals.png")
    finally:
      temp.close()  # Delete the temp file

    return run.info.run_uuid


# COMMAND ----------

# 定义参数组合
experiment_params = [
    {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    {"n_estimators": 150, "max_depth": 5, "random_state": 42},
    {"n_estimators": 200, "max_depth": 5, "random_state": 42},
    {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    {"n_estimators": 100, "max_depth": 15, "random_state": 42},
    {"n_estimators": 150, "max_depth": 10, "random_state": 42},
    {"n_estimators": 150, "max_depth": 15, "random_state": 42},
    {"n_estimators": 200, "max_depth": 10, "random_state": 42},
    {"n_estimators": 200, "max_depth": 15, "random_state": 42},
    {"n_estimators": 100, "max_depth": None, "random_state": 42}
]


# 循环运行实验
for i, params in enumerate(experiment_params, start=1):
    run_name = f"Experiment {i}"
    print(f"Running {run_name} with params: {params}")
    log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test)

