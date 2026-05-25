# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 1 · MLflow tracking
# MAGIC 
# MAGIC We'll train a classifier twice with different hyperparameters and watch
# MAGIC MLflow capture both runs automatically — no extra `log_param` / `log_metric` calls needed.
# MAGIC 
# MAGIC **What we'll do:**
# MAGIC 1. Load a familiar sklearn dataset
# MAGIC 2. Turn on autologging (one line)
# MAGIC 3. Train two models with different settings
# MAGIC 4. Find the runs in the Experiments UI and compare them

# COMMAND ----------

import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the data
# MAGIC sklearn's breast cancer dataset — 569 rows, 30 features, binary target.

# COMMAND ----------

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42
)
print(f"Training rows: {len(X_train)},  test rows: {len(X_test)}")
print(f"First 5 features: {list(X_train.columns[:5])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Turn on MLflow autologging
# MAGIC One line. From here on, every `.fit()` call is captured.

# COMMAND ----------

mlflow.sklearn.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. First run — small forest

# COMMAND ----------

with mlflow.start_run(run_name="rf_small"):
    model_a = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    model_a.fit(X_train, y_train)

    preds = model_a.predict(X_test)
    print(f"F1:       {f1_score(y_test, preds):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Second run — bigger forest

# COMMAND ----------

with mlflow.start_run(run_name="rf_big"):
    model_b = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model_b.fit(X_train, y_train)

    preds = model_b.predict(X_test)
    print(f"F1:       {f1_score(y_test, preds):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Find the runs
# MAGIC 
# MAGIC Open the **Experiments** panel — flask icon on the right sidebar, or
# MAGIC `View → Experiment Runs` from the menu.
# MAGIC 
# MAGIC You'll see both runs. For each one, autolog captured:
# MAGIC - **Parameters**: `n_estimators`, `max_depth`, every other constructor arg
# MAGIC - **Metrics**: training score, plus the metrics it could compute
# MAGIC - **Artifacts**: the trained model itself, ready to load
# MAGIC - **Environment**: Python version, sklearn version, the notebook source
# MAGIC 
# MAGIC None of this was logged manually. That's the point.
# MAGIC 
# MAGIC ### Try it
# MAGIC Tick both runs in the experiments table → click **Compare** → see them side-by-side.
