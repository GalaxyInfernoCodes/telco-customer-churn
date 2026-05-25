# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 3 · Task B — Batch inference
# MAGIC 
# MAGIC **Second task in the workflow.** Reads the Silver table that Task A produced,
# MAGIC applies a "model" to score each row, writes predictions to a Gold table.
# MAGIC 
# MAGIC The model here is a simple rule for demo purposes. In production this is where you'd write:
# MAGIC ```python
# MAGIC model = mlflow.pyfunc.load_model("models:/heat_demand_forecast/Production")
# MAGIC predictions = model.predict(features)
# MAGIC ```

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# Read the Silver table that Task A produced
features = spark.read.table("workspace.default.demo_hourly_features")

# Compute a threshold (in production: this comes from your trained model)
avg_count = features.agg(F.avg("trip_count")).first()[0]
print(f"Threshold (avg trip count across hours): {avg_count:.0f}")

# COMMAND ----------

# Score each row — flag busy vs. quiet hours
predictions = (
    features
    .withColumn(
        "predicted_label",
        F.when(F.col("trip_count") > avg_count, "busy").otherwise("quiet")
    )
    .withColumn("scored_at", F.current_timestamp())
)

# Write the Gold (predictions) table
(predictions.write
    .mode("overwrite")
    .saveAsTable("workspace.default.demo_predictions")
)

print(f"Wrote {predictions.count()} predictions to workspace.default.demo_predictions")
display(predictions)
