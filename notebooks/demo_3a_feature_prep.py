# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 3 · Task A — Feature prep
# MAGIC 
# MAGIC **First task in the workflow.** Reads raw trip data, builds an hourly feature table.
# MAGIC 
# MAGIC In a real pipeline this is the **Silver layer**: cleaned, typed, ready to feed downstream tasks.

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# Read the Bronze (raw) table
trips = spark.read.table("samples.nyctaxi.trips")

# Build hourly features — drop bad rows, group by pickup hour
features = (
    trips
    .filter(F.col("fare_amount") > 0)
    .filter(F.col("trip_distance") > 0)
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .groupBy("pickup_hour")
    .agg(
        F.count("*").alias("trip_count"),
        F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
    )
    .orderBy("pickup_hour")
)

# Write the Silver table
(features.write
    .mode("overwrite")
    .saveAsTable("workspace.default.demo_hourly_features")
)

print(f"Wrote {features.count()} rows to workspace.default.demo_hourly_features")
display(features)
