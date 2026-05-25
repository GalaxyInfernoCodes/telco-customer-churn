# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 2 · Reading and writing Delta tables
# MAGIC 
# MAGIC We'll read a sample table from Unity Catalog, transform it with Spark,
# MAGIC write the result as a Delta table, query it with SQL, and look at its history.
# MAGIC 
# MAGIC **The point:** this is what data living in a platform looks like, vs. a CSV on a laptop.

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read a sample table
# MAGIC `samples.nyctaxi.trips` ships with every Databricks workspace.

# COMMAND ----------

trips = spark.read.table("samples.nyctaxi.trips")
print(f"Rows: {trips.count():,}")
trips.printSchema()

# COMMAND ----------

display(trips.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transform — average fare and distance by pickup zip code
# MAGIC Looks like pandas. Runs as Spark.

# COMMAND ----------

avg_fare = (
    trips
    .filter(F.col("trip_distance") > 0)
    .groupBy("pickup_zip")
    .agg(
        F.count("*").alias("n_trips"),
        F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
    )
    .orderBy(F.desc("n_trips"))
)
display(avg_fare.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write it as a Delta table
# MAGIC Now this lives in the catalog — anyone with permission can read it from any notebook, in any language.

# COMMAND ----------

(avg_fare.write
    .mode("overwrite")
    .saveAsTable("workspace.default.demo_avg_fare_by_zip")
)
print("Written. Refresh Catalog Explorer → workspace → default to see it appear.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Same table, queried with SQL
# MAGIC Python wrote it. SQL reads it. No conversion, no export step.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT pickup_zip, n_trips, avg_fare, avg_distance
# MAGIC FROM workspace.default.demo_avg_fare_by_zip
# MAGIC ORDER BY n_trips DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delta keeps history — for free
# MAGIC Every write creates a new version. You can roll back, or query "what did this table look like yesterday?"

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.default.demo_avg_fare_by_zip

# COMMAND ----------

# MAGIC %md
# MAGIC Try writing the table again — change the filter, re-run cell 3, then re-run this `DESCRIBE HISTORY` cell. You'll see version 1 appear next to version 0. That's not a feature you get with a CSV file.
