import os
from pyspark.sql.functions import col, when, count, round, avg, percentile_approx, coalesce
from spark_setup import spark

# Ensure output directory exists
output_dir = "./df_data"
os.makedirs(output_dir, exist_ok=True)

# Read parquet file
input_path = "./df_data/silver_data.parquet"
df_imputed = spark.read.parquet(input_path)
print(f"Read {df_imputed.count()} rows from {input_path}")
print(f"\nBeginning Aggregation...\n")

# --- Aggregations ( Gold Layer ) ---

# Price per Land Space Unit
df_imputed = df_imputed.withColumn("price_per_land_space_unit", when(col("land_space") != 0, col("price") / col("land_space")).otherwise(0))

# Price per SqFt. of Living Space
df_imputed = df_imputed.withColumn("price_per_sqft_living_space", when(col("living_space") != 0, col("price") / col("living_space")).otherwise(0))

# Price per Bedroom
df_imputed = df_imputed.withColumn("price_per_bedroom", col("price") / col("bedroom_number"))

# Average Price by Postcode
df_price_by_postcode = df_imputed.groupBy("postcode").agg(avg("price").alias("avg_price_by_postcode"))

# Average Price by City
df_price_by_city = df_imputed.groupBy("city").agg(avg("price").alias("avg_price_by_city"))

# Average Price by State
df_price_by_state = df_imputed.groupBy("state").agg(avg("price").alias("avg_price_by_state"))


# Merge back to DataFrame
df_imputed = df_imputed.join(df_price_by_postcode, "postcode", "left") \
                       .join(df_price_by_city, "city", "left")\
                       .join(df_price_by_state, "state", "left")

print(f"\nFinished Aggregation.\n")
df_imputed.show(10)



# --- Handle leftover nulls from aggregation ---
print(f"Filling Possible Null Values from Aggregating...\n")
# Impute price_per_land_space_unit to handle nulls
df_land_price_median_by_type = df_imputed.groupBy("property_type").agg(
    percentile_approx("price_per_land_space_unit", 0.5).alias("median_price_per_sqft_type")
)

df_imputed = df_imputed.join(df_land_price_median_by_type, "property_type", "left") \
                       .withColumn("price_per_land_space_unit", coalesce(col("price_per_land_space_unit"), col("median_price_per_sqft_type")))

df_imputed = df_imputed.drop("median_price_per_sqft_type")

# Drop null price_per_sqft_living_space & price_per_bedroom rows (< 2%)
df_imputed = df_imputed.dropna(subset=["price_per_sqft_living_space"])
df_imputed = df_imputed.dropna(subset=["price_per_bedroom"])

print(f"\nRemaining Nulls: ")
df_imputed.select([count(when(col(c).isNull(), c)).alias(c) for c in df_imputed.columns]).show()
print(f"\nRows & Columns:")
print((df_imputed.count(), len(df_imputed.columns)))

print(f"\nFirst Few Rows with Aggregation:")
print(df_imputed.show(5))

# Write cleaned data to a new Parquet file
output_path = "./df_data/gold_data.parquet"
df_imputed.write.mode("overwrite").parquet(output_path)
print(f"\nWrote {df_imputed.count()} rows to {output_path}\n")