import os
from pyspark.sql.functions import col, when, count, round, avg
from spark_setup import spark


# Ensure output directory exists
output_dir = "./df_data"
os.makedirs(output_dir, exist_ok=True)

# Read parquet file
input_path = "./df_data/bronze_data.parquet"
df = spark.read.parquet(input_path)
print(f"Read {df.count()} rows from {input_path}")


# ----- Feature Selection -----

# Dropping columns that are not useful or contain > 25% NULL values
# Dropping price_per_unit because it is poorly aggregated
df_dropped = df.drop('property_url', 'property_id', 'address', 'street_name', 'apartment', 'broker_id', 'listing_age', 'year_build', 'total_num_units', 'agent_name', 'agent_phone', 'price_per_unit', 'agency_name', 'RunDate', 'property_status', 'is_owned_by_zillow')

# Drop Multi-Family & lot units to focus on houses/condos/apartments/manufactured
df_dropped = df_dropped.filter(~df_dropped["property_type"].isin(["MULTI_FAMILY", "LOT"]))

print(f"\nTotal Rows & Columns After Feature Selection:")
print((df_dropped.count(), len(df_dropped.columns)))
print(f"\n\nCleaning Data...\n")


# ----- Data Cleaning ----- ( Silver Layer )

# --- Cleaning Target Variable: Price ---

# Drop null prices
df_dropped = df_dropped.dropna(subset=["price"])

# Cast price to an integer
df_dropped = df_dropped.withColumn("price", col("price").cast("int"))

# Define price bounds
price_upper_bound = df_dropped.approxQuantile("price", [0.95], 0.01)[0]

# Define the lower bound
price_lower_bound = 5000

# Filter the DataFrame to remove rows outside the bounds
df_filtered = df_dropped.filter((df_dropped["price"] >= price_lower_bound) & (df_dropped["price"] <= price_upper_bound))



# ---------- Handling Missing Values ----------

# Drop the rows with the missing city and missing postcode
df_filtered = df_filtered.filter(df_filtered["city"].isNotNull() & df_filtered["postcode"].isNotNull() & df_filtered["state"].isNotNull())

# --- Impute latitude and longitude based on postcode ---
postcode_lat_lng = df_filtered.groupBy("postcode").agg(
    avg("latitude").alias("avg_latitude"),
    avg("longitude").alias("avg_longitude")
)

# Join back to df_filtered to impute
df_imputed = df_filtered.join(postcode_lat_lng, "postcode", "left")

df_imputed = df_imputed.withColumn(
    "latitude", when(col("latitude").isNull(), col("avg_latitude")).otherwise(col("latitude")))

df_imputed = df_imputed.withColumn(
    "longitude", when(col("longitude").isNull(), col("avg_longitude")).otherwise(col("longitude")))

# Drop temporary columns
df_imputed = df_imputed.drop("avg_latitude", "avg_longitude")

# Drop lat & long rows we can't impute
df_imputed = df_imputed.filter(df_imputed["latitude"].isNotNull() & df_imputed["longitude"].isNotNull())


# --- Impute bedroom number ---
# Compute median bedroom_number for each living_space range
df_avg_bedrooms = df_imputed.groupBy("living_space").agg(
    round(avg("bedroom_number")).alias("avg_bedroom_number")
)

# Join the imputation data back to df_imputed
df_imputed = df_imputed.join(df_avg_bedrooms, "living_space", "left")

# Impute missing bedroom_number values using avg_bedroom_number
df_imputed = df_imputed.withColumn(
    "bedroom_number",
    when(col("bedroom_number").isNull(), col("avg_bedroom_number")).otherwise(col("bedroom_number"))
)

# Drop temporary column used for imputation
df_imputed = df_imputed.drop("avg_bedroom_number")

# Drop rows remaining rows where bedroom_number & bathroom_number are null
df_imputed = df_imputed.dropna(subset=["bedroom_number", "bathroom_number"])

# --- Handle land_space nulls for condo/townhouse/apartment/manufactured ---

df_imputed = df_imputed.withColumn(
    "land_space",
    when((col("property_type").isin(["CONDO", "TOWNHOUSE", "APARTMENT", "MANUFACTURED"])) & col("land_space").isNull(), "0")
    .otherwise(col("land_space"))
)

df_imputed = df_imputed.withColumn(
    "land_space_unit",
    when((col("property_type").isin(["CONDO", "TOWNHOUSE", "APARTMENT", "MANUFACTURED"])) & col("land_space_unit").isNull(), "N/A")
    .otherwise(col("land_space_unit"))
)

# Fill remaining land_space and land_space_unit nulls with Unknown since it's not given, but worth keeping the rows
df_imputed = df_imputed.fillna({"land_space": 0})
df_imputed = df_imputed.fillna({"land_space_unit": "Unknown"})

# --- Impute remaining living_space nulls based on avg. # of bedrooms & bathrooms ---

df_avg_living_space = df_imputed.groupBy("bedroom_number", "bathroom_number").agg(
    avg("living_space").alias("avg_living_space")
)

df_imputed = df_imputed.join(df_avg_living_space, ["bedroom_number", "bathroom_number"], "left")

df_imputed = df_imputed.withColumn(
    "living_space",
    when(col("living_space").isNull(), col("avg_living_space")).otherwise(col("living_space"))
)

# Drop temporary column after imputation
df_imputed = df_imputed.drop("avg_living_space")

# Drop remaining living_space with nulls
df_imputed = df_imputed.dropna(subset=["living_space"])

# Make sure dtypes are correct
df_imputed = df_imputed.withColumn("bedroom_number", col("bedroom_number").cast("int"))
df_imputed = df_imputed.withColumn("bathroom_number", col("bathroom_number").cast("int"))
df_imputed = df_imputed.withColumn("living_space", col("living_space").cast("int"))
df_imputed = df_imputed.withColumn("land_space", col("land_space").cast("float"))
df_imputed = df_imputed.withColumn("postcode", col("postcode").cast("int"))

# Drop remaining postcodes with nulls
df_imputed = df_imputed.dropna(subset=["postcode"])


# ---------- Normalization ----------

# Compute Q1 and Q3 for each column 
q1_bath, q3_bath = df_imputed.approxQuantile("bathroom_number", [0.25, 0.75], 0.01)
q1_bed, q3_bed = df_imputed.approxQuantile("bedroom_number", [0.25, 0.75], 0.01)
q1_living, q3_living = df_imputed.approxQuantile("living_space", [0.25, 0.75], 0.01)

# Compute IQR for each column
iqr_bath = q3_bath - q1_bath
iqr_bed = q3_bed - q1_bed
iqr_living = q3_living - q1_living

# Define bounds for each
lower_bound_bath = q1_bath - (1.5 * iqr_bath)
upper_bound_bath = q3_bath + (1.5 * iqr_bath)

lower_bound_bed = q1_bed - (1.5 * iqr_bed)
upper_bound_bed = q3_bed + (1.5 * iqr_bed)

lower_bound_living = q1_living - (1.5 * iqr_living)
upper_bound_living = q3_living + (1.5 * iqr_living)

# Remove outliers
df_imputed = df_imputed.filter(
    (col("bathroom_number") >= lower_bound_bath) & (col("bathroom_number") <= upper_bound_bath) &
    (col("bedroom_number") >= lower_bound_bed) & (col("bedroom_number") <= upper_bound_bed) &
    (col("living_space") >= lower_bound_living) & (col("living_space") <= upper_bound_living)
)

print(f"\nRemaining Nulls:\n")
df_imputed.select([count(when(col(c).isNull(), c)).alias(c) for c in df_imputed.columns]).show()
print(f"\nSaving Cleaned Data to Parquet...\n")

# Write data to a new parquet file
output_path = "./df_data/silver_data.parquet"
df_imputed.write.mode("overwrite").parquet(output_path)
print(f"\nWrote {df_imputed.count()} rows to {output_path}\n")