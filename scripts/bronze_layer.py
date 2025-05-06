from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from spark_setup import spark


# ----- Spark data ingestion ( Bronze Layer ) -----
file_path = "../dataset/us_houses.csv"
df = spark.read.csv(file_path, header=True)

# Verify DataFrame
row_count, col_count = df.count(), len(df.columns)
print(f"Input CSV: {row_count} rows, {col_count} columns")
if row_count == 0:
    raise ValueError("Input CSV is empty or contains no valid rows!")
df.printSchema()
df.show(5)

# Write to parquet
output_path = "./df_data/bronze_data.parquet"
df.write.mode("overwrite").parquet(output_path)
print(f"Wrote {row_count} rows to {output_path}")

# Verify parquet file
df_parquet = spark.read.parquet(output_path)
parquet_row_count = df_parquet.count()
print(f"Parquet file: {parquet_row_count} rows, {len(df_parquet.columns)} columns")