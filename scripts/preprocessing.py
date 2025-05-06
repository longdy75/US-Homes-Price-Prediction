
from sklearn.cluster import KMeans
import os
import numpy as np
import pandas as pd
from spark_setup import spark

# Ensure output directory exists
output_dir = "./df_data"
os.makedirs(output_dir, exist_ok=True)

# Read Parquet file
input_path = "./df_data/gold_data.parquet"
df_imputed = spark.read.parquet(input_path)
print(f"Read {df_imputed.count()} rows from {input_path}")

# Convert Spark DataFrame to Pandas
df_pandas = df_imputed.toPandas()
print(f"\nConverted Spark DataFrame to Pandas DataFrame.")

# Cap price at 95th percentile
price_cap = df_pandas['price'].quantile(0.95)
df_pandas['price'] = df_pandas['price'].clip(upper=price_cap)
print(f"\nCapped price at 95th percentile.")

# Log-transform target
df_pandas['log_price'] = np.log1p(df_pandas['price'])
print(f"\nLog-transformed 'price'.")

# Standardize land_space to sqft
df_pandas['land_space_sqft'] = df_pandas.apply(
    lambda row: row['land_space'] * 43560 if row['land_space_unit'] == 'acres'
    else row['land_space'] if row['land_space_unit'] == 'sqft'
    else 0, axis=1
)
print(f"\nStandardized land_space to sqft.")

# Ensure non-negative living_space and land_space_sqft
df_pandas['living_space'] = df_pandas['living_space'].clip(lower=0)
df_pandas['land_space_sqft'] = df_pandas['land_space_sqft'].clip(lower=0)

# Check for negative values
#print("Negative living_space:", (df_pandas['living_space'] < 0).sum())
#print("Negative land_space_sqft:", (df_pandas['land_space_sqft'] < 0).sum())

# Handle zero bedroom_number to avoid division-by-zero
df_pandas['bedroom_number'] = df_pandas['bedroom_number'].replace(0, 1)

# Log-transform skewed features
df_pandas['living_space_log'] = np.log1p(df_pandas['living_space'])
df_pandas['land_space_sqft_log'] = np.log1p(df_pandas['land_space_sqft'])
(f"\nLog-transformed living_space and land_space.")

# Create interaction feature
df_pandas['living_space_per_bedroom'] = df_pandas['living_space'] / df_pandas['bedroom_number']

# Spatial clustering
coords = df_pandas[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=10, random_state=42).fit(coords)
df_pandas['neighborhood_cluster'] = kmeans.labels_
df_pandas = pd.get_dummies(df_pandas, columns=['neighborhood_cluster'], prefix='cluster')
print(f"\nClustered latitude and longitude into 10 Clusters.")

# One-hot encode property_type
df_pandas = pd.get_dummies(df_pandas, columns=['property_type'], prefix='property_type')
print(f"\nOne-hot encoded property_type.")

# One-hot encode state
df_pandas = pd.get_dummies(df_pandas, columns=['state'], prefix='state')
print(f"\nOne-hot encoded state.")

# Write preprocessed data out to pickle
output_path = "./df_data/data.pkl"
df_pandas.to_pickle(output_path)
print(f"Wrote {len(df_pandas)} rows to {output_path}")

# Verify output
loaded_df = pd.read_pickle(output_path)
print(f"Verified {len(loaded_df)} rows in {output_path}")
spark.stop()