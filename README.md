# Big Data Tools and Techniques - Final Project
## Contributors: Dylan Long

## To run the scripts on the data, the data must be downloaded from my Drive or from the link below due to GitHub file size restrictions:
### https://drive.google.com/file/d/1cDupbjVcnwEgioxcnJOGyKVetTSlacB9/view?usp=drive_link

## To view the data output from the scripts or to view the model output checkout this link:
### https://drive.google.com/drive/folders/1P4D7-7oDGPlaG8kHD83IMW5fToR-r77_?usp=drive_link

### This is a repository consisting of a public US housing dataset from Kaggle that can be found here: https://www.kaggle.com/datasets/polartech/500000-us-homes-data-for-sale-properties

### The enclosed scripts utilizes Apaches Spark to store, clean, and transform the data for ML modeling with Pandas.

#### Aggregated Features:
    - price_per_land_space_unit (Check unit column to identify sqft/acre) (Standardized during preprocessing)
    - price_per_sqft_living_space
    - price_per_bedroom
    - avg_price_by_postcode
    - avg_price_by_city
    - avg_price_by_state

#### Initial modeling with Linear Regression performed poorly indicating that relationships are non-linear and complicated.
#### Switched to HistGradientBoosting for better performance.

#### Modeling Outcomes (Target Variable: Price):
    - HistGradientBoosting: RMSE=111264.38, R²=0.7614
    - Random Forest: RMSE=91537.18, R²=0.8385
    - Tuned XGBoost: RMSE=103103.31, R²=0.7951
    - Ensemble: RMSE=93930.66, R²=0.8300

## Requirements:
    - Read requirements.txt

## How To Setup:
    - Run spark_setup.py to establish spark session
    - Run the scripts in the following order:
        - bronze_layer.py
        - silver_layer.py
        - gold_layer.py
        - preprocessing.py
        - models.py
