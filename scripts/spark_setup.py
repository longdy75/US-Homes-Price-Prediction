import os
import toml
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Load config from TOML
def load_config(config_path="../config.toml"):
    return toml.load(config_path)

# Set environment variables from config
def setup_enviro(config):
    env = config.get("env", {})
    os.environ["JAVA_HOME"] = env.get("java_home", "")
    os.environ["HADOOP_HOME"] = env.get("hadoop_home", "")
    os.environ["SPARK_HOME"] = env.get("spark_home", "")
    os.environ["HADOOP_USER_NAME"] = env.get("hadoop_user", "")

    os.environ["PATH"] = f"{os.environ['JAVA_HOME']}\\bin;" \
                         f"{os.environ['HADOOP_HOME']}\\bin;" \
                         f"{os.environ['SPARK_HOME']}\\bin;" \
                         f"{os.environ.get('PATH', '')}"

# Stop Spark context (if exists)
def if_exists_context():
    if SparkContext._active_spark_context is not None:
        SparkContext._active_spark_context.stop()

# Initialize Spark session
def init_spark(config):
    spark_config = config.get("spark", {})
    spark_builder = SparkSession.builder \
        .appName(spark_config.get("app_name", "Test")) \
        .master(spark_config.get("master", "local[*]")) \
        .config("spark.executor.memory", spark_config.get("executor_memory", "4g")) \
        .config("spark.driver.memory", spark_config.get("driver_memory", "2g"))

    return spark_builder.getOrCreate()

# Main setup
def setup_spark():
    config = load_config()
    setup_enviro(config)
    if_exists_context()
    spark = init_spark(config)
    print("Spark initialized successfully!")
    return spark

# Run setup
spark = setup_spark()