from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count


def get_spark() -> SparkSession:
    spark = SparkSession.builder.appName("Reddit").getOrCreate()
    return spark

def read_parquet(input_path: str) -> DataFrame:
    df = spark.read.parquet(input_path)
    return df

def get_frequency(df : DataFrame, column : str) : 
    nb = df.groupBy(column).agg(count(column).alias("count"))
    return nb

def write_to_parquet(df : DataFrame, output_path : str):
    df.write.mode("overwrite").parquet(output_path)

def write_in_csv(df : DataFrame, file_name : str) : 
    df.write.csv(file_name, mode='overwrite')


spark = get_spark()

df_post = read_parquet('final_dataframe_with_topics.parquet')
df_cluster = read_parquet('cluster_and_topic_names.parquet')
df_subcluster = read_parquet('subcluster_and_topic_names.parquet')

cluster_counts = get_frequency(df_post, "cluster")
subcluster_counts = get_frequency(df_post, "subcluster")

cluster_join = cluster_counts.join(df_cluster, "cluster", "left")
subcluster_join = subcluster_counts.join(df_subcluster, "subcluster", "left")

cluster_join = cluster_join.select("cluster", "count", "topic").orderBy(col("count").desc())
subcluster_join = subcluster_join.select("subcluster", "count", "topic_subcluster").orderBy(col("count").desc())

print("Fréquence des clusters:")
cluster_join.show(5)
subcluster_counts.show(10)

write_in_csv(cluster_join, "cluster_count.csv")
write_in_csv(subcluster_join, "subcluster_count.csv")
