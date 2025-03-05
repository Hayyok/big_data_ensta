from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, to_date, rank, from_unixtime, lag, mean, stddev
from pyspark.sql.window import Window
import os



def get_spark() -> SparkSession:
    spark = SparkSession.builder.appName("Reddit").getOrCreate()
    return spark

def read_parquet(input_path: str) -> DataFrame:
    df = spark.read.parquet(input_path)
    return df

def get_frequency(df : DataFrame, column : str) : 
    nb = df.groupBy(column).agg(count(column).alias("count"))
    return nb

def convert_date(df : DataFrame) :
    df = df. withColumn("date", to_date(from_unixtime(col("created_utc"))))
    return df

def write_to_parquet(df : DataFrame, output_path : str):
    df.write.mode("overwrite").parquet(f"{output_path}_dir")
    os.system(f"cat {output_path}_dir/p* > parquet_files/{output_path}")
    os.system(f"rm -R {output_path}_dir")

def write_in_csv(df : DataFrame, file_name : str) : 
    df.write.csv(f"{file_name}_dir", header = True, mode='overwrite')
    os.system(f"cat {file_name}_dir/p* > csv_files/{file_name}")
    os.system(f"rm -R {file_name}_dir")


spark = get_spark()

df_post = read_parquet('db/final_dataframe_with_topics.parquet')
df_cluster = read_parquet('db/cluster_and_topic_names.parquet')
df_subcluster = read_parquet('db/subcluster_and_topic_names.parquet')

# # Calcule le nombre de post par cluster
# cluster_counts = get_frequency(df_post, "cluster")
# cluster_join = cluster_counts.join(df_cluster, "cluster", "left")
# cluster_join = cluster_join.select("cluster", "count", "topic").orderBy(col("count").desc())
# print("Fréquence des clusters:")
# cluster_join.show(5)
# write_in_csv(cluster_join, "cluster_count.csv")
# write_to_parquet(cluster_join, "cluster_count.parquet")

# # Calcule le nombre de post par souscluster
# subcluster_counts = get_frequency(df_post, "subcluster")
# subcluster_join = subcluster_counts.join(df_subcluster, "subcluster", "left")
# subcluster_join = subcluster_join.select("subcluster", "count", "topic_subcluster").orderBy(col("count").desc())
# print("Fréquence des subclusters:")
# subcluster_join.show(10)
# write_in_csv(subcluster_join, "subcluster_count.csv")
# write_to_parquet(subcluster_join, "subcluster_count.parquet")

# # Calcule le nombre de sous-cluster par cluster
# subcluster_in_cluser_count = get_frequency(df_subcluster, "cluster")
# subcluster_in_cluser_join = subcluster_in_cluser_count.join(df_cluster, "cluster", "left")
# subcluster_in_cluser_join = subcluster_in_cluser_join.select("cluster", "topic", "count").orderBy(col("count").desc())
# print("Fréquence des subclusters:")
# subcluster_in_cluser_join.show(10)
# write_in_csv(subcluster_in_cluser_join, "subbycluster.csv")
# write_to_parquet(subcluster_in_cluser_join, "subbycluster.parquet")

# # Calcule le nombre de post par sub reddit
# subreddit_count = get_frequency(df_post, "subreddit")
# subreddit_count = subreddit_count.orderBy(col("count").desc())
# print("Fréquence des subreddit:")
# subreddit_count.show(10)
# write_in_csv(subreddit_count, "subreddit_count.csv")
# write_to_parquet(subreddit_count, "subreddit_count.parquet")

# Etudie l'évolution des clusters les plus populaires en fonction du temps
df_evolution = convert_date(df_post)
cluster_time_evolution = df_evolution.groupBy("date", "cluster").agg(count("cluster").alias("count"))
cluster_time_evolution_join = cluster_time_evolution.join(df_cluster, "cluster", "left")
cluster_time_evolution_join = cluster_time_evolution_join.select("date", "cluster", "topic", "count").orderBy("date", col("count").desc())
window_spec = Window.partitionBy("date").orderBy(col("count").desc())
cluster_daily_top = cluster_time_evolution_join.withColumn("rank", rank().over(window_spec))
cluster_daily_top = cluster_daily_top.filter(col("rank") <= 5).drop("rank")
print("Populatrité des clusters en fonction du temps :")
cluster_daily_top.show(10)
write_in_csv(cluster_daily_top, "cluster_time_evolution.csv")
write_to_parquet(cluster_daily_top, "cluster_time_evolution.parquet")

# # Detection d'événements marquants
# df_evolution = convert_date(df_post)
# cluster_time_evolution = df_evolution.groupBy("date", "cluster").agg(count("cluster").alias("count"))
# cluster_time_evolution_join = cluster_time_evolution.join(df_cluster, "cluster", "left")
# cluster_time_evolution_join = cluster_time_evolution_join.select("date", "cluster", "topic", "count")
# window_spec = Window.partitionBy("cluster").orderBy("date")
# cluster_daily_count = cluster_time_evolution_join.withColumn("prev_day_count", lag("count").over(window_spec))
# cluster_daily_count = cluster_daily_count.withColumn("daily_variation", col("count") - col("prev_day_count"))
# stats = cluster_daily_count.select(mean("daily_variation").alias("mean_var"), stddev("daily_variation").alias("stddev_var")).collect()
# mean_var = stats[0]["mean_var"]
# stddev_var = stats[0]["stddev_var"]
# threshold = mean_var + (3*stddev_var)
# print(mean_var)
# print(threshold)
# anomalies = cluster_daily_count.filter(col("daily_variation") > threshold)
# anomalies = anomalies.orderBy(col("daily_variation").desc())
# print("Evenements marquants : ")
# anomalies.show(10)
# write_in_csv(anomalies, "evenements_marquants.csv")
# write_to_parquet(anomalies, "evenements_marquants.parquet")