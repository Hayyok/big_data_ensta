from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, to_date, rank, from_unixtime, lag, mean, stddev
from pyspark.sql.window import Window
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import os
import pandas as pd
import numpy as np


# Initialisation de la session Spark
def get_spark() -> SparkSession:
    spark = SparkSession.builder.appName("Reddit").getOrCreate()
    return spark

# Lecture d'un fichier Parquet que l'on charge dans un DataFrame
def read_parquet(input_path: str) -> DataFrame:
    df = spark.read.parquet(input_path)
    return df

# Calcul de la fréquence d'apparition des valeurs d'une colonne donnée
def get_frequency(df : DataFrame, column : str) : 
    nb = df.groupBy(column).agg(count(column).alias("count"))
    return nb

# Convertion des dates en format UNIX (de la colonne created_utc) en un format lisible
def convert_date(df : DataFrame) :
    df = df. withColumn("date", to_date(from_unixtime(col("created_utc"))))
    return df

# Ecriture d'un dataframe en Parquet
def write_to_parquet(df : DataFrame, output_path : str):
    df.write.mode("overwrite").parquet(f"{output_path}_dir")
    os.system(f"cat {output_path}_dir/p* > parquet_files/{output_path}")
    os.system(f"rm -R {output_path}_dir")

# Ecriture d'un dataframe en csv
def write_in_csv(df : DataFrame, file_name : str) : 
    df.write.csv(f"{file_name}_dir", header = True, mode='overwrite')
    os.system(f"cat {file_name}_dir/p* > csv_files/{file_name}")
    os.system(f"rm -R {file_name}_dir")

# Initialisation de la session Spark
spark = get_spark()


# Chargement des fichiers de données
df_post = read_parquet('db/final_dataframe_with_topics.parquet')
df_cluster = read_parquet('db/cluster_and_topic_names.parquet')
df_subcluster = read_parquet('db/subcluster_and_topic_names.parquet')


# Calcule le nombre de posts par cluster
def nb_post_by_cluster():
    cluster_counts = get_frequency(df_post, "cluster")
    cluster_join = cluster_counts.join(df_cluster, "cluster", "left")
    cluster_join = cluster_join.select("cluster", "count", "topic").orderBy(col("count").desc())
    print("Fréquence des clusters:")
    cluster_join.show(5)
    # Sauvegarde en csv et Parquet
    write_in_csv(cluster_join, "cluster_count.csv")
    write_to_parquet(cluster_join, "cluster_count.parquet")


# Calcule le nombre de posts par sous-cluster
def nb_post_by_subcluster():
    subcluster_counts = get_frequency(df_post, "subcluster")
    subcluster_join = subcluster_counts.join(df_subcluster, "subcluster", "left")
    subcluster_join = subcluster_join.select("subcluster", "count", "topic_subcluster").orderBy(col("count").desc())
    print("Fréquence des subclusters:")
    subcluster_join.show(10)
    # Sauvegarde en csv et Parquet
    write_in_csv(subcluster_join, "subcluster_count.csv")
    write_to_parquet(subcluster_join, "subcluster_count.parquet")


# Calcule le nombre de sous-cluster par cluster
def nb_subcluster_by_cluster():
    subcluster_in_cluser_count = get_frequency(df_subcluster, "cluster")
    subcluster_in_cluser_join = subcluster_in_cluser_count.join(df_cluster, "cluster", "left")
    subcluster_in_cluser_join = subcluster_in_cluser_join.select("cluster", "topic", "count").orderBy(col("count").desc())
    print("Fréquence des subclusters:")
    subcluster_in_cluser_join.show(10)
    # Sauvegarde en csv et Parquet
    write_in_csv(subcluster_in_cluser_join, "subbycluster.csv")
    write_to_parquet(subcluster_in_cluser_join, "subbycluster.parquet")


# Calcule le nombre de posts par subreddit
def post_by_sub():
    subreddit_count = get_frequency(df_post, "subreddit")
    subreddit_count = subreddit_count.orderBy(col("count").desc())
    print("Fréquence des subreddit:")
    subreddit_count.show(10)
    # Sauvegarde en csv et Parquet
    write_in_csv(subreddit_count, "subreddit_count.csv")
    write_to_parquet(subreddit_count, "subreddit_count.parquet")


# Etudie l'évolution des clusters les plus populaires en fonction du temps
def cluster_evolution():
    df_evolution = convert_date(df_post) # Convertit la date des posts
    cluster_time_evolution = df_evolution.groupBy("date", "cluster").agg(count("cluster").alias("count")) # Compte les posts par jour et cluster
    cluster_time_evolution_join = cluster_time_evolution.join(df_cluster, "cluster", "left") # Jointure avec les noms des clusters
    cluster_time_evolution_join = cluster_time_evolution_join.select("date", "cluster", "topic", "count").orderBy("date", col("count").desc()) # Sélection et tri
    # Définition d'une fenêtre pour classer les clusters les plus populaires chaque jour
    window_spec = Window.partitionBy("date").orderBy(col("count").desc())
    # Classement des clusters par popularité quotidienne
    cluster_daily_top = cluster_time_evolution_join.withColumn("rank", rank().over(window_spec))
    cluster_daily_top = cluster_daily_top.filter(col("rank") <= 5).drop("rank") # Ne garde que les 5 plus populaires par jour
    print("Populatrité des clusters en fonction du temps :")
    cluster_daily_top.show(10)
    # Sauvegarde en csv et Parquet
    write_in_csv(cluster_daily_top, "cluster_time_evolution.csv")
    write_to_parquet(cluster_daily_top, "cluster_time_evolution.parquet")

# Detection d'événements marquants
def big_event():
    df_evolution = convert_date(df_post)
    cluster_time_evolution = df_evolution.groupBy("date", "cluster").agg(count("cluster").alias("count"))
    cluster_time_evolution_join = cluster_time_evolution.join(df_cluster, "cluster", "left")
    cluster_time_evolution_join = cluster_time_evolution_join.select("date", "cluster", "topic", "count")
    window_spec = Window.partitionBy("cluster").orderBy("date")
    cluster_daily_count = cluster_time_evolution_join.withColumn("prev_day_count", lag("count").over(window_spec))
    cluster_daily_count = cluster_daily_count.withColumn("daily_variation", col("count") - col("prev_day_count"))
    # Calcule la moyenne et l'écart type des variations journalières
    stats = cluster_daily_count.select(mean("daily_variation").alias("mean_var"), stddev("daily_variation").alias("stddev_var")).collect()
    mean_var = stats[0]["mean_var"]
    stddev_var = stats[0]["stddev_var"]
    threshold = mean_var + (3*stddev_var)
    print(mean_var)
    print(threshold)
    # Filtrage des anomalies (variations supérieures au seuil)
    anomalies = cluster_daily_count.filter(col("daily_variation") > threshold)
    anomalies = anomalies.orderBy(col("daily_variation").desc())
    print("Evenements marquants : ")
    anomalies.show(10)
    # Sauvegarde en csv et Parquet
    write_in_csv(anomalies, "evenements_marquants.csv")
    write_to_parquet(anomalies, "evenements_marquants.parquet")

# Calcule la corrélation des clusters
def correlation_clusters():
    cluster_dic = df_cluster.select("cluster", "topic").rdd.collectAsMap() # Crée un dictionnaire de correspondance entre les cluster et leur nom
    df_evolution = convert_date(df_post) # Convertit la date des posts
    cluster_time_evolution = df_evolution.groupBy("date", "cluster").agg(count("cluster").alias("count")) # Compte les posts par jour et cluster
    # On crée une matrice où chaque ligne est une date et chaque colonne un cluster
    pivot_df = cluster_time_evolution.groupBy("date").pivot("cluster").sum("count").fillna(0)
    # On convertit les données pour Spark ML (1 seule colonne avec toutes les valeurs numériques dans un vecteur)
    feature_cols = [col for col in pivot_df.columns if col != "date"] # On retire la colonne date
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features") # On combine tout les ids de clusters d'une même date en 1 vecteur
    df_vector = assembler.transform(pivot_df).select("features") 
    # Calcul de la matrice de corrélation
    corr_matrix = Correlation.corr(df_vector, "features").head()[0]
    corr_array = np.array(corr_matrix.toArray()) # on la transforme en matrcie numpy
    cluster_ids = feature_cols
    cluster_names = [f"{cluster_dic[int(c)]} (id {c})" for c in cluster_ids] # On remplace les numéros des clusters par leur nom (on garde l'id parce qu'il semble y avoir des doublons de nom)
    corr_df = pd.DataFrame(corr_array, index=cluster_names, columns=cluster_names) # On crée un df pandas avec la matrice de corrélation
    corr_unstacked = corr_df.unstack().reset_index()
    corr_unstacked.columns = ["Cluster 1", "Cluster 2", "Correlation"]
    corr_unstacked = corr_unstacked[corr_unstacked["Cluster 1"] < corr_unstacked["Cluster 2"]] # On supprime la diagonale et les doublons 
    corr_unstacked_sorted = corr_unstacked.sort_values(by="Correlation", ascending=False) # Trie par corrélation décroissante
    print("Top 10 des clusters les plus corrélés avec leurs noms :")
    print(corr_unstacked_sorted.head(10))
    corr_unstacked_sorted.to_csv("csv_files/clusters_correlation.csv", index=False) # On sauvegarde le fichier en format csv
    corr_unstacked_sorted.to_parquet("parquet_files/clusters_correlation.parquet", index=False) # On sauvegarde le fichier en format parquet


nb_post_by_cluster()
nb_post_by_subcluster()
nb_subcluster_by_cluster()
post_by_sub()
cluster_evolution()
big_event()
correlation_clusters()