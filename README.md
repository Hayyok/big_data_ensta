# Etude de Popularité des Sujets sur Reddit

## Description du Projet

### Contexte
Reddit est la plateforme de discussions la plus populaire, où des millions d'utilisateurs partagent et discutent sur divers sujets chaque jour. L'analyse des tendances peut fournir des informations intéressantes sur l'intérêt des internautes et l'évolution des sujets populaires au cours du temps.

### Objectifs 
Les discussions Reddit sont massives et non structurées, notre objectif est d'identifier les sujets populaires et leur évolution dans le temps. 
Plus particulièrement, nous allons : 
- étudier la dynamique des sujets sur Reddit
- détecter les tendances émergentes
- comprendre les évènements marquants en fonction des tendances

## Choix Techniques

### Structure du dataset
Le dataset utilisé est un dataset Kaggle optimisé pour une analyse structurée de 43 millions de messages Reddit : [Reddit Topic Dataset](https://www.kaggle.com/datasets/stefano1283/reddit-topic-dataset). 
Lors de la création du dataset (non réalisée par nous), l'objectif était de catégoriser les messages textuels non structurés en sujets significatifs à l'aide du modèle BERTopic. Ce processus comprenait une intégration des messages avec BERT, une réduction de la dimensionnalité avec UMAP, le clustering grâce à HDBSCAN, la représentation des sujets avec c-TF-IDF et l'attribution de noms lisibles par l'homme par chatGPT.

Le jeu de données est constitué de quatre fichiers parquet : 
- **`cluster_and_topic_names.parquet`** : Contient les numéros de cluster associés avec un sujet et une liste de mots clés.
- **`final_dataframe_with_topics.parquet`** : Contient les posts Reddit (id, subreddit, date de publication, auteur, sujet) avec leur classification en cluster et sous-cluster.
- **`subcluster_and_topic_names.parquet`** : Contient les descriptions des sous-clusters.
- **`title_selftext_dataframe.parquet`** : Contient les titres et contenus des posts Reddit.

### Technologies utilisées
- **Parquet** : pour le stockage des données
- **PySpark** : pour le traitement des grands volumes de données
- **Pandas et NumPy (Python)** : pour la manipulation et l'analyse des données

### Autres outils utilisés
- **DuckDB** : nous avons utilisé DuckDB pour pouvoir manipuler les parquet plus rapidement en dehors du programme, comme par exemple pour obtenir rapidement les premières lignes du fichier afin de vérifier les résultats ou des connaître le type des données. L'ouverture avec python et spark était étrangement beaucoup plus longue. 

## Méthodologie détaillée

Le traitement des données s'effectue en plusieurs étapes.

### 1. Initialisation de la session Spark
Nous utilisons Spark pour traiter les fichiers volumineux et effectuer des transformations sur nos données. Il faut donc commencer par initialiser la session Spark :
```
def get_spark() -> SparkSession:
    spark = SparkSession.builder.appName("Reddit").getOrCreate()
    return spark

spark = get_spark()
```

### 2. Chargement des données
On charge ensuite les fichiers Parquet dans des DataFrames Spark : 
```
def read_parquet(input_path: str) -> DataFrame:
    df = spark.read.parquet(input_path)
    return df

df_post = read_parquet('db/final_dataframe_with_topics.parquet')
df_cluster = read_parquet('db/cluster_and_topic_names.parquet')
df_subcluster = read_parquet('db/subcluster_and_topic_names.parquet')
```

### 3. Traitement des données

**a) Conversion des timestamps UNIX en dates lisibles**\
Les dates de publication des posts Reddit sont stockées en format Unix, on les convertit donc en format `date` plus lisible.
```
def convert_date(df : DataFrame) :
    df = df. withColumn("date", to_date(from_unixtime(col("created_utc"))))
    return df
```

**b) Calcul de la fréquence d'apparition des sujets**\
On regroupe les posts par clusters pour déterminer leur popularité avec `get_frequency`. 
```
def get_frequency(df : DataFrame, column : str) : 
    nb = df.groupBy(column).agg(count(column).alias("count"))
    return nb
```
```
cluster_counts = get_frequency(df_post, "cluster")
```

Puis on joint ces données avec `df_cluster` pour associer les clusters à leurs noms : 
```
cluster_join = cluster_counts.join(df_cluster, "cluster", "left")
cluster_join = cluster_join.select("cluster", "count", "topic").orderBy(col("count").desc())
```

On fait de même pour compter le nombre de posts par sous-cluster et le nombre de sous-clusters par cluster.

**c) Etude de l'évolution des clusters les plus populaires**\
Nous suivons la fréquence des clusters en fonction du temps en comptant le nombre de posts par jour et par cluster et en classant les clusters les plus populaires par jour :
```
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
```

**d) Détection des évènements marquants**\
On analyse ensuite les variations des posts pour identifier des pics d'activité : 
```
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
```

**e) Analyse de corrélation entre clusters**\
Enfin, nous avons créé une matrice de corrélation des clusters :
```
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
    cluster_names = [f"{cluster_dic[int(c)]} (ID {c})" for c in cluster_ids] # On remplace les numéros des clusters par leur nom (on garde l'id parce qu'il semble y avoir des doublons de nom)
    corr_df = pd.DataFrame(corr_array, index=cluster_names, columns=cluster_names) # On crée un df pandas avec la matrice de corrélation
    corr_unstacked = corr_df.unstack().reset_index()
    corr_unstacked.columns = ["Cluster 1", "Cluster 2", "Correlation"]
    corr_unstacked = corr_unstacked[corr_unstacked["Cluster 1"] < corr_unstacked["Cluster 2"]] # On supprime la diagonale et les doublons 
    corr_unstacked_sorted = corr_unstacked.sort_values(by="Correlation", ascending=False) # Trie par corrélation décroissante
    print("Top 10 des clusters les plus corrélés avec leurs noms :")
    print(corr_unstacked_sorted.head(10))
    corr_unstacked_sorted.to_csv("csv_files/clusters_correlation.csv", index=False) # On sauvegarde le fichier en format csv
```

## Résultats

### Les thèmes les plus populaires sur Reddit
Notre premier objectif était d'étudier les différentes tendances sur Reddit. Pour cela, nous avons effectuer plusieurs calculs car la notion de popularité peut être interprétée différemment.

**a) Les thèmes qui comptabilise le plus de posts**\
Voici les 14 thèmes avec le plus de posts : 
|cluster|count|topic|
|:-:|:-:|:-:|
|51|2494998|Music Streaming and Listening Experience|
|190|1904043|Communication and Emotions|
|173|870204|U.S. Political Elections|
|234|772541|cooking and food preparation|
|185|753851|COVID-19 Vaccination|
|101|712279|religion|
|191|589514|Crime and Violence|
|240|543217|dogs and their importance in people's lives|
|237|518860|football|
|184|513909|sexual desire|
|127|496960|Relationships and Personal Reflections|
|241|494784|cats|
|183|482611|firearms|
|105|470482|grooming|

**b) Les thèmes avec le plus de sous-clusters**\
Voici les 14 thèmes avec le plus de sous-clusters :
|cluster|topic|count|
|:-:|:-:|:-:|
|51|Music Streaming and Listening Experience|143|
|190|Communication and Emotions|80|
|127|Relationships and Personal Reflections|67|
|173|U.S. Political Elections|57|
|234|cooking and food preparation|48|
|69|space exploration|46|
|101|religion|43|
|185|COVID-19 Vaccination|39|
|184|sexual desire|	4|
|187|houseplants|32|
|214|PC Building Components and Resources|31|
|191|Crime and Violence|30|
|162|video games|30|
|41|Star Wars|29|

### L'évolution des thèmes populaires dans le temps

**a) Les cinq thèmes les plus populaires par jours**\
Certains thèmes sont extrêmement récurrents et apparaissent grossièrement presque tous les jours. Ce sont les thèmes les plus populaires comme *Music Streaming and Listening Experience* ou *Communication and Emotions*.
Cependant il est intéressant de remarquer les thèmes moins populaires qui apparaissent de temps en temps. Par exemple : 
|date|cluster|topic|count|
|:-:|:-:|:-:|:-:|
|01/01/2023|51|Music Streaming and Listening Experience|80|
|01/01/2023|190|Communication and Emotions|55|
|01/01/2023|184|sexual desire|31|
|01/01/2023|234|cooking and food preparation|28|
|01/01/2023|165|celebration|25|

Le jour de l'an 2023, on peut remarquer que certains sujets moins récurrents mais cohérents avec l'évènement du jour apparaissent : *celebration* et *cooking and food preparation*.

**b) Les évènements marquants**\
Ensuite nous avons identifié les jours où un sujet a connu une explosion soudaine d'intérêt. Voici quelques exemples intéressants :
|date|cluster|topic|count|prev_day_count|daily_variation|
|:-:|:-:|:-:|:-:|:-:|:-:|
|25/12/2020|165|celebration|8960|4646|4314|
|14/02/2021|20|Valentine's Day|5272|1146|4126|
|25/12/2022|165|celebration|8101|4263|3838|
|25/12/2021|165|celebration|8071|4249|3822|
|25/12/2019|165|celebration|7168|3422|3746|
|14/02/2022|20|Valentine's Day|4519|779|3740|
|14/02/2020|20|Valentine's Day|3905|732|3173|
|24/12/2020|165|celebration|4646|1627|3019|
|31/10/2021|70|Halloween celebrations|3789|814|2975|
|24/02/2022|248|Ukraine-Russia Conflict|3609|712|2897|

Ici on peut voir que les évènements les plus marquants correspondent aux jours de célébration comme Noël ou la St Valentin. On a également le conflit Ukraine-Russie qui a une grosse apparition le jour du début de l'invasion de l'Ukraine par la Russie.

En tant qu'autre exemple un peu moins évident on peut remarquer le jour d'une éclipse solaire : 
|date|cluster|topic|count|prev_day_count|daily_variation|
|:-:|:-:|:-:|:-:|:-:|:-:|
|21/08/2017|69|space exploration|1513|315|1198|

**c) Corrélation entre les clusters**\

Enfin, nous avons cherché à savoir si certains sujets sont liés entre eux. Pour cela nous avons calculer la corrélation entre les différents clusters en fonction de leur popularité. Plus le facteur de corrélation est proche de 1, plus les sujets évoluent de manière similaire dans le classement de popularité au fil du temps. Les 14 clusters les plus corrélés sont les suivants : 
| Cluster 1                                        | Cluster 2                                        | Correlation         |
|--------------------------------------------------|--------------------------------------------------|---------------------|
| Communication and Emotions (id 190)              | Music Streaming and Listening Experience (id 51) | 0.9847988165645469  |
| relationships (id 157)                           | sexual desire (id 184)                           | 0.9846417502867045  |
| footwear (id 171)                                | sexual desire (id 184)                           | 0.9844244135255688  |
| Music Streaming and Listening Experience (id 51) | firearms (id 183)                                | 0.979540655346468   |
| firearms (id 183)                                | grooming (id 105)                                | 0.9784924500694284  |
| Communication and Emotions (id 190)              | Relationships and Personal Reflections (id 127)  | 0.9759434646255989  |
| Communication and Emotions (id 190)              | firearms (id 183)                                | 0.9754872188552838  |
| Photography Equipment and Film Types (id 39)     | firearms (id 183)                                | 0.974997527842619   |
| footwear (id 171)                                | relationships (id 157)                           | 0.9746499155434675  |
| Communication and Emotions (id 190)              | beverages (id 164)                               | 0.9744224236189695  |
| Relationships and Personal Reflections (id 127)  | firearms (id 183)                                | 0.9742281808059525  |
| Communication and Emotions (id 190)              | language learning (id 117)                       | 0.9740603619900647  |
| Communication and Emotions (id 190)              | grooming (id 105)                                | 0.9739243628544884  |
| Relationships and Personal Reflections (id 127)  | grooming (id 105)                                | 0.9736269450484489"  |


On retrouve certains clusters qui n'ont pas l'air d'avoir de thèmes communs, comme "communication et émotions" et "Music". Cela s'explique par la grande popularité de ces thèmes qui, hors événements inhabituels, sont toujours en haut du classements des clusters les plus populaires. 
## Conclusion
Cette étude nous a permis de mieux comprendre les dynamiques de discussion sur Reddit et d’identifier des tendances fortes.

### Pistes d'amélioration
Pour aller plus loin dans l’analyse des tendances Reddit, plusieurs axes d’amélioration peuvent être envisagés :
- **Prévision des futurs sujets populaires** : Anticiper l’évolution de la popularité des sujets et prédire ceux qui émergeront dans les prochaines semaines.
- **Étude des subreddits les plus proches en termes d’utilisateurs** : En analysant les habitudes de certains utilisateurs (posts/commentaires sur plusieurs subreddits), on pourrait regrouper les subreddits similaires et cartographier les communautés qui partagent des centres d’intérêt communs.

Ces améliorations permettraient d'affiner l'analyse et d'offrir des points de vue plus précis sur les dynamiques de discussion sur Reddit.
