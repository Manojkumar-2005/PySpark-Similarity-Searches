from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local[*]").appName("Advanced Similarity Search").getOrCreate()

df = spark.read.csv("large_product_dataset_1000.csv", header=True, inferSchema=True)

df.show(5, truncate=False)

tokenizer = Tokenizer(inputCol="description", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
idf = IDF(inputCol="raw_features", outputCol="features")

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])

model = pipeline.fit(df)
result = model.transform(df)

result.cache()

result.select("product_id", "description", "features").show(5, truncate=False)

def cosine_similarity(df):
    similarities = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                similarity = float(vec1.dot(vec2) / (Vectors.norm(vec1, 2) * Vectors.norm(vec2, 2)))
                similarities.append((i + 1, j + 1, similarity))
    
    return similarities

def euclidean_distance(df):
    distances = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.linalg.norm(vec1.toArray() - vec2.toArray()))
                distances.append((i + 1, j + 1, distance))
    
    return distances

def jaccard_similarity(df):
    similarities = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                intersection = float(np.sum(np.minimum(vec1.toArray(), vec2.toArray())))
                union = float(np.sum(np.maximum(vec1.toArray(), vec2.toArray())))
                similarity = intersection / union if union != 0 else 0
                similarities.append((i + 1, j + 1, similarity))
    
    return similarities

def manhattan_distance(df):
    distances = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.sum(np.abs(vec1.toArray() - vec2.toArray())))
                distances.append((i + 1, j + 1, distance))
    
    return distances

def hamming_distance(df):
    distances = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.sum(np.abs(vec1.toArray() - vec2.toArray())) != 0)
                distances.append((i + 1, j + 1, distance))
    
    return distances

start_time = time.time()
cosine_sim = cosine_similarity(result)
cosine_time = time.time() - start_time
print("Cosine Similarity Time:", cosine_time)

start_time = time.time()
euclidean_dist = euclidean_distance(result)
euclidean_time = time.time() - start_time
print("Euclidean Distance Time:", euclidean_time)

start_time = time.time()
jaccard_sim = jaccard_similarity(result)
jaccard_time = time.time() - start_time
print("Jaccard Similarity Time:", jaccard_time)

start_time = time.time()
manhattan_dist = manhattan_distance(result)
manhattan_time = time.time() - start_time
print("Manhattan Distance Time:", manhattan_time)

start_time = time.time()
hamming_dist = hamming_distance(result)
hamming_time = time.time() - start_time
print("Hamming Distance Time:", hamming_time)

metrics = ['Angle', 'Intersect', 'EucDist', 'CityBlock', 'BitDist'] 
times = [cosine_time, jaccard_time, euclidean_time, manhattan_time, hamming_time]

plt.bar(metrics, times)
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time of Similarity Metrics')
plt.show()

cosine_sim_df = pd.DataFrame(cosine_sim, columns=["Product 1", "Product 2", "Angle"])
cosine_sim_df.to_csv("cosine_results.csv", index=False)

euclidean_dist_df = pd.DataFrame(euclidean_dist, columns=["Product 1", "Product 2", "EucDist"]) 
euclidean_dist_df.to_csv("euclid_results.csv", index=False)

jaccard_sim_df = pd.DataFrame(jaccard_sim, columns=["Product 1", "Product 2", "Intersect"]) 
jaccard_sim_df.to_csv("jaccard_results.csv", index=False)

manhattan_dist_df = pd.DataFrame(manhattan_dist, columns=["Product 1", "Product 2", "CityBlock"])  
manhattan_dist_df.to_csv("manhattan_results.csv", index=False)

hamming_dist_df = pd.DataFrame(hamming_dist, columns=["Product 1", "Product 2", "BitDist"])  
hamming_dist_df.to_csv("hamming_results.csv", index=False)
cosine_sim_link = 'cosine_results.csv'
euclidean_dist_link = 'euclid_results.csv'
jaccard_sim_link = 'jaccard_results.csv'
manhattan_dist_link = 'manhattan_results.csv'
hamming_dist_link = 'hamming_results.csv'

print(f"Cosine results: {cosine_sim_link}")
print(f"Euclidean results: {euclidean_dist_link}")
print(f"Jaccard results: {jaccard_sim_link}")
print(f"Manhattan results: {manhattan_dist_link}")
print(f"Hamming results: {hamming_dist_link}")
