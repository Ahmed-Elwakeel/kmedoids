import pyspark
import random
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

# from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import FloatType,DoubleType
from pyspark.sql.functions import col,split
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType 
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, explode, least, when,concat_ws,array, min,round,abs,row_number , max,struct, lit
from pyspark.ml.linalg import VectorUDT, DenseVector
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
from numpy.linalg import norm
from pyspark.sql.window import Window
from pyspark.sql.types import StructType,StructField, StringType
from pyspark.sql.functions import col
from pyspark.mllib.linalg.distributed import RowMatrix, CoordinateMatrix
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
from pyspark.sql.window import Window


from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
import pandas as pd
import time
from random import sample




# Expects a list of vectors and returns the norm 
def get_norm(vector):
    float_casted_vector = [float(i) for i in vector]
    norm_value = norm(np.array(float_casted_vector))
    return float(norm_value)
   

def  get_cosine_similarity_given_norms(XNorm, YNorm, dot ):
    denom = XNorm * YNorm
    if denom == 0.0:
        return -1.0
    else:
        return float(dot / float(denom))

# Return the distance between the vectors based on the cosine simalirty
def get_pair_wise_distance(embeddings_df):
    # Join the embeddings_df to itself to create a all to all mapping of each embeddings. 
    # The condition here col("i.row_id") < col("j.row_id") to remove the replicated joins 
    all_to_all_df = embeddings_df.alias("i").join(embeddings_df.alias("j"),col("i.row_id") < col("j.row_id"))
    # Add the column "dot" which is the dot product between the the features vector for each pair
    all_to_all_dot_df = all_to_all_df.select("*",udf_dot_product("i.features", "j.features").alias("dot"))
    # Add the column "sim" which is the cosine simalarity calculated from the norm and dot 
    all_to_all_dist_df = all_to_all_dot_df.withColumn("dist",1 - udf_get_cos_sim("i.norm", "j.norm","dot"))
    # Select only the required columns, ids and the cosine simaliry 
    all_to_all_dist_df = all_to_all_dist_df.select( col("i.row_id").alias("i"),col("j.row_id").alias("j") , "dist")
    return all_to_all_dist_df     

# Return the distance between the vectors based on the cosine simalirty
def lib_get_pair_wise_distance(embeddings_df,spark):
    embeddings_df = embeddings_df_string_id.select("features")
    # Converting due to library changes from ML to MLIB
    embeddings_df = MLUtils.convertVectorColumnsFromML(embeddings_df)
    embeddings_row_matrix = RowMatrix(embeddings_df)
    cm = CoordinateMatrix(embeddings_row_matrix.rows.zipWithIndex().flatMap(lambda x: [MatrixEntry(x[1], j, v) for j, v in enumerate(x[0])]))
 
    sims = cm.transpose().toRowMatrix().columnSimilarities(threshold=0.1)
    all_to_all_dist_df = spark.createDataFrame(sims.entries)
    all_to_all_dist_df = all_to_all_dist_df.withColumn("dist",1 - abs(all_to_all_dist_df.value)).drop("value")
    return all_to_all_dist_df
    
#  This will take as an input the path of the embeddings csv file and return dataframe of of two columns, 
# feature (vector of 1024 number between -1 and 1 ) and a row_id (unique numbers, doesn't gurantee consecutive numbers)
#  
# features         | row_id
# [0.1,0.2,....]  | string_id
# [0.4,....... ]  | string_Id
def formulate_embeddings(input_file_path):
    csv_df = spark.read.csv(input_file_path)
    csv_df.show()
    # It is assumed that the file has 1025 column where the first row represent the id with string "id description ... .."
    # The line is split on the space character and the first item considered the id 
    # Then the rest of the 1024 columns will be casted to float 
    csv_embeddings_in_float_df = csv_df.select(
        split(csv_df.columns[0], ' ').getItem(0) ,
        *(col(column).cast(FloatType()) for column in csv_df.columns[1:])
        )

    # The 1024 column excluding the first column (id) 
    embeddingCols = csv_embeddings_in_float_df.columns[1:]
    assembler = VectorAssembler(inputCols=embeddingCols, outputCol="features")
    # This will skip any of the invalid rows 
    assembled_df = assembler.setHandleInvalid("skip").transform(csv_embeddings_in_float_df)
    
    embeddings_df = assembled_df.select(assembled_df.columns[0] ,"features").withColumnRenamed(assembled_df.columns[0], "string_row_id")
    return embeddings_df


def assign_points_to_clusters(cos_sim_df, clusters_col_names_list):
    # this will add a new column "min" which will have the minimum value for each row from the clustesr
    cos_sim_with_min_col_df = cos_sim_df.withColumn("min", least(*clusters_col_names_list))
    
    # This will add a new column "assigned_cluster" which will have the cluster name (column name) which corrosponds to the one that has the minimum value
    clustered_df = cos_sim_with_min_col_df.select("*",concat_ws(";",array([
       when(col(c)==col("min") ,c).otherwise(None)
       for c in clusters_col_names_list
     ])).alias("assigned_cluster") )

    # This will calculate the summation of the values for each cluster
    # td_score = clustered_df.groupBy("assigned_cluster").sum("min")
    return clustered_df

def get_best_k_medoids_based_on_total_distance(sum_and_min_distance_df , k):

    # Create Window specification to order by each element by the total distances
    window_spec = Window.orderBy(F.col("total_dist"))

    # Add row_number column
    sum_and_min_distance_df_with_row_number = sum_and_min_distance_df.withColumn("row_number", F.row_number().over(window_spec))
    
    # Filter top k rows
    best_k_medoids_df = sum_and_min_distance_df_with_row_number.filter(F.col("row_number") <= k)
    return best_k_medoids_df
    # best_k_medoids = best_k_medoids_df.collect()
def get_random_k_medoids(sum_and_min_distance_df, k,spark):
    
    # Set a random seed for reproducibility (optional)
    seed = 42

    # Sample random rows from the DataFrame
    random_k_medoids = sum_and_min_distance_df.sample(fraction=1.0 , seed=seed).limit(k)

    return random_k_medoids
   
def initial_assignment(pair_wise_distance_df, k_medoids_list):
    pair_wise_distance_df_filtered = pair_wise_distance_df.filter(col("j").isin(k_medoids_list))
    windowDist = Window.partitionBy("i").orderBy(col("dist").asc())
    first_and_second_min = pair_wise_distance_df_filtered.withColumn("row",row_number().over(windowDist)).filter(col("row") <= 2)
    first_and_second_min = first_and_second_min.withColumn("object" , struct("dist" , "j")).groupBy(col("i")).agg(
        min("object").alias("first"), 
        max("object").alias("second")
        )
    first_and_second_min = first_and_second_min.withColumn("removal_loss", col("second.dist") - col("first.dist"))
    return first_and_second_min


# def initial_assignment(pair_wise_distance_df, k_medoids_list):
#     pair_wise_distance_df_filtered = pair_wise_distance_df.filter(col("j").isin(k_medoids_list))
  
#     distances_in_object_df  = pair_wise_distance_df_filtered.withColumn("object" , struct("dist", "j"))
#     first_min = distances_in_object_df.groupBy(col("i")).agg(min("object")).withColumnRenamed("min(object)","first")

#     rest_pair_wise_distance_df = distances_in_object_df.alias("a").join(
#         first_min.alias("b"), (col("a.i") == col("b.i")) & (col("a.object.j") ==  col("b.first.j")),"leftanti")
#     second_min = rest_pair_wise_distance_df.groupBy(col("i")).agg(min("object")).withColumnRenamed("min(object)", "second")
   
#     first_and_second_min = first_min.alias("a").join(second_min.alias("b"), col("a.i") ==  col("b.i"),"inner").select("a.i" , "a.first" , "b.second")
#     first_and_second_min = first_and_second_min.withColumn("removal_loss", col("second.dist") - col("first.dist"))
#     return first_and_second_min


# Removal loss here is what will be the added loss to the original loss if we removed the one the medoids  
# and here it is calculated for each medoid
def get_removal_loss(embeddings_assigned_df):
    medoids_loss_df = embeddings_assigned_df.groupBy("first.j").sum("removal_loss").withColumnRenamed("sum(removal_loss)","removal_loss")
    return medoids_loss_df    

def get_loss_df(assignment_df , k_medoids_list):
    loss_df = assignment_df.groupBy().sum("first.dist").withColumnRenamed("sum(first.dist AS dist)", "loss")
    loss_df = loss_df.withColumn("medoids" ,lit(k_medoids_list).cast(ArrayType(IntegerType())))
    return loss_df

def get_loss(assignment_df):
    return assignment_df.groupBy().sum("first.dist").withColumnRenamed("sum(first.dist AS dist)", "loss").collect()[0]["loss"]

if __name__ == "__main__":
    k = 3
    spark = SparkSession.builder.appName("test").getOrCreate()
    sc = SparkContext.getOrCreate()
    #--------------------------- UDFS ----------------------------------------#
    udf_get_cos_sim = F.udf(get_cosine_similarity_given_norms, FloatType())
    udf_get_norm = F.udf(get_norm, FloatType())
    udf_dot_product = F.udf(lambda x,y: float(x.dot(y)), FloatType())

    #--------------------------- UDFS ----------------------------------------#

    # this will return the following dataframe 
    # features         | string_row_id
    # [0.1,0.2,....]  | string_id
    # [0.4,....... ]  | string_Id
    embeddings_df_string_id = formulate_embeddings("embeddings/input/100_sampled_embeddings.csv")
    # embeddings_df_string_id.show()
    # embeddings_df_string_id.show(5,False)

    # -------------------------- BUILD PHASE : Pair wise distance using ML library --------------------------------#
    # Having 64 bit ids is better than having string id with more consumed space 
    embeddings_df_string_id = embeddings_df_string_id.withColumn("row_id", monotonically_increasing_id())
    # remove the string_row_id for better storag consumption 
    embeddings_df = embeddings_df_string_id.drop("string_row_id")
    embeddings_df = embeddings_df.withColumn("norm", udf_get_norm(col('features')))

    pair_wise_distance_df = get_pair_wise_distance(embeddings_df)
    # pair_wise_distance_df/
    pair_wise_distance_df.show()
    pair_wise_distance_df.write.parquet("test_1_10d00df0_pair_wise_distance") 

    # pair_wise_distance_df.cache()
   
   
#     spark-submit  --master yarn --deploy-mode cluster --num-executors 10 --executor-cores=1 --driver-memory=4G  --executor-me
# mory=6G kmedoids_first.py
# 1 2
# 1 3
# 1 4
# 1 4
# 2 3
# 2 4
# 2 5
# 3 4