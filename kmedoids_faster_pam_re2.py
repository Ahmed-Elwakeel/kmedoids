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
import sys

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
   
def medoids_assignment(pair_wise_distance_df, k_medoids_list):
    # J here is medoids and I is the whole dataset 
    # And here we are trying to assign for each embeddings the nearest and the second nearest medoid 
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

def k_meansinit(pair_wise_distance_df,embeddings_ids_list,k):
    random_medoid_df_index = sample(embeddings_ids_list,1)[0]
    
    rest_of_random_medoids_df = pair_wise_distance_df.filter(col("i") == random_medoid_df_index).orderBy(col("dist").desc()).limit(k-1).select("j")
    rest_of_random_medoids_indices =rest_of_random_medoids_df.rdd.map(lambda x: x.j).collect()
    rest_of_random_medoids_indices.append(random_medoid_df_index)
    print(rest_of_random_medoids_indices)
    return rest_of_random_medoids_indices

def get_random_medoids_index(embeddings_ids_list,k):
   
    k_medoids_list = sample(embeddings_ids_list,k)
    return k_medoids_list

def get_min_removal(all_medoids_loss , added_embedding ):
    removal_loss_struct_df = all_medoids_loss.withColumn("loss_struct", struct("removal_loss", "j"))
    minimum_removal_loss_df = removal_loss_struct_df.agg(min(col("loss_struct")).alias("min_removal_loss"))
    minimum_removal_loss_df = minimum_removal_loss_df.withColumn(
        "min_removal_loss_medoid", col("min_removal_loss.j")
        ).withColumn("min_removal_loss_value", col("min_removal_loss.removal_loss"))
    # minimum_removal_loss_value = minimum_removal_loss_df.collect()[0]["min_removal_loss_value"]
    # row = minimum_removal_loss_df.collect()[0]
    # minimum_removal_loss_medoid = row["min_removal_loss_medoid"]
    # minimum_removal_loss_value = row["min_removal_loss_value"]
    filtered_df = all_medoids_loss.filter(col('j') == added_embedding)
    # if(filtered_df.count()!= 0):
    filtered_df = filtered_df.withColumn('index', lit(1))
    minimum_removal_loss_df = minimum_removal_loss_df.withColumn('index' , lit(1))
    new_df = filtered_df.alias('a').join(minimum_removal_loss_df.alias('b') , col('a.index') == col('b.index') )
    new_df= new_df.withColumn('loss' , col('min_removal_loss_value') + col('removal_loss'))
    new_df =new_df.withColumnRenamed('min_removal_loss_medoid','removed_medoid')
    new_df =new_df.withColumnRenamed('j','added_medoid')
    new_df = new_df.drop('index','j','removal_loss','min_removal_loss','min_removal_loss_medoid','min_removal_loss_value')
    return new_df
    # else:
    #     empty_df = spark.createDataFrame([], StructType([
    #     StructField("loss", IntegerType(), True),  
    #     StructField("removed_medoid" ,IntegerType(),True),
    #     StructField("added_medoid" ,IntegerType(),True)
    #                                      ]))
    #     return empty_df
        
    # added_embedding_removal_loss_value = 0
    # if(filtered_df.count() != 0):
    #     added_embedding_removal_loss_value = all_medoids_loss.filter(col('j') == added_embedding).collect()[0]['removal_loss']

    # minimum_removal_loss_value = added_embedding_removal_loss_value + minimum_removal_loss_value
    # return minimum_removal_loss_value , minimum_removal_loss_medoid

def func2(row ):
    return row    
    
def func1(embedding_id , initial_assignment_rdd ):
    # new_embedding_pair_wise_distance_rdd = pair_wise_dsitance_df.filter( (col("j") == embedding_id) | (col("i") == embedding_id) ).rdd
    new_mapped = initial_assignment_rdd.map(lambda row: func2(row))
    return new_mapped
    
if __name__ == "__main__":
    k = 10
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

    pair_wise_distance_df = spark.read.parquet("100_pair_wise")
    # pair_wise_distance_df=pair_wise_distance_df.repartition(100, "i")
    pair_wise_distance_df.cache()


    # # --------------------------- Get Best K medoids OR Get Random k medoids  --------------------------#
    unique_embeddings_df = pair_wise_distance_df.select("i").distinct()
    # k_medoids_df = get_random_k_medoids(unique_embeddings_df, k,spark)
    # k_medoids_rows = k_medoids_df.select('i').collect()
    # for row in k_medoids_rows:
    #    k_medoids_list.append(row['i'])
    unique_embeddings_ids = unique_embeddings_df.collect()
    embeddings_ids_list = []
    for row in unique_embeddings_ids:
        embeddings_ids_list.append(row["i"])
    
    # k_medoids_list = get_random_medoids_index(get_random_medoids_index,k)
    
    k_medoids_list = k_meansinit(pair_wise_distance_df,embeddings_ids_list, k)
    
    k_medoids_set = {medoid for medoid in k_medoids_list}

    assignment_df = medoids_assignment(pair_wise_distance_df, k_medoids_list)


    # -------------------------- Main Iteration -------------------------------
    # min_agg = spark.createDataFrame([], StructType([
    #     StructField("loss", IntegerType(), True),  
    #     StructField("medoids_list" ,ArrayType(IntegerType()),True)
    #                                      ]))
    min_agg = spark.createDataFrame([], StructType([
        StructField("loss", IntegerType(), True),  
        StructField("removed_medoid" ,IntegerType(),True),
        StructField("added_medoid" ,IntegerType(),True)
                                         ]))

    maxIterations = 1
    # min_loss_value = loss
    min_loss_medoids = k_medoids_list
    iteration_losses_values = []
    iteration_losses_medoids = []
    iteration_medoids_set = k_medoids_set
    current_loss = sys.maxsize

    for _ in range(maxIterations):
        iterT0 = time.time()
        k_medoids_set = iteration_medoids_set.copy()
        totalTimePerIteration = 0
        countPerIteration = 0
        # embeddings_ids_list.map((embedding_id)=>{
        #     initial_assignment.forEach((row)=>{
        #         return loss , medoids   
        #     })
        # })
        assignment_rdd = assignment_df.rdd
        embeddings_list_rdd = sc.parallelize(embeddings_ids_list)

        x = embeddings_list_rdd.map(lambda embedding_id: func1(embedding_id , assignment_rdd ))
        y = x.collect()
        for elem in y:
            print(elem)
    #     # Iterate over all the non medoids and try swap it with any of the current medoids and if it affects the total loss then consider it 
    #     for embedding in embeddings_ids_list:
    #         countPerIteration +=1
    #         t0 = time.time()
            
    #         # This is already a medoid, skip 
    #         if(embedding in k_medoids_set):
    #             continue
    #         new_k_medoids_set = k_medoids_set.copy()
    #         new_k_medoids_set.add(embedding)
    #         # Here we are trying to assign each point to one of the existing medoids + the new one 
    #         # After that we will have removal_loss for each medoids 
    #         # Now we want to get the best k medoids out of k+1 medoids 
    #         # This means that we need to remove the medoids that has the least impact on the total loss which is the minmum removal_loss 
    #         to_be_passed_list = list(new_k_medoids_set)
    #         all_medoids_assignments_df = medoids_assignment(pair_wise_distance_df , to_be_passed_list)
    #         all_medoids_loss = get_removal_loss(all_medoids_assignments_df)
    #         new_df = get_min_removal(all_medoids_loss ,embedding)
    #         # if(minimum_removal_loss_medoid == embedding):
    #         #     continue
            
    #         # new_k_medoids_set.remove(minimum_removal_loss_medoid)
    #         # new_row = spark.createDataFrame([(minimum_removal_loss_value, list(new_k_medoids_set))], ['loss','medoids_list'])
           
    #         min_agg = min_agg.union(new_df)
            
    #         t1 = time.time()

    #         total = t1-t0
    #         print( "Iteration time :-" , total)
           
    #         totalTimePerIteration += total
    #         tAfterPrint = time.time()
    #         print( "printing --- > " ,tAfterPrint - t1)
    #         # if(current_loss < min_loss_value):
    #         #     min_loss_value = current_loss
    #         #     min_loss_medoids = list(k_medoids_set)
    #     # loss_df.show()
    #     print("--------------- loss df ----------------")
    #     min_agg.show()
    #     min_loss_value = min_agg.groupBy().min("loss").collect()[0]["min(loss)"]
    #     # min_loss_df.show()
    #     print(" -------------- min _lost -----------------")    
     
    #     res_medoids = min_agg.filter(col("loss") == min_loss_value).collect()[0]
    #     print("-----------------------------------------------------")
    #     print(res_medoids)
    #     print("=================================================")
    #     # if(min_loss_value >= current_loss):
    #     #     break
    #     current_loss = min_loss_value
    #     iteration_losses_medoids = min_loss_medoids.copy()
    #     print(" ------ > sss ", min_loss_value , " --- " , min_loss_medoids , " ----- " , iteration_losses_medoids)
        
    #     print("-------------->" ,iteration_losses_medoids)
    #     print("Average time per inner iteration -> " , totalTimePerIteration /countPerIteration )
    #     iterT1 = time.time()
    #     iterTimeTotal = iterT1-iterT0
    #     print( "OUTER Iteration time :-" , iterTimeTotal)
    #     totalTimePerIteration += iterTimeTotal

    #     # iteration_losses_values.append(min_loss_value)
    #     # iteration_losses_medoids.append(min_loss_medoids)
    # # --------------------------- SWAP PHASE ---------------------------------#
    # print("Average time per a total iteration -> ",  totalTimePerIteration / maxIterations)
    
    