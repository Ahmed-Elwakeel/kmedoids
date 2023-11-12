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
    all_to_all_df = embeddings_df.alias("i").join(embeddings_df.alias("j"),col("i.row_id") != col("j.row_id"))
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
    pair_wise_distance_df_filtered = pair_wise_distance_df.filter(col("j").isin(k_medoids_list) & ~col("i").isin(k_medoids_list) )
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

def func2(row1, row2):
    return row1[6] + row2[6]
    


    
def func1(row):
    if(row['dist'] < row['first']['dist']):
        return (row['i'] ,
                row['j'],
                row['dist'] ,
                row['j'] , 
                row['first']['dist'] ,
                row['first']['j'] , 
                row['first']['dist'] - row['dist'])
    elif(row['dist'] < row['second']['dist']):
        return (row['i'] ,
                row['j'],
                row['first']['dist'] , 
                row['first']['j'],
                row['dist'],
                row['j'] ,
                row['dist'] - row['first']['dist'] )
    else:
        return (row['i'], 
                row['j'],
                row['first']['dist'] ,
                row['first']['j'] , 
                row['second']['dist'], 
                row['second']['j'] , 
                row['removal_loss'])

def func3(accum,row):
    if accum[1] > row[1]:
        return(row)
    else: return(accum)



def func0(accum , row):

    return (accum + row)
    # return (accum[0] , accum[1][6] + row[1][6])


def calculDist(row):
    f1= row[0][0]
    f2 = row[1][0]
    n1 = row[0][2]
    n2 = row[1][2]
    dot = float(f1.dot(f2))
    denom = n1 * n2
    if denom == 0.0:
        return (row[0][1] , row[1][0] ,0)
    else:
        return (row[0][1]  , row[1][1] , 1 - float(dot / float(denom)))

def calculDist2(row):
    f1= row[0][0]
    f2 = row[1][0]
    n1 = row[0][2]
    n2 = row[1][2]
    dot = float(f1.dot(f2))
    denom = n1 * n2
    if denom == 0.0:
        return (row[0][1] , row[1][0] ,0 , row[2])
    else:
        return (row[0][1]  , row[1][1] , 1 - float(dot / float(denom)) , row[2])
    
def reduceMin(accum , row):
    if(accum[2] < row[2]):
        return accum
    else:
        return row
     
if __name__ == "__main__":
    k = 1000
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
    embeddings_df_string_id = formulate_embeddings("embeddings/input/100000_sampled_embeddings2.csv")
    
    # Having 64 bit ids is better than having string id with more consumed space 
    embeddings_df_string_id = embeddings_df_string_id.withColumn("row_id", monotonically_increasing_id())
    # remove the string_row_id for better storag consumption 
    embeddings_df = embeddings_df_string_id.drop("string_row_id")
    embeddings_df = embeddings_df.withColumn("norm", udf_get_norm(col('features')))

    # # --------------------------- Get Best K medoids OR Get Random k medoids  --------------------------#
    unique_embeddings_df = embeddings_df.select("row_id").distinct()
    # k_medoids_df = get_random_k_medoids(unique_embeddings_df, k,spark)
    # k_medoids_rows = k_medoids_df.select('i').collect()
    # for row in k_medoids_rows:
    #    k_medoids_list.append(row['i'])
    unique_embeddings_ids = unique_embeddings_df.collect()
    embeddings_ids_list = []
    for row in unique_embeddings_ids:
        embeddings_ids_list.append(row["row_id"])
    
    k_medoids_list = get_random_medoids_index(embeddings_ids_list,k)
    
    # k_medoids_list = k_meansinit(pair_wise_distance_df,embeddings_ids_list, k)
    
    k_medoids_set = {medoid for medoid in k_medoids_list}
    embeddings_rdd= embeddings_df.rdd
    min_sum = sys.maxsize
    currentStoppingIter = 0
    stoppingNumber = 2
    for i in range(1):
        t0 = time.time()
        k_embeddings_rdd = embeddings_rdd.filter(lambda x: x[1] in k_medoids_set)
        joined_rdd = embeddings_rdd.cartesian(k_embeddings_rdd)
        dist_rdd = joined_rdd.map(lambda row: calculDist(row))
        final_rdd = dist_rdd.keyBy(lambda row: row[0])
        final_rdd = final_rdd.reduceByKey(lambda accum, row: reduceMin(accum , row))
        current_sum_row = final_rdd.reduce(lambda accum, row: (accum[0] , (accum[1][0] , accum[1][1], accum[1][2] + row[1][2]) ))
        current_sum = current_sum_row[1][2]
        print(current_sum)
        if(current_sum < min_sum):
            min_sum = current_sum
        else:
            currentStoppingIter +=1
        
        if(stoppingNumber == currentStoppingIter):
            break
        grouped_rdd = final_rdd.keyBy(lambda row: row[1][1])
        
        grouped_rdd = grouped_rdd.map(lambda row: (row[0], row[1][0]))
        grouped_rdd = grouped_rdd.keyBy(lambda row: row[0])
        pair_wise_rdd = grouped_rdd.join(grouped_rdd)
        pair_wise_rdd = pair_wise_rdd.map(lambda row: (row[0] , row[1][0][1] , row[1][1][1]))
        pair_wise_rdd = pair_wise_rdd.keyBy(lambda row: (row[1]))
        embeddings_rdd_key  = embeddings_rdd.keyBy(lambda row: row[1])
        
        pair_wise_rdd = pair_wise_rdd.join(embeddings_rdd_key)
        pair_wise_rdd = pair_wise_rdd.map( lambda row: (row[1][0][0] , row[1][0][1] , row[1][0][2] , row[1][1][0] , row[1][1][2]))
        pair_wise_rdd = pair_wise_rdd.keyBy(lambda row: row[2])
        
        pair_wise_rdd = pair_wise_rdd.join(embeddings_rdd_key)
        pair_wise_rdd = pair_wise_rdd.map(lambda row: calculDist2(( 
            (row[1][0][3] , row[1][0][1] , row[1][0][4]),
            (row[1][1][0], row[1][1][1], row[1][1][2]), 
            row[1][0][0])) 
                                          )
        pair_wise_rdd_key = pair_wise_rdd.keyBy(lambda row: row[1])
        sum_rdd = pair_wise_rdd_key.reduceByKey(lambda accum,row: (accum[0], accum[1] ,row[2] + accum[2] , accum[3]) )
        sum_rdd = sum_rdd.map(lambda row: (row[1][1] , row[1][2] , row[1][3]) )

        sum_rdd=  sum_rdd.keyBy(lambda row: row[2])
        reduced_k = sum_rdd.reduceByKey(lambda accum,row : accum if accum[1] < row[1] else row )
        reduced_k_collected = reduced_k.collect()
        k_medoids_set.clear()
        for elem in reduced_k_collected:
            k_medoids_set.add(elem[1][0])
        print("----------------------------------------" , i)
        
        # pair_wise_rdd = pair_wise_rdd.map( lambda row: (row[1][0][0] , row[1][0][1] , row[1][0][2] , row[1][1][0] , row[1][1][2]))
        # pair_wise_rdd = pair_wise_rdd.keyBy(lambda row: row[2])
        # pair_wise_rdd.toDF().show(100)
        # k_embeddings_key_rdd = k_embeddings_rdd.keyBy(lambda row: row[1])
        # joined_grouped_rdd = grouped_rdd.join(k_embeddings_key_rdd)
        # joined_grouped_rdd = joined_grouped_rdd.map(lambda row: (row[0] , row[1][0][1] , row[1][1][0]))
        # joined_grouped_rdd.toDF().show(100)
        # embeddings_key_rdd = embeddings_rdd.keyBy(lambda row: row[1])
     

        # # assignment_rdd = joined_rdd.reduceByKey(lambda accum,row: reduceMin(accum, row) )
    
    print("Minimum sum :- " , min_sum )
    print("k_medoids_set :- " , k_medoids_set)
    # assignment_df = medoids_assignment(pair_wise_distance_df, k_medoids_list)
    # joined_df = pair_wise_distance_df.alias('a').join( assignment_df.alias('b') , col('a.i') == col('b.i')) 
    # joined_rdd =joined_df.rdd
    # joined_rdd = joined_rdd.map(lambda row: func1(row))
    # # joined_rdd = joined_rdd.groupBy(lambda row: ( row[1] , row[3]))
    # joined_rdd = joined_rdd.keyBy(lambda row: ( row[1] , row[3]))
    # # when I added row[0][0] , the removal loss of assigning points to row[0][1] = 
    # joined_rdd = joined_rdd.map(lambda row: (row[0] , row[1][6]))
    # joined_rdd = joined_rdd.reduceByKey(lambda accum,row: func0(accum,row))
    # # joined_rdd = joined_rdd.map(lambda x: (x[0], sum(item[6] for item in x[1])))
    # filtered_joined_rdd = joined_rdd.filter(lambda x: x[0][0] == x[0][1])
    # filtered_joined_rdd = filtered_joined_rdd.keyBy(lambda x: (x[0][0]))
    # other_join_rdd = joined_rdd.filter(lambda x: x[0][0] != x[0][1])
    # other_join_rdd = other_join_rdd.keyBy(lambda x: (x[0][0]))
    # # other_join_rdd = other_join_rdd.reduceByKey(lambda x,y: func3(x,y))
    # new_joined_rdd = other_join_rdd.join(filtered_joined_rdd)
    # new_joined_rdd = new_joined_rdd.map( lambda row: (row[0], (row[1][0][0][1] , row[1][0][1] + row[1][1][1])))
    # # new_joined_rdd = new_joined_rdd.keyBy(lambda row: row[0])
    # new_joined_rdd = new_joined_rdd.reduceByKey(lambda accum,row : func3(accum , row))
    # new_joined_rdd = new_joined_rdd.reduce(lambda accum,row : func3(accum , row))
    # print(new_joined_rdd)
    # t1 = time.time()
    # print("total time ", t1 - t0)
