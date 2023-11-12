import pyspark
import random
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.sql.types import FloatType ,StructType,StructField , StringType
from pyspark.sql.types import IntegerType, ArrayType 
import pyspark.sql.functions as F
from pyspark.sql.functions import least, when,concat_ws,array, min,round,abs,row_number , max,struct, lit , col,split,monotonically_increasing_id ,broadcast
import numpy as np
from numpy.linalg import norm
from pyspark.sql.window import Window
from pyspark.sql.functions import col
from pyspark.mllib.linalg.distributed import RowMatrix, CoordinateMatrix
from pyspark.mllib.util import MLUtils
from pyspark.sql.window import Window
from pyspark.mllib.linalg.distributed import RowMatrix
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



# Expects the input file path and the file type (csv , parquet) 
# Returns a formulate embeddings dataframe in the following format 
# features  | string_row_id
# [0.01,...]| A20-xy
#  .....

# It is assumed that the file has 1025 column where the first row represent the id with string "id description ... .."
# The line is split on the space character and the first item considered the id 
# Then the rest of the 1024 columns will be casted to float 
# It can also accepts file with 1024 columns where the first id column is omitted
def formulate_embeddings(input_file_path , file_type , spark):
    if(file_type == "csv"):
        df = spark.read.csv(input_file_path)
    elif(file_type =="parquet"):
        df = spark.read.parquet(input_file_path)
        
    # Handling missing string id column in the input file
    cols_count = len(df.columns)
    if(cols_count == 1024):
        df = df.select(lit(0).alias("temp_col"), "*")
    
    embeddings_in_float_df = df.select(
        split(df.columns[0], ' ').getItem(0) ,
        *(col(column).cast(FloatType()) for column in df.columns[1:])
    )
    # The 1024 column excluding the first column (id) 
    embeddingCols = embeddings_in_float_df.columns[1:]
    assembler = VectorAssembler(inputCols=embeddingCols, outputCol="features")
    # This will skip any of the invalid rows 
    assembled_df = assembler.setHandleInvalid("skip").transform(embeddings_in_float_df)
    embeddings_df = assembled_df.select(assembled_df.columns[0] ,"features").withColumnRenamed(assembled_df.columns[0], "string_row_id")
    return embeddings_df

   

# The function uses the k_means strategy to select the initial k-medoids by selecting the most far away medoids from each other based on the distance 
# The function is more time expensive than just using random numbers
def k_means_init(pair_wise_distance_df,embeddings_ids_list,k):
    random_medoid_df_index = sample(embeddings_ids_list,1)[0]
    rest_of_random_medoids_df = pair_wise_distance_df.filter(col("i") == random_medoid_df_index).orderBy(col("dist").desc()).limit(k-1).select("j")
    rest_of_random_medoids_indices =rest_of_random_medoids_df.rdd.map(lambda x: x.j).collect()
    rest_of_random_medoids_indices.append(random_medoid_df_index)
    return rest_of_random_medoids_indices

# The function returns random k medoids of the list 
def get_random_medoids_index(embeddings_ids_list,k):
    k_medoids_list = sample(embeddings_ids_list,k)
    return k_medoids_list


def get_loss_rdd(assignment_rdd):
    final_value =  assignment_rdd.reduce(lambda accum , row: (accum[0] , accum[1] , (accum[2][0] + row[2][0] , accum[2][1]) , accum[3] , ))
    return final_value[2][0]



# The function compare each distance of all to all distance and recalculate the first and second objects 
# The following format is returned [ the considered medoid to be added , the nearest medoid ,  the updated removal_loss ]
def formulate_all_to_all_dist_plus_initial_assignment_df(input_row):
    # case where the new distance is the better than the actual assigned
    row = {}
    row['dist'] = input_row[1][0][2]
    row['j'] = input_row[1][0][1]
    row['first'] = {'dist' : input_row[1][1][2][0] , 'j' : input_row[1][1][2][1] }
    row['second'] = {'dist' : input_row[1][1][1][0] , 'j' : input_row[1][1][1][1] }
    row['removal_loss'] = input_row[1][1][3]
 
    if(row['dist'] < row['first']['dist']):
        return (
                row['j'],
                row['j'] , 
                row['dist']  - row['first']['dist'])
    # Case where the new distance is better the second and not better than the first 
    elif(row['dist'] < row['second']['dist'] and row['dist'] != row['first']['dist']):
        return (
                row['j'],
                row['first']['j'],
                row['dist'] - row['first']['dist'])
    # Case where the distance should not affect teh removal loss, then return it as it is 
    else:
        return (
                row['j'],
                row['first']['j'] , 
                row['removal_loss'])

  
def add_dist_from_cos_sim(row):
    f1 = row[0]
    f2 = row[3]
    n1 = row[2]
    n2 = row[5]
    i = row[1]
    j = row[4]
    dot = float(f1.dot(f2))
    denom = n1 * n2
    cos_sim = -1
    if denom != 0.0:
        cos_sim = float( dot / float(denom))
    return (i , j , 1 - cos_sim)   
# The function returns the pair-wise distance between each row in the provided embeddings 
def get_pair_wise_distance_rdd(embeddings_rdd):
    embeddings_df = embeddings_rdd.toDF()
    # broadcast_embeddings_df = broadcast(embeddings_df)
    pair_wise_df = embeddings_df.crossJoin(embeddings_df)
    # pair_wise_df.show()
    pair_wise_rdd = pair_wise_df.rdd
    pair_wise_distance_rdd = pair_wise_rdd.map(lambda row: add_dist_from_cos_sim(row) )
    # pair_wise_distance_rdd = pair_wise_distance_rdd.map(lambda row: ( row[0] , row[1] , row[]) )
    return pair_wise_distance_rdd
    
def get_second_min(accum , row):
    if(accum[1][0][1] == row[1][1][1]):
        return row
    if(row[1][0][1] == row[1][1][1]):
        return accum

    elif accum[1][0][2] < row[1][0][2]:
        return accum
    else:
        return row
    
# The function assigned each embedding to one of the medoids passed in k_medoids_set 
def medoids_assignment_rdd(pair_wise_distance_rdd , k_medoids_set):

    assignments = pair_wise_distance_rdd.filter( lambda row : row[1][1] in k_medoids_set )
    min_assignments = assignments.reduceByKey(lambda accum ,row : accum if accum[2] < row[2] else row)
    assignments = assignments.join(min_assignments)
    assignments = assignments.keyBy(lambda row : row[0])
    
    final_assignments = assignments.reduceByKey(lambda accum ,row : get_second_min(accum , row))
    final_assignments = final_assignments.map(lambda row : (row[0] , (row[1][1][0][2] , row[1][1][0][1] ) , (row[1][1][1][2] , row[1][1][1][1] ) ,   row[1][1][0][2] - row[1][1][1][2] ) )
    return final_assignments
   
   
if __name__ == "__main__":
    # ---------------------------- Initializations ------------------------------- #
    spark = SparkSession.builder.appName("kmedoids").getOrCreate()
    sc = SparkContext.getOrCreate()
    k = 10
    filePath = "embeddings/input/5000_sampled_embeddings.csv"
    
    #--------------------------- UDFS ----------------------------------------#
    udf_get_norm = F.udf(get_norm, FloatType())
    udf_dot_product = F.udf(lambda x,y: float(x.dot(y)), FloatType())

    # --------------------------- Cleaning and formatting --------------------- #
    # This function will read the data from the file and return a dataframe in the following format
    # features         | string_row_id
    # [0.1,0.2,....]  | string_id
    # [0.4,....... ]  | string_Id
    embeddings_df_string_id = formulate_embeddings(filePath , 'csv',spark)

    # Having 64 bit ids is better than having string id with more consumed space, so dropping the string row id
    # embeddings_df_string_id = embeddings_df_string_id.withColumn("row_id", monotonically_increasing_id())
    # remove the string_row_id for better storage consumption 
    embeddings_df = embeddings_df_string_id.drop("string_row_id")
    embeddings_df = embeddings_df.withColumn("norm", udf_get_norm(col('features')))
    embeddings_rdd = embeddings_df.rdd
    embeddings_rdd = embeddings_rdd.zipWithIndex()
    embeddings_rdd = embeddings_rdd.map(lambda row: (row[0][0] , row[1] , row[0][1]))
    
    
    #---------------------------  Create the pair wise distance matrix --------------------------- #
    # This is the most expensive operation in the algorithm.
    # It basically cross joins the rdd on itself to create a single rdd that has the distances between each row and the other
    # The following format will be produced 
    # _0  | _1 | _2
    # 0  | 0 | 0
    # 0  | 1 | 0.012
    # 0  | 2 | 0.006
    # 1  | 0 | 0.012
    # 1  | 1 | 0
    pair_wise_distance_rdd = get_pair_wise_distance_rdd(embeddings_rdd)
 
    pair_wise_distance_rdd = pair_wise_distance_rdd.keyBy(lambda row: (row[0]))
    # Caching the data here will be beneficial later for the fast access since this dataframe will always be our reference to get the distances
    pair_wise_distance_rdd.cache()
    
    
    
    # --------------------------- Initialize k points   --------------------------#
    # First we need to get all the row_ids generated 
    embeddings_ids_list_rdd = embeddings_rdd.map(lambda row : row[1])
    embeddings_ids_list = embeddings_ids_list_rdd.collect()
    
    #  Either use the random initializations or the k_meansinit initializations 
    k_medoids_list = get_random_medoids_index(embeddings_ids_list,k)
    #  Comment the above line and uncomment the below line, if the k_means_init is wanted
    # k_medoids_list = k_means_init(pair_wise_distance_df,embeddings_ids_list, k)
    
    k_medoids_set = {medoid for medoid in k_medoids_list}
 
    # -------------------------- Start iterating until convergence ---------------------- #
    # Maximum number of iterations to stop after 
    max_iteration = 5
    # Number of iterations taken for the algorithm to converge
    numIter = 0
    # This will have the total removal loss value over the iterations
    removal_loss_values = []
    # This will have the trace of how medoids change over the iterations 
    medoids_values_considered = []
    # k_medoids_set = { 5, 7 }

    for _ in range(max_iteration):
        t0 = time.time()

        numIter +=1
        # Assign each point to one of the selected points and return an rdd of rows 
        # Where each row is (i from 1 to n , first: {dist, j is medoid id} , second : {dist,j is medoid id } ,removal_loss  )
        # +---+----------------+----------------+------------+
        # |i  |first           |second          |removal_loss|
        # +---+----------------+----------------+------------+
        # |0  |{0.49680316, 17}|{0.5600259, 94} |0.06322271  |
        # |1  |{0.45789117, 59}|{0.604762, 17}  |0.14687085  |
        assignment_rdd = medoids_assignment_rdd(pair_wise_distance_rdd, k_medoids_set)
        # Joins the pair_wise distance with the assignment_df to add the distance between each i and each other element in the dataset
        # +---+---+----------+---+----------------+---------------+------------+
        # |i  |j  |dist      |i  |first           |second         |removal_loss|
        # +---+---+----------+---+----------------+---------------+------------+
        # |0  |1  |0.6995637 |0  |{0.49680316, 17}|{0.5600259, 65}|0.06322271  |
        # |0  |2  |0.69839835|0  |{0.49680316, 17}|{0.5600259, 65}|0.06322271  |
        # |0  |3  |0.8194193 |0  |{0.49680316, 17}|{0.5600259, 65}|0.06322271  |
        assignment_rdd = assignment_rdd.keyBy(lambda row : (row[0]))
        all_to_all_dist_plus_initial_assignment_rdd = pair_wise_distance_rdd.join(assignment_rdd)  
        # all_to_all_dist_plus_initial_assignment_rdd = all_to_all_dist_plus_initial_assignment_df.rdd
        # +---+---+------------------- +---+-------------------+---+--------------------------------------+
        # |i  |j   |dist nearest       |1st medoid | 2nd nearest dist    |2nd medoid |removal_loss (2nd - 1st)|
        # +---+---+------------------- +---+-------------------+---+--------------------------------------+
        # |0  |1   |0.48289060592651367|46         |0.4968031644821167   |17         |0.013912558555603027|
        # |0  |2   |0.48289060592651367|46         |0.4968031644821167   |17         |0.013912558555603027|
        all_assignments_rdd = all_to_all_dist_plus_initial_assignment_rdd.map(lambda row: formulate_all_to_all_dist_plus_initial_assignment_df(row))
        # We don't want to consider adding the current medoid, it is redundant since we know already 
        all_assignments_rdd = all_assignments_rdd.filter(lambda row: row[0] not in k_medoids_set)

        # The goal from on to find the medoid replacement which has th lowest removal loss 
        # Given that we have the current medoid, what is the removal loss if we wanted to add each other embedding 
        # Key by each added medoid and the assigned first medoid to later sum the removal loss for each 
        all_assignments_rdd_key= all_assignments_rdd.keyBy(lambda row: ( row[0] , row[1]))
        all_assignments_rdd_key = all_assignments_rdd_key.map(lambda row: (row[0] , row[1][2]))
        removal_loss_summation = all_assignments_rdd_key.reduceByKey(lambda accum,value:  accum + value)
        # Separate the records where the added medoid is the same as the current medoid from the rest 
        # Add change the key to have a key only on the 
        same_medoid_removal_loss_sum = removal_loss_summation.filter(lambda x: x[0][0] == x[0][1])
        same_medoid_removal_loss_sum = same_medoid_removal_loss_sum.keyBy(lambda x: (x[0][0]))
        rest_removal_loss_sum = removal_loss_summation.filter(lambda x: x[0][0] != x[0][1])
        rest_removal_loss_sum = rest_removal_loss_sum.keyBy(lambda x: (x[0][0]))
        # Join both rdd because we want to add the removal loss of replacing the minimum and the added medoid it self 
        new_joined_rdd = rest_removal_loss_sum.join(same_medoid_removal_loss_sum)
        # At this point you have a row with each medoid and the corresponding possible replacement with the total removal_loss
        new_joined_rdd = new_joined_rdd.map( lambda row: (row[0], (row[1][0][0][1] , row[1][0][1] + row[1][1][1])))
        # Find the minimum summation of removal loss for each element
        final_value = new_joined_rdd.reduce(lambda accum,row : accum if accum[1][1] < row[1][1] else row)
        # The final value will have the following format ( The added medoid id ( the to be replace medoid id form the current , minimum total removal loss ) )
        removal_loss_values.append(final_value)
        medoids_values_considered.append(k_medoids_set.copy())
        
        
        if(final_value[1][1] >= 0 ):
            break
        else:
            # Update the current k_medoids_set for the next iteration by do the actual swapping of the medoids 
            k_medoids_set.remove(final_value[1][0])
            k_medoids_set.add(final_value[0])
        
    
        t1 = time.time()
        print(" current removal loss :- ",  final_value[1][1])
        # print(" current medodis :- ", k_medoids_set)

        print("total time for iteration " , numIter , " is :", t1 - t0)

    
    print("Total number of iterations needed for convergence :- " , numIter)
    print("The removal loss updates till convergence :- " , removal_loss_values)
    print("The medoids update till convergence :- ", medoids_values_considered)
    print("The final selected medoids :- ",  k_medoids_set)
    # Get the total loss for the best selected medoids 
    assignment_rdd = medoids_assignment_rdd(pair_wise_distance_rdd, k_medoids_set)
    loss_value =  get_loss_rdd(assignment_rdd) 
    print("final total loss for the selected medoids  :- ", loss_value)
    
    
    
    