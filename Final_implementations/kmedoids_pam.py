from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.types import FloatType 
import pyspark.sql.functions as F
from pyspark.sql.functions import  col
import pandas as pd
import time
import sys
from random import sample
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import  lit , col,split
import numpy as np
from numpy.linalg import norm





# Expects a list of vectors and returns the norm 
def get_norm(vector):
    float_casted_vector = [float(i) for i in vector]
    norm_value = norm(np.array(float_casted_vector))
    return float(norm_value)
   

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

# The function returns the distance based on the cosine similarity 
def add_dist_from_cos_sim(row):
    f1 = row[0][0]
    f2 = row[1][0]
    n1 = row[0][2]
    n2 = row[1][2]
    i = row[0][1]
    j = row[1][1]
    dot = float(f1.dot(f2))
    denom = n1 * n2
    cos_sim = -1
    if denom != 0.0:
        cos_sim = float( dot / float(denom))
    return (i , j , 1 - cos_sim)
   
# The function returns the pair-wise distance between each row in the provided embeddings 
def get_pair_wise_distance_rdd(embeddings_rdd):
        pair_wise_rdd = embeddings_rdd.cartesian(embeddings_rdd)
        pair_wise_distance_rdd = pair_wise_rdd.map(lambda row: add_dist_from_cos_sim(row) )
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

def formulate_all_to_all_dist_plus_initial_assignment_df(input_row , to_be_replaced_medoid):
    row = {}
    row['dist'] = input_row[1][0][2]
    row['j'] = input_row[1][0][1]
    row['first'] = {'dist' : input_row[1][1][2][0] , 'j' : input_row[1][1][2][1] }
    row['second'] = {'dist' : input_row[1][1][1][0] , 'j' : input_row[1][1][1][1] }
    row['removal_loss'] = input_row[1][1][3]
    
    # case where the new distance is the better than the actual assigned
    if(row['dist'] < row['first']['dist']):
        return (row['j'],row['dist'])
    
    # Case that the new distance is not the closest, but the nearest medoid is the to be replaced medoid
    elif(row['first']['j'] == to_be_replaced_medoid ):
        # If it is less than the second, then choose it 
        if(row['dist'] < row['second']['dist'] and row['dist'] != row['first']['dist']):
            return (row['j'] , row['dist'])
        else:
            # Otherwise, the second is the best option
            return (row['j'] , row['second']['dist'])
    else:
        return(row['j'] , row['first']['dist'])
  
  
if __name__ == "__main__":
    # ---------------------------- Initializations ------------------------------- #
    spark = SparkSession.builder.appName("pam").getOrCreate()
    sc = SparkContext.getOrCreate()
    k = 50
    filePath = "embeddings/input/100_sampled_embeddings.csv"
    
    #--------------------------- UDFS ----------------------------------------#
    udf_get_norm = F.udf(get_norm, FloatType())
    udf_dot_product = F.udf(lambda x,y: float(x.dot(y)), FloatType())

    # --------------------------- Cleaning and formatting --------------------- #
    # This function will read the data from the file and return a dataframe in the following format
    # features         | string_row_id
    # [0.1,0.2,....]  | string_id
    # [0.4,....... ]  | string_Id
    embeddings_df_string_id = formulate_embeddings(filePath , 'csv' , spark)

    # Having 64 bit ids is better than having string id with more consumed space, so dropping the string row id
    # embeddings_df_string_id = embeddings_df_string_id.withColumn("row_id", monotonically_increasing_id())
    # remove the string_row_id for better storage consumption 
    embeddings_df = embeddings_df_string_id.drop("string_row_id")
    # The following block of code will add the norm to each embedding and adding an id for each embedding
    embeddings_df = embeddings_df.withColumn("norm", udf_get_norm(col('features')))
    embeddings_rdd = embeddings_df.rdd
    embeddings_rdd = embeddings_rdd.zipWithIndex()
    embeddings_rdd = embeddings_rdd.map(lambda row: (row[0][0] , row[1] , row[0][1]))
    # embeddings_rdd.cache()
    
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
    # Caching the data here will be beneficial later for the fast access since this rdd will always be our reference to get the distances
    pair_wise_distance_rdd.cache()
    
    # --------------------------- Initialize k points   --------------------------#
    # First we need to get all the row_ids generated 
    embeddings_ids_list = embeddings_rdd.map(lambda row : row[1]).collect()
    #  Either use the random initializations or the k_meansinit initializations 
    k_medoids_list = get_random_medoids_index(embeddings_ids_list,k)
    #  Comment the above line and uncomment the below line, if the k_means_init is wanted
    # k_medoids_list = k_means_init(pair_wise_distance_df,embeddings_ids_list, k)
    
    k_medoids_set = {medoid for medoid in k_medoids_list}
 
    
    # -------------------------- Start iterating until convergence ---------------------- #
    # Maximum number of iterations to stop after 
    maxIteration = 500
    # Number of iterations taken for the algorithm to converge
    numIter = 0
    # The best min loss found so far over all the iterations
    outer_best_min_loss  = sys.maxsize
    # This will later have the best medoids that output the minimum min loss 
    best_medoids = k_medoids_set.copy()
    
    for _ in range(maxIteration):
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
        assignment_rdd.cache()
        # Joins the pair_wise distance with the assignment rdd to add the distance between each i and each other element in the dataset
        # +---+---+----------+---+----------------+---------------+------------+
        # |i  |j  |dist      |i  |first           |second         |removal_loss|
        # +---+---+----------+---+----------------+---------------+------------+
        # |0  |1  |0.6995637 |0  |{0.49680316, 17}|{0.5600259, 65}|0.06322271  |
        # |0  |2  |0.69839835|0  |{0.49680316, 17}|{0.5600259, 65}|0.06322271  |
        # |0  |3  |0.8194193 |0  |{0.49680316, 17}|{0.5600259, 65}|0.06322271  |
        assignment_rdd = assignment_rdd.keyBy(lambda row : (row[0]))
        all_to_all_dist_plus_initial_assignment_rdd = pair_wise_distance_rdd.join(assignment_rdd)  
        # +---+---+------------------- +---+-------------------+---+--------------------------------------+
        # |i  |j   |dist nearest       |1st medoid | 2nd nearest dist    |2nd medoid |removal_loss (2nd - 1st)|
        # +---+---+------------------- +---+-------------------+---+--------------------------------------+
        # |0  |1   |0.48289060592651367|46         |0.4968031644821167   |17         |0.013912558555603027|
        # |0  |2   |0.48289060592651367|46         |0.4968031644821167   |17         |0.013912558555603027|
        best_min_loss = sys.maxsize
        best_medoid_replacement = {}
        for medoid in k_medoids_set:
            # Create thee table with all the nearest distances for each added medoid
            # +---+---+------------------- +---+-------------------+---+------------
            # |_0 (added medoid)  |_1 (assigned_medoid)  |_2 (distance to nearest) |  
            # +---+---+------------------- +---+-------------------+---+------------
            # |0                  |1                     |0.48                     |
            # |0                  |2                     |0.41                     |
            all_assignments_rdd = all_to_all_dist_plus_initial_assignment_rdd.map(lambda row: formulate_all_to_all_dist_plus_initial_assignment_df(row , medoid))
            # We don't want to consider adding the current medoid, it is redundant since we know already 
            all_assignments_rdd = all_assignments_rdd.filter(lambda row: row[0] not in k_medoids_set)
            all_assignments_rdd = all_assignments_rdd.keyBy(lambda row: row[0])
            # For each to be replaced medoid, sum the losses 
            total_loss_summation = all_assignments_rdd.reduceByKey(lambda accum , row:  ( row[0] , row[1] + accum[1] ))
            # Finally, reduce to find the best replacement which has the minimum loss
            min_total_loss_record = total_loss_summation.reduce(lambda accum , row :  accum  if accum[1][1] < row[1][1] else row)
            if(min_total_loss_record[1][1] < best_min_loss ):
                best_min_loss = min_total_loss_record[1][1]
                best_medoid_replacement = {'to_be_replaced_medoid' : medoid , 'replacement_medoid': min_total_loss_record[0]}
        
        print("----- Current total loss :- ",best_min_loss  )
        # If found a new min, remove the to be replaced one with the current 
        if(best_min_loss < outer_best_min_loss):
            outer_best_min_loss = best_min_loss 
            k_medoids_set.remove(best_medoid_replacement['to_be_replaced_medoid'])
            k_medoids_set.add(best_medoid_replacement['replacement_medoid']) 
            best_medoids = k_medoids_set.copy()
        else:
            break
        
        assignment_rdd.unpersist()
        t1 = time.time()
        print("total time for iteration " , numIter , " is :", t1 - t0)

    
            
    print("Total number of iterations needed for convergence :- " , numIter)
    print("The final selected medoids :- ",  best_medoids)
    # Get the total loss for the best selected medoids 
    print("final total loss for the selected medoids  :- ", outer_best_min_loss)
    
    

    
    

    
    
    
    