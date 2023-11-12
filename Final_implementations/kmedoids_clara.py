import pyspark
import random
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col,split,monotonically_increasing_id , broadcast
import numpy as np
from numpy.linalg import norm
from pyspark.sql.functions import col
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

# Return the distance between the vectors based on the cosine similarity
def get_pair_wise_distance(embeddings_df ):
    # Join the embeddings_df to itself to create a all to all mapping of each embeddings. 
    all_to_all_df = embeddings_df.alias("i").join(embeddings_df.alias("j"))
    # Add the column "dot" which is the dot product between the the features vector for each pair
    all_to_all_dot_df = all_to_all_df.select("*",udf_dot_product("i.features", "j.features").alias("dot"))
    # Add the column "dist" which is 1 - cosine similarity calculated from the norm and dot 
    all_to_all_dist_df = all_to_all_dot_df.withColumn("dist",1 - udf_get_cos_sim("i.norm", "j.norm","dot"))
    # Select only the required columns, ids and the distance based  
    all_to_all_dist_df = all_to_all_dist_df.select( col("i.row_id").alias("i"),col("j.row_id").alias("j") , "dist")
    return all_to_all_dist_df     


#  
# features         | row_id
# [0.1,0.2,....]  | string_id
# [0.4,....... ]  | string_Id
def formulate_embeddings(input_file_path ):
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


def get_second_min(accum , row):
    if(accum[1][0][1] == row[1][1][1]):
        return row
    if(row[1][0][1] == row[1][1][1]):
        return accum

    elif accum[1][0][2] < row[1][0][2]:
        return accum
    else:
        return row
   
    
def medoids_assignment_rdd(pair_wise_distance_rdd , k_medoids_set):

    assignments = pair_wise_distance_rdd.filter( lambda row : row[1][1] in k_medoids_set )
    min_assignments = assignments.reduceByKey(lambda accum ,row : accum if accum[2] < row[2] else row)
    assignments = assignments.join(min_assignments)
    assignments = assignments.keyBy(lambda row : row[0])
    
    final_assignments = assignments.reduceByKey(lambda accum ,row : get_second_min(accum , row))
    final_assignments = final_assignments.map(lambda row : (row[0] , (row[1][1][0][2] , row[1][1][0][1] ) , (row[1][1][1][2] , row[1][1][1][1] ) ,   row[1][1][0][2] - row[1][1][1][2] ) )
    return final_assignments


def get_loss(assignment_df):
    return assignment_df.groupBy().sum("first.dist").withColumnRenamed("sum(first.dist AS dist)", "loss").collect()[0]["loss"]


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
       
def get_cos_sim(row):
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

def get_pair_wise_distance_rdd(embeddings_df):
        embeddings_rdd = embeddings_df.rdd
        pair_wise_rdd = embeddings_rdd.cartesian(embeddings_rdd)

        pair_wise_distance_rdd = pair_wise_rdd.map(lambda row: get_cos_sim(row) )
        # pair_wise_distance_rdd = pair_wise_distance_rdd.map(lambda row: ( row[0] , row[1] , row[]) )
        return pair_wise_distance_rdd
    
def get_best_medoids_fast_pam(embeddings_df , k_medoids_set_passed):
    k_medoids_set = k_medoids_set_passed.copy()
    # k_medoids_set ={25769803777, 4, 8589934602, 8589934604, 8589934605, 25769803790, 25769803791, 20, 25769803796, 8589934613, 8589934615, 24, 8589934619, 25769803807, 25769803809, 35, 8589934633, 25769803817, 34359738409, 17179869227, 34359738413, 8589934639, 52, 34359738422, 17179869239, 17179869240, 57, 8589934647, 25769803835, 34359738423, 25769803837, 25769803841, 25769803843, 8589934664, 73, 8589934666, 34359738443, 8589934667, 25769803853, 8589934669, 75, 8589934672, 17179869264, 17179869261, 17179869267, 86, 8589934678, 88, 17179869273, 34359738461, 17179869280, 25769803872, 98, 25769803875, 8589934691, 17179869282, 17179869286, 8589934697, 34359738474, 25769803883, 25769803886, 34359738486, 34359738492, 133, 8589934727, 17179869324, 25769803919, 144, 147, 149, 25769803931, 25769803934, 158, 8589934751, 25769803937, 8589934754, 17179869348, 17179869352, 25769803945, 169, 25769803953, 25769803957, 17179869367, 25769803960, 17179869371, 8589934780, 188, 17179869372, 17179869373, 17179869376, 8589934783, 25769803972, 17179869380, 17179869384, 8589934794, 25769803981, 25769803982, 25769803984, 25769803985, 213}
    #---------------------------  Create the pair wise distance matrix --------------------------- #
    # This is the most expensive operation in the algorithm.
    # It basically cross joins the dataframe on itself to create a single dataframe that has the distances between each row and the other
    # The following format will be produced 
    # i  | j | dist
    # 0  | 0 | 0
    # 0  | 1 | 0.012
    # 0  | 2 | 0.006
    # 1  | 0 | 0.012
    # 1  | 1 | 0
    pair_wise_distance_rdd = get_pair_wise_distance_rdd(embeddings_df)
    pair_wise_distance_rdd = pair_wise_distance_rdd.keyBy(lambda row: (row[0]))
    # pair_wise_distance_rdd = pair_wise_distance_rdd.partitionBy(20, lambda row: row[0])

    pair_wise_distance_rdd.cache()
    
    # Caching the data here will be beneficial later for the fast access since this dataframe will always be our reference to get the distances
    # -------------------------- Start iterating until convergence ---------------------- #
    # Maximum number of iterations to stop after 
    maxIteration = 10
    # Number of iterations taken for the algorithm to converge
    numIter = 0
    # This will have the total removal loss value over the iterations
    removal_loss_values = []
    # This will have the trace of how medoids change over the iterations 
    medoids_values_considered = []
    # k_medoids_set = { 5, 7 }

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
    pair_wise_distance_rdd.unpersist()
    return k_medoids_set
    # Get the total loss for the best selected medoids 
    # assignment_rdd = medoids_assignment_rdd(pair_wise_distance_rdd, k_medoids_set)
    # loss_value =  get_loss_rdd(assignment_rdd) 
    # print("final total loss for the selected medoids  :- ", loss_value)
    



def get_loss_rdd(assignment_rdd):
    final_value =  assignment_rdd.reduce(lambda accum , row: (accum[0] , accum[1] , (accum[2][0] + row[2][0] , accum[2][1]) , accum[3] , ))
    return final_value[2][0]
    
def total_loss_from_medoids_assignments(embeddings_df , k_medoids_set, k_medoids_df):
    embeddings_df_without_medoids_df = embeddings_df.filter(~col('row_id').isin(list(k_medoids_set)))
    joined_df = embeddings_df_without_medoids_df.crossJoin(broadcast(k_medoids_df)) 
    joined_rdd = joined_df.rdd
    pair_wise_distance_rdd_internal = joined_rdd.map(lambda row: get_cos_sim(row) )
    pair_wise_distance_rdd_internal = pair_wise_distance_rdd_internal.keyBy(lambda row: (row[0]))
    assignment_rdd = medoids_assignment_rdd(pair_wise_distance_rdd_internal, k_medoids_set)
    loss_value =  get_loss_rdd(assignment_rdd) 
    return loss_value
    
  
    
if __name__ == "__main__":
    # ---------------------------- Initializations ------------------------------- #
    spark = SparkSession.builder.appName("kmedoids").getOrCreate()
    sc = SparkContext.getOrCreate()
    k = 10
    originalN = 1000
    filePath = "embeddings/input/1000_sampled_embeddings.csv"
    sampledN =  40 + (2*(k)) 
    if((sampledN + 10 )>= originalN or sampledN > 10000):
        sys.exit(" N is very small or k is very Large, use FasterPAM without sampling")
        
    
    #--------------------------- UDFS ----------------------------------------#
    udf_get_cos_sim = F.udf(get_cosine_similarity_given_norms, FloatType())
    udf_get_norm = F.udf(get_norm, FloatType())
    udf_dot_product = F.udf(lambda x,y: float(x.dot(y)), FloatType())

    # --------------------------- Cleaning and formatting --------------------- #
    # This function will read the data from the file and return a dataframe in the following format
    # features         | string_row_id
    # [0.1,0.2,....]  | string_id
    # [0.4,....... ]  | string_Id
    embeddings_df_string_id = formulate_embeddings(filePath)
    # Having 64 bit ids is better than having string id with more consumed space, so dropping the string row id
    embeddings_df_string_id = embeddings_df_string_id.withColumn("row_id", monotonically_increasing_id())
    # remove the string_row_id for better storag consumption 
    embeddings_df = embeddings_df_string_id.drop("string_row_id")
    original_embeddings_df = embeddings_df.withColumn("norm", udf_get_norm(col('features')))
    
    
    # We add 10 here because sampling of dataframe is not accurate, so we add more data such that we can limit later the exact number we want 
    sampleFraction = (sampledN + 10) / originalN
    #  initial medoid selections 
    print(sampleFraction , sampledN , originalN)
    print("---------------------------------")
    embeddings_df = original_embeddings_df.sample(fraction=sampleFraction).limit(sampledN)
    # --------------------------- Initialize k points   --------------------------#
    # First we need to get all the row_ids generated 
    unique_embeddings_df = embeddings_df.select("row_id").distinct()
    unique_embeddings_ids = unique_embeddings_df.collect()
    embeddings_ids_list = []
    for row in unique_embeddings_ids:
        embeddings_ids_list.append(row["row_id"])
    
    #  Either use the random initializations or the k_meansinit initializations 
    k_medoids_list = get_random_medoids_index(embeddings_ids_list,k)
    # Get random k-medoids and union it with the sampled embeddings 
    #  Comment the above line and uncomment the below line, if the k_means_init is wanted
    # k_medoids_list = k_means_init(pair_wise_distance_df,embeddings_ids_list, k)
    
    k_medoids_set = {medoid for medoid in k_medoids_list}
    print(k_medoids_set)
    # --------------------------- Sampling iterations ------------------------------
    max_sampling_iterations = 5
    min_total_loss = sys.maxsize
    best_medoids = k_medoids_set.copy()
    for i in range(max_sampling_iterations):
        new_k_medoids_set = get_best_medoids_fast_pam(embeddings_df , k_medoids_set)
        k_medoids_set = new_k_medoids_set.copy()
        k_medoids_df = original_embeddings_df.filter(col('row_id').isin(list(k_medoids_set)))
        total_loss = total_loss_from_medoids_assignments(original_embeddings_df ,k_medoids_set, k_medoids_df)

        print("final total loss for the selected medoids  :- ", total_loss)
        if(total_loss < min_total_loss):
            min_total_loss = total_loss
            best_medoids = k_medoids_set.copy()
        # Sample again the embeddings
        embeddings_df = original_embeddings_df.sample(fraction=sampleFraction).limit(sampledN)
        embeddings_df.show()

        # remove the current k_medoids from the set to add them once later
        embeddings_df = embeddings_df.filter(~col('row_id').isin(list(k_medoids_set)))
        embeddings_without_medoids_count = embeddings_df.count()
        if(embeddings_without_medoids_count == sampledN):
            # None of the medoids exist in the new dataframe. Remove the last k embeddings and add the medoids 
            to_be_removed = k
            embeddings_df = embeddings_df.limit(embeddings_without_medoids_count - to_be_removed)
        elif(embeddings_without_medoids_count < sampledN):
            # Some of the medoids are removed. remove the rest embeddings and add all the medoids 
            to_be_removed = k - (sampledN - embeddings_without_medoids_count)
            embeddings_df = embeddings_df.limit(embeddings_without_medoids_count - to_be_removed)

            
        print(embeddings_df.count())
        print("----2222322********************************")
        k_medoids_df.show()
        print(k_medoids_df.count())
        print("----33333333333333********************************")
        # Update the embeddings df wit the medoids and the new embeddings sampled to be used in the next iteration
        embeddings_df = embeddings_df.union(k_medoids_df)
        print(embeddings_df.count())
        print("----4444444********************************")
        print("Finished sampling iterations number :- " , i)
