import pyspark
import random
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import FloatType,DoubleType
from pyspark.sql.functions import col
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, explode, least, when,concat_ws,array, min,round
from pyspark.ml.linalg import VectorUDT, DenseVector
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
from numpy.linalg import norm
from pyspark.sql.window import Window
from pyspark.sql.types import StructType,StructField, StringType
from pyspark.sql.functions import col
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

def cosine_similarity(X,Y,):
    denom = X.norm(2) * Y.norm(2)
    if denom == 0.0:
        return -1.0
    else:
        return float(X.dot(Y) / float(denom))

def cosine_similarity_given_norms(XNorm, YNorm, dot ):
    denom = XNorm * YNorm
    if denom == 0.0:
        return -1.0
    else:
        return float(dot / float(denom))

def get_one_to_many_cos_sim(random_index , cluster_name, indexed_df):
    nRows = indexed_df.count()
    # Index of the vector "one-to-many" vector to compute the cosine sim
    vector_df = indexed_df.filter(indexed_df["row_id"]==random_index).withColumn('new_column', F.explode(F.array([F.array(F.lit(e)) for e in list(range(0,indexed_df.count()))]))).drop("new_column")
    # Reindex dataframe
    vector_df = vector_df.withColumn("row_id", monotonically_increasing_id())
    # Rename column to avoid name conflicts
    vector_df = vector_df.withColumnRenamed("features","vector")
    # Join based on the index
    joined_df = indexed_df.join(vector_df, ("row_id"))
    # Define cosine_similarity as UDF function so that we can apply it on the dataframe columns
    udf_dot_prod=udf((lambda a,b: cosine_similarity(a,b)), FloatType())
    # Calculate cosine similarity between vectors in "features" column and the singl vector in "feature column"

    dot_prod_df = joined_df.withColumn(cluster_name, F.lit(udf_dot_prod(joined_df["features"],joined_df["vector"]))).select("row_id",cluster_name)
    return dot_prod_df

def assign_points_to_clusters(cos_sim_df, clusters_col_names_list):
    # this will add a new column "min" which will have the minimum value for each row from the clustesr
    cos_sim_with_min_col_df = cos_sim_df.withColumn("min", least(*clusters_col_names_list))
    
    # This will add a new column "assigned_cluster" which will have the cluster name (column name) which corrosponds to the one that has the minimum value
    clustered_df = cos_sim_with_min_col_df.select("*",concat_ws(";",array([
       when(col(c)==col("min") ,c).otherwise(None)
       for c in clusters_col_names_list
     ])).alias("assigned_cluster") )

    # This will calculate the summation of the values for each cluster
    td_score = clustered_df.groupBy("assigned_cluster").sum("min")
    return clustered_df, td_score


def calculate_distance_matrix(df_vect2):
    mat = RowMatrix(df_vect2.rdd.map(list))
    # Transpose function does not work on RowMatrices
    # Convert to Coordinate Matrix in order to be able to transpose it.
    cm = CoordinateMatrix(
        mat.rows.zipWithIndex().flatMap(
            lambda x: [MatrixEntry(x[1], j, v) for j, v in enumerate(x[0])]
        )
    )
    print("===================================================")
    print("===================================================")
    print("===================================================")
    print(" FINISHED CM ")
    print("===================================================")
    print("===================================================")   
    print("===================================================")
    sims = cm.transpose().toRowMatrix().columnSimilarities()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++=")

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    return sims
    
def convert_coordinate_matrix_to_df(sims,spark):
    # columns = ['i', 'j', 'sim']
    # vals = sims.entries.map(lambda e: (e.i, e.j, e.value)).collect()
    # dfsim = spark.createDataFrame(vals, columns)
    # df_dist = dfsim.withColumn("dist",1 - dfsim.sim).drop("sim")
    
    # return df_dist
    df_sim = spark.createDataFrame(sims.entries)
    df_dist = df_sim.withColumn("dist",1 - df_sim.value).drop("value")
    return df_dist

def create_empty_dataframe():
    schema = StructType([
        StructField('medoid_index', StringType(), True),
        StructField('td_score', IntegerType(), True)
    ])
    emptyRDD = spark.sparkContext.emptyRDD()
    return spark.createDataFrame(emptyRDD, schema)

def get_some_to_all_cosine_matrix(medoids_indices, all_indices,many_to_many_df):
    
    non_medoids_indexes = np.asarray(all_indices)[np.in1d(all_indices, medoids_indices, invert=True)].tolist()
    non_medoids_df = spark.createDataFrame(non_medoids_indexes, IntegerType()).toDF("index")
    cos_sim_df = non_medoids_df.alias('non_medoids_df2')
    # Empty dataframe to store the results
    for medoid_ind in medoids_indices:
        one_to_many_i = many_to_many_df.filter(many_to_many_df.i == medoid_ind).withColumn("index", many_to_many_df.j).drop(*["i","j"])
        one_to_many_j = many_to_many_df.filter(many_to_many_df.j == medoid_ind).withColumn("index", many_to_many_df.i).drop(*["i","j"])
        one_to_many = one_to_many_i.union(one_to_many_j).withColumnRenamed("dist", str(medoid_ind))
        # Dataframe with rows containing all non-medoid datapoints and columns containing medoids 
        cos_sim_df = cos_sim_df.join(one_to_many, ["index"], "inner")
    return cos_sim_df

def get_medoid_of_points(many_to_many_df, indices):
    i_grouped_df = many_to_many_df.filter(col("i").isin(indices)).groupBy("i").sum("dist").withColumnRenamed("i", "index")
    j_grouped_df = many_to_many_df.filter(col("j").isin(indices)).groupBy("j").sum("dist").withColumnRenamed("j","index")
    i_j_grouped_df = i_grouped_df.union(j_grouped_df)
    distances_df= i_j_grouped_df.groupBy("index").sum("sum(dist)").withColumnRenamed("sum(sum(dist))", "sum")
    min_distance_df = distances_df.select(min("sum")).withColumnRenamed("min(sum)","min")
    min_distance_df_rounded  = min_distance_df.withColumn("min", round(col("min"),5))
    min_value = min_distance_df_rounded.collect()[0][0]
    medoid_index_df = distances_df.withColumn("sum", round(col("sum"),5)).filter(col('sum') == min_value)
    medoid_index = medoid_index_df.collect()[0][0]
    return medoid_index   

def norm_2_func(features):
    features = [float(i) for i in features]
    x = norm(np.array(features))
    return float(x)
    # return [float(i) for i in features/np.linalg.norm(features, 2)]
    # you can also use
    # return list(map(float, features/np.linalg.norm(features, 2)))
if __name__ == "__main__":
    spark = SparkSession.builder.appName("test").getOrCreate()
    sc = SparkContext.getOrCreate()

    df = spark.read.csv("embeddings/input/10_sampled_embeddings.csv")
    df2 = df.select([col(column).cast(FloatType()) for column in df.columns[1:]])
    embeddingCols = df2.columns
    assembler = VectorAssembler(inputCols=embeddingCols, outputCol="features")
    output = assembler.setHandleInvalid("skip").transform(df2)
    #print("done assembler") 
    df_vect2 = output.select("features")
    indexed_df = df_vect2.withColumn("row_id", monotonically_increasing_id())
    norm_2_udf = F.udf(norm_2_func, FloatType())
    indexed_df_norm = indexed_df.withColumn("norm", norm_2_udf(F.col('features')))

    indexed_df_norm.show()
    dot_udf = F.udf(lambda x,y: float(x.dot(y)), FloatType())
    cos_sim_udf = F.udf(cosine_similarity_given_norms, FloatType())

    indexed_df_norm.alias("i").join(indexed_df_norm.alias("j"), col("i.row_id") < col("j.row_id")).select(
        col("i.row_id").alias("i"), 
        col("j.row_id").alias("j"), 
        dot_udf("i.features", "j.features").alias("dot")).withColumn("sim",cos_sim_udf("i.norm", "j.norm","dot") ).show()
   
    # The number of embeddings will be changed after skipping the invalid rows 
    embeddings_length = indexed_df.count()
    # k is the number or randomly selected points (medoids) to generate the one to many cosine similarity 
    k = 3
    random_medoid_indices = random.sample(range(embeddings_length), k)
   

    # Get many-to-many distances Matrix.
    many_to_many_coordinateMatrix = calculate_distance_matrix(df_vect2)
    many_to_many_df = convert_coordinate_matrix_to_df(many_to_many_coordinateMatrix , spark)
   
    df_size = df_vect2.count()

    all_indices = np.arange(df_size)  
    best_td_score = 100000000
    # get_medoid_of_points(many_to_many_df,[1,4,5,2])
    # calculate the some to many cosine matrix given the initially random indices 
    
   
    medoids_printing_array = [random_medoid_indices]
    values_printing_array =[]
    for i in range(3):
  
        cos_sim_df = get_some_to_all_cosine_matrix(random_medoid_indices ,all_indices,many_to_many_df)
       
        clustered_df, td_score = assign_points_to_clusters(cos_sim_df , list(map(str, random_medoid_indices)))

        # Get Minimum td_score
        total_td_score = td_score.groupBy().sum().collect()[0][0]
        if(total_td_score < best_td_score):
            best_td_score = total_td_score
        values_printing_array.append(total_td_score)
        new_medoid_indices = []
        for medoid_index in range(len(random_medoid_indices)):
            medoid_index = random_medoid_indices[medoid_index]
            indices_in_cluster_df = clustered_df.filter(col("assigned_cluster") == str(medoid_index)).select("index")
            indices_in_cluster = indices_in_cluster_df.rdd.map(lambda x: x[0]).collect()
            if len(indices_in_cluster) != 0:
               
                new_medoid_index = get_medoid_of_points(many_to_many_df, indices_in_cluster)
                
                new_medoid_indices.append(new_medoid_index)
            else:
                new_medoid_indices.append(medoid_index)

        random_medoid_indices = new_medoid_indices.copy()
        medoids_printing_array.append(random_medoid_indices)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(medoids_printing_array)
    print(values_printing_array)
    print(best_td_score)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

