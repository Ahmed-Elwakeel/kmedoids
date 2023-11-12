# K-medoids (Spark)
The repo contains multiple variations of implementing k-medoids using Spark on N embeddings where each row contains 1024 or 1025 where the first row in the 1025 column should be a string ID, otherwise the 1024 are numbers between -1 and 1.
The `final_implementation` folder contains three variations of the algorithm (PAM , FastPAM , CLARA). 
PAM is the original algorithm and it has the slowest running time. 
FastPAM is faster when it comes to increasing k, the algorithm and explanation can be found here [1](https://www.sciencedirect.com/science/article/pii/S0306437921000557). 
CLARA is a sampling approach for implementing FastPAM that has a faster time complexity, but lower accuracy 
## Initialization 
* `file_path` : should contain a string that represents the path of the file of the embeddings in hdfs. It can accept both csv and parquet files
* `k` : a number represents the number of clusters needed
* `max_iteration` : the maximum number of iterations if not converged
## Build phase 
### Pairwise distance dataframe
PAM and FastPAM create a pairwise distance dataframe which is an O(N^2) in time complexity and space. 
This dataframe is cached in memory for faster access in each iteration instead of calculating it every time. 
In the case of having a huge dataset that exceeds the resources available in your instance, you will probably need to remove caching. 
The dataframe to be saved is 3 columns ( 64bit, 64bit, 32bit). To calculate how much space will be consumed for this, multiply (64 + 64 + 32) by the number of samples to know how many bits will it need.
### Initial selection of the medoids 
* **Random**: A random selection of the initial medoids which can be used by uncommenting the line that `calls get_random_medoids_index`
* **k-means++** : An initialization technique that depends on selecting the first medoid randomly and then selecting the rest of the medoids that are furthest from each other. Find more details [2] (https://neptune.ai/blog/k-means-clustering#:~:text=K%2Dmeans%2B%2B%20is%20a,as%20possible%20from%20one%20another)
Only choose one of them by uncommenting one and commenting the other

## Algorithm iterations (SWAP phase) [PAM and FastPAM[ 
PAM and FastPAM uses RDD and dataframe to distribute the processing work of swapping the current medoids with the non-medoids to find the best that produce the minimum total loss. 
PAM does this in an iterative way by swapping each medoid with non-medoid and check which one should be swapped. In FastPAM )please check the reference mentioned above), it calculates the removal loss between the initial nearest and second nearest medoid to get what best replacement that will introduce the lowest removal loss. 
FastPAM is faster then PAM as it doesn't have this (k) iterations. 
Total loss is calculated by summing the distance between each point and its assigned medoid. 


## CLARA
Clara samples the dataset into multiple subsets with a fixed size which is ( 40 + (2*(k)) ), which turns out to be a good sample size according to [1](https://www.sciencedirect.com/science/article/pii/S0306437921000557). 
Then runs FastPAM on each subsets, for each iteration,
* a set of medoids is selected from the result FastPAM on the subsample.
*  The total loss is calculated on the whole data set given the selected medoids
*  IF the total loss from the new medoids is the best so far, then use it for the next sample, until you found the best
*  Run FastPAM again on the new sample. 

You can control how many samples to control from `max_sampling_iterations` 
