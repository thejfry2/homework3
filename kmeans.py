import numpy as np
from scipy.spatial.distance import cdist # read documentation
import random
#from scipy.spatial.distance import euclidean
class KMeans():
    def __init__(self, k: int, metric:str, max_iter: int, tol: float):
        """
        Args:
            k (int): 
                The number of centroids you have specified. 
                (number of centroids you think there are)
                
            metric (str): 
                The way you are evaluating the distance between the centroid and data points.
                (euclidean)
                
            max_iter (int): 
                The number of times you want your KMeans algorithm to loop and tune itself.
                
            tol (float): 
                The value you specify to stop iterating because the update to the centroid 
                position is too small (below your "tolerance").
        """ 
        # In the following 4 lines, please initialize your arguments
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        
        # In the following 2 lines, you will need to initialize 1) centroid, 2) error (set as numpy infinity)
        self.centriod = None #there can be diffrent numbers of centriods stored here
        self.error = np.inf

    
    def fit(self, matrix: np.ndarray):
        """
        This method takes in a matrix of features and attempts to fit your created KMeans algorithm
        onto the provided data 
        
        Args:
            matrix (np.ndarray): 
                This will be a 2D matrix where rows are your observation and columns are features.
                
                Observation meaning, the specific flower observed.
                Features meaning, the features the flower has: Petal width, length and sepal length width 
        """
    


        # In the line below, you need to randomly select where the centroid's positions will be.
        # Also set your initialized centroid to be your random centroid position
        ## find method in numpy that picks random value for centriod in feture space, this is in readme
        ## set initalized centriod to this value

        self.centriod = matrix[np.random.choice(matrix.shape[0],self.k, replace = False)]

        
        # In the line below, calculate the first distance between your randomly selected centroid positions
        # and the data points
        distances = cdist(self.centriod, matrix, metric= self.metric)



        # In the lines below, Create a for loop to keep assigning data points to clusters, updating centroids, 
        # calculating distance and error until the iteration limit you set is reached
    
       
       
        # ref value to initialise them in the loop
        iteration = 0
        prev_inertia = 0 
        while iteration < self.max_iter:
            
                
            # Within the loop, find the each data point's closest centroid 
            ## used funtion returns distances to centriods
    
            distances = cdist(self.centriod, matrix, metric= self.metric)
            closest_clusters = np.argmin(distances, axis=0) # this should make a list of numbers represtending which cluster that data point is closest to
            
            # Within the loop, go through each centroid and update the position.
            # Essentially, you calculate the mean of all data points assigned to a specific cluster. This becomes the new position for the centroid
            
            inertia = 0
            for cluster in range(self.k):
                clustered_points = matrix[closest_clusters == cluster] # bool mask over matrix, this pulls only data points assocated with cluster 
                
                if clustered_points.shape[0] > 0:  # Check if the cluster has points, this was a problem with large K values
                    self.centriod[cluster] = np.mean(clustered_points, axis = 0) # use numpy.mean 
                
                
            
            # Within the loop, calculate distance of data point to centroid then calculate MSE or SSE (inertia)
                  
                    squared_distances = np.sum((clustered_points - self.centriod[cluster])**2, axis=1)
                    inertia += np.sum(squared_distances)
                    

            # Step 4: Calculate the Mean Squared Error
                    mse = np.mean(squared_distances)
                    self.error = inertia
            # Break if the error is less than the tolerance you've set
            # Within the loop, compare your previous error and the current error
            # Set your error as calculated inertia here before you break!
            

            
            print("iteration", iteration)
            print("centriods",  self.centriod)
            print("mse", mse)
            print('inertia', inertia)
            
            if inertia < self.tol:
                self.error = inertia
                break
            
            if inertia == prev_inertia: #checking for finality 
                self.error = inertia
                break
            prev_inertia = inertia
            iteration += 1 
            self.error = inertia
                
            
    
    
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Predicts which cluster each observation belongs to.
        Args:
            matrix (np.ndarray): 
                

        Returns:
            np.ndarray: 
                An array/list of predictions will be returned.
        """
        # In the line below, return data point's assignment 
        distances = cdist(self.centriod, matrix, metric= self.metric)
        closest_clusters = np.argmin(distances, axis=0)
        print(closest_clusters)
        return closest_clusters
    
        pass
    
    def get_error(self) -> float:
        """
        The inertia of your KMeans model will be returned

        Returns:
            float: 
                inertia of your fit

        """
        print("this is the error look at me im mister error", self.error)
        return self.error 
        pass
    
    
    def get_centroids(self) -> np.ndarray:
    
        """
        Your centroid positions will be returned. 
        """
        # In the line below, return centroid location
        print("centriods got", self.centriod)
        return self.centriod
        pass
        
        

