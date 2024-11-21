import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
# from sklearn.metrics import  
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters
from scipy.spatial.distance import cdist
from cluster.kmeans import KMeans



k_means = KMeans(3, 'euclidean', 200 , 30)
def main(): 
    # Set random seed for reproducibility
    np.random.seed(65)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
   
    # Fit model
    k_means.fit(og_iris)

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    pred = k_means.predict(og_iris)
    mse = k_means.get_error()
    pred_centriod = k_means.get_centroids()
    
    # You can choose which scoring method you'd like to use here:
    
    mse = k_means.get_error() # i am using MSE
    
    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(og_iris, pred ,"model1", mse )
    
# Try different numbers of clusters
def elbow():
    inertia = []
    k_range = range(1, 8)
    np.random.seed(885)
    og_iris = np.array(load_iris().data)
    for k in k_range:
        k_elbow_means =  KMeans(k,'euclidean', 500, 1)
        k_elbow_means.fit(og_iris)
        inertia.append(k_elbow_means.get_error())
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Plot for KMeans Clustering')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(k_range)
    plt.grid()
    plt.show()

    
    # Plot the elbow plot
    ## use plt.show()
    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    
    """
    How many species of flowers are there: i belive that there are 3 species of flowers in this data set.
    
    Reasoning: the elbow plot levels off at 3 centroids. while the data apears to show only two distinct 
    clusters of data, this could be due to a overlap in charatarsics of the two of the species. 

    
    
    
    
    """

    
if __name__ == "__main__":
    main()
    #input("elbow?")
    elbow()
