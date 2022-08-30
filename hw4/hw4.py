import csv
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


def load_data(filepath):
    list_dict=list()
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            list_dict.append(row)
    
    return list_dict

def calc_features(row):
    array=np.append(np.zeros(shape=(6,0)),[int(row['Attack']),int(row['Sp. Atk']),int(row['Speed']),int(row['Defense']),int(row['Sp. Def']),int(row['HP'])])
    
    return array.astype(np.int)



def hac(features):
    #distance matrix and dictonary of points
    features_length=len(features)
    distanceMatrix = np.empty([features_length, features_length])
    datapoints= dict()

    #creating a dictonary of all the clusters 
    for label in range(features_length):
        datapoints[label] = [label]
   
    #distance matrix 
    for i in range(features_length):
        for j in range(features_length): 
            if i==j:
                continue
            distanceMatrix[i][j] = np.linalg.norm(features[i] - features[j])

    #empty array intialized w/stacking 
    Z = np.empty([0,4])
    
    #to append new clusters to the dictonary for further clustering
    index=features_length
    
    for n in range(features_length - 1):
        minimum = float('inf')
        
        z_0 = 0
        z_1 = 0
        #iterating through datapoints 
        for i in datapoints:
            for j in datapoints:
                if i == j: 
                    continue;
                   
                clust_i = datapoints[i]
                clust_j = datapoints[j]
                maximum = float('-inf')
                
                #finding the maximum among the elements selected
                for ci in clust_i:
                    for cj in clust_j:
                        if maximum < distanceMatrix[ci][cj]:
                            maximum = distanceMatrix[ci][cj]
                
                #setting the new minimum after each pass 
                if minimum > maximum: 
                    minimum = maximum
                    z_0 =  i
                    z_1 = j
                    
                #for when there are identical distances, tiebreaking by cluster #
                elif maximum == minimum:
                    if z_0 > i:
                        z_0 =  i
                        z_1 = j
                    elif z_0 == i:
                        if z_1 > j:
                            z_1 = j
       
        #creating new cluster
        datapoints[index] = datapoints[z_0] + datapoints[z_1]
        
        #removing old clusters
        del datapoints[z_0]
        del datapoints[z_1]
       
        #appending to the stack 
        Z = np.vstack([Z, [z_0, z_1, minimum, len(datapoints[index])]])
        
        index += 1
    return Z

def imshow_hac(Z):
    plt.figure()
    hierarchy.dendrogram(Z)
    plt.show()

