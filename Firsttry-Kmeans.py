import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import random

wareHouse = ((random.randint(18882,19326))/1000,(random.randint(72771,73008))/1000)
numberOfScvs  = 3
# volumeCapacityofScvs = [3000,2000,1000]
weightCapacityofScvs = [3000,2000,1000]


weight=[]
lat=[] # day of delivery
long=[] # day of order received
pickup_data=[]
number_pickup_data=100
for i in range(0,number_pickup_data):
    weight.append(random.randint(1,100))
    lat.append((random.randint(18882,19326))/1000)
    long.append((random.randint(72771,73008))/1000)
    pickup_data.append([weight[i],lat[i],long[i]])

def sortDay(val): 
    return val[0]  

pickup_data.sort(key=sortDay)
print(pickup_data[:][0])
print("pickup_data: Weight, Latitude, Longitude")
print('Total Weight:',sum(weight))
#print(*pickup_data, sep = "\n")

x=[]
y=[]
L=[]
w=[]
for entry in pickup_data:
    w.append(entry[0])
    x.append(entry[1])
    y.append(entry[2])
    
for i in range(0,len(x)):
    L.append([x[i],y[i]])
L=np.asarray(L)
#print(L)
#print(w)
#print(x)
#print(y)
distanceMatrix = np.sqrt(np.sum(np.square(np.tile(L,[L.shape[0],1,1])-np.swapaxes(np.tile(L,[L.shape[0],1,1]),0,1))
                                ,axis = 2))
# plt.scatter(x,y,s=w)
# plt.grid(b=True,which='both',axis='both')
# plt.xlabel("lat")
# plt.ylabel("long")

wcss =[]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(L)
    wcss.append(kmeans.inertia_)

# plt.plot(range(1,11),wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

class Cluster():
    def __init__(self,label,L=L,w=w,distanceMatrix=distanceMatrix,kmeans=kmeans):
        self.label = label
        self.filledCapacity = 0
#         self.distanceCovered = 0
        self.locationsAdded = []
        self.indexesAdded = []
        
        self.filter = [True if kmeans.labels_[i]==label else False for i,j in enumerate(L)]
        self.availableIndexes = np.array([i for i,j in enumerate(L) if kmeans.labels_[i]==label])
        
        self.availableLocations = np.compress(self.filter,L,axis=0)
        self.availableCargoCapacity = np.compress(self.filter,w)
#         self.requiredDistanceMatrix = np.compress(self.filter,np.compress(self.filter,distanceMatrix,axis = 0),axis = 1)
        
        self.order = np.argsort(np.square(self.availableLocations[:,0]-wareHouse[0])+np.square(self.availableLocations[:,1]-wareHouse[1]))
        self.inverseOrder = np.argsort(self.order)
        
        self.availableIndexes = self.availableIndexes[self.order]
        self.availableLocations = self.availableLocations[self.order]
        self.availableCargoCapacity = self.availableCargoCapacity[self.order]
#         self.requiredDistanceMatrix = self.requiredDistanceMatrix[self.order,:][:,self.order]
        
    def fillCapacity(self,capacity):
        while self.filledCapacity < capacity:
            self.filledCapacity += self.availableCargoCapacity[0]
            self.locationsAdded.append(self.availableLocations[0,:])
            self.indexesAdded.append(self.availableIndexes[0])
            
            self.availableCargoCapacity = self.availableCargoCapacity[1:]
            self.availableLocations = self.availableLocations[1:,:]
            self.availableIndexes = self.availableIndexes[1:]
            
        return self.indexesAdded

clusterList = {}
shipmentToBeAlloted = np.array([True for i,j in enumerate(L)])

# for scvNo in range(numberOfScvs):
    # clusterList[scvNo] = Cluster(scvNo)
    # ocuppiedShippments = clusterList[scvNo].fillCapacity(1000)
    # shipmentToBeAlloted[ocuppiedShippments] = False
    
# sum(shipmentToBeAlloted)

#print(distanceMatrix)

for u in range(2,9):
    kmeans = KMeans(n_clusters=u, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y=kmeans.fit_predict(L,sample_weight=w)

    #plt.scatter(L[:,0], L[:,1])
    #print(kmeans.labels_)
    #print(kmeans.labels_[0])
    plt.figure()
    for i,point in enumerate(L):
        x,y=point
        if kmeans.labels_[i]==0:
            clr='black'
        elif kmeans.labels_[i]==1:
            clr='orange'
        elif kmeans.labels_[i]==2:
            clr='magenta'
        elif kmeans.labels_[i]==3:
            clr='green'
        elif kmeans.labels_[i]==4:
            clr='blue'
        elif kmeans.labels_[i]==5:
            clr='cyan'
        plt.scatter(x,y,c=clr,s=w[i])

    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='red')
    plt.scatter(*wareHouse, s=300, c='cyan')
    plt.grid(b=True,which='both',axis='both')
    plt.show()
    continue