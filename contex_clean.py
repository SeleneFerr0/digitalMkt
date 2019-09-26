# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:25:55 2019

@author: seleneferro

context cleaning
"""

def fn(x):
    if x.is lower():
        return x.upper()
    elif x.is upper():
        return x.lower()
    else:
        return x



def quickSort(arg):
    if(arg==[]):
         return []
    return quickSort([i for i in arg[1:] if i<=arg[0]])+[arg[0]]+quickSort([i for i in arg[1:] if i>arg[0]])


def quicksort(array, left, right):
    if left >= right:
        return
    low = left
    high = right
    key = array[low]

    while left < right:
        
        while left < right and array[right] > key:
            right -= 1
        
        array[left] = array[right]

        
        while left < right and array[left] <= key:
            left += 1
        
        array[right] = array[left]

    
    array[right] = key

    
    quicksort(array, low, left - 1)
    quicksort(array, left + 1, high)


def quicksort(array, l, r):

    if l < r:
        q = partition(array, l, r)
        quick_sort(array, l, q - 1)
        quick_sort(array, q + 1, r)

def partition(array, l, r):
    x = array[r]
    i = l - 1


    for j in range(l, r):
        if array[j] <= x:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i+1]
    return i + 1

#array = [11,22,8,23,7,33,13,28,66,777]
#print("Quick Sort: ")
#quicksort(array,0,len(array)-1)
#print(array)
#[7, 8, 11, 13, 22, 23, 28, 33, 66, 777]
    
def f1():    
    with open("D:\\r\\2.txt","r") as f:
        for s in f:
            l=s.rsplit ()
            #align left
            t='{0:<5} {1:<7} {2}'.format(l[0],l[1],l[2])    
            print(str(t))
f1()




import re
reg=["[a-z]","[A-Z]","\d","[^\da-zA-Z]"]

for s in reg:    
    rega=re.compile(s)
    s=re.findall(rega,a)
    print("".join(s))



''' k-means clusters'''
for doc_v in docs_matrix:
    if doc_v.sum() == 0:
        doc_v = doc_v / 1
    else:
        doc_v = doc_v / (doc_v.sum())
    tfidf = np.dot(docs_matrix,idf)
    return names,tfidf

def gen_sim(A,B):
    num = float(np.dot(A,B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
    
def kMeans(dataSet, k, distMeas=gen_sim, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    counter = 0
    while counter <= 50:
        counter += 1
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; 
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; 
                    minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment