import numpy as np
from sklearn.metrics import pairwise_distances

#save the pairwise distances somewhere so i dont have to calculate them for every run
def getOGdis(dataset):
  return pairwise_distances(dataset, metric='euclidean')

def writeToFile(data, filename):
    np.savetxt(filename, data, delimiter=',')


def minDimCalc(eps, delta, N):
    return np.ceil(4/(eps**2) * np.log(N/delta))

def getProjMat(dim, N):
    return np.random.choice([-1, 1], (dim, N))

def testRed(dim, redFac):
    return dim * redFac

def projectedVec(matrix, vector):
    return np.matmul(matrix, vector)

def splitSubsets(data):
  


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


import numpy as np
import time
import math

start = time.time()
X_train, y_train = load_mnist('/content/fashion', kind='train')

smalltest = X_train[:3000]
# N is the original dimension of the data
# eps is the reduction factor
# delta is the confidence interval
N = len(X_train[0])
eps = 0.25
delta = 0.1
setsize = 200

#subsetted run


def run():
    distances =[]
    dim = minDimCalc(eps, delta, N)
    dim = dim.astype(int)
    projMat = getProjMat(dim, N)
    for vec in smalltest:
        projVec = (1/math.sqrt(dim)) * projectedVec(projMat, vec)
        distances.append(projVec)
    
    assert(len(distances) == len(smalltest))
    assert(len(distances[0])== len(smalltest[0]))
    writeToFile(getOGdis(distances), '/content/distances' + str(dim) + '.csv')

run()
writeToFile(getOGdis(smalltest), '/content/ogdistances.csv')

import pandas as pd
import numpy as np

data = pd.read_csv("ogdistances.csv", header = None)
sketched = pd.read_csv("distances574.csv", header = None)


data.replace(0, np.nan, inplace = True)
data.loc[:,:] = data.loc[:,:]**2
sketched.replace(0, np.nan, inplace = True)
sketched.loc[:,:] = sketched.loc[:,:]**2

data.loc[:,:] = sketched.loc[:,:].div(data.loc[:,:])
print(data.iat[1,0])

data.to_csv("sketchchange.csv")
#print(data)

import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv("sketchchange.csv")
data = data.iloc[: , 1:]
data = data.values.flatten()
data = data[~np.isnan(data)]

box = plt.boxplot(data)
print(stats.describe(data))
plt.show()
for i in box['fliers']:
  print(i.get_data()[1])
