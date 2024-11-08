# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KaNW5UV8HTmLkRyVpG4Fu-09C73FOWS6

**Setup environment**
"""

from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy import stats
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image
import statsmodels.api as sm
import cudf
import cupy as cp

#save the pairwise distances somewhere so i dont have to calculate them for every run
def getDis(vec1, vec2):
  return cp.linalg.norm(vec1-vec2)

def minDimCalc(eps, delta, N):
    return np.ceil(4/(eps**2) * np.log(N/delta))

def getProjMat(dim, N):
    return np.random.choice([-1, 1], (dim, N))

def testRed(dim, redFac):
    return dim * redFac

def projectedVec(matrix, vector):
  if not isinstance(vector, cp.ndarray):
    vector = cp.asarray(vector)
  if not isinstance(matrix, cp.ndarray):
    matrix = cp.asarray(matrix)
  return cp.matmul(matrix, vector)

def getSubset(data, size):
  return data.sample(size)


def run_pairs(data, eps, delta, dim, runs, projMat):
  distances = []
  for i in range(runs):
    time = time.time()
    vectors = getSubset(data,2).values
    dis = getDis(vectors[0], vectors[1])
    sketch = getDis(projectedVec(projMat, vectors[0]), projectedVec(projMat, vectors[1]))
    distances.append((dis-sketch)/dis)
    print(time.time() - time)
  return distances

def mse(distances):
  return cp.square(cp.mean(cp.asarray(distances)))

def findGoodN(data,eps,delta,dim,projMat):
  mses = []
  progress = 0
  for N in range(50,1000,50):
    distances = run_pairs(data, eps, delta, dim, N, projMat)
    mses.append(mse(distances))
    progress = progress + 1
    print(progress)
  mses = cp.concatenate(mses)
  mses = cp.asnumpy(mses)
  plt.plot(mses)
  plt.xlabel('N')
  plt.ylabel('MSE')
  plt.show()

def boxplot(data):
  box = plt.boxplot(data)
  print(stats.describe(data))
  plt.show()


def load_imagenette_as_dataframe():
  transform = transforms.Compose([
      transforms.Resize((100, 100)),
      transforms.ToTensor()
  ])
  try:
    train_dataset = datasets.Imagenette(root = '/content/', transform = transform, size = "320px", split = "val")
  except:
    train_dataset = datasets.Imagenette(root = '/content/', transform = transform, size = "320px", download = True, split = "val")
  image_data = []
  for img, _ in train_dataset:
    img_flat = np.array(img).flatten()
    image_data.append(img_flat)
  df = cudf.DataFrame(image_data)
  return df

"""**Install dataset**"""

images = load_imagenette_as_dataframe()
images.to_csv('/content/images.csv')

"""**Load dataset into a dataframe**"""

df = cudf.read_csv('/content/images.csv')
df = df.iloc[:,1:]

"""**Parameter choice**"""

eps = 0.2
delta = 0.05
N = df.shape[1]
dim = minDimCalc(eps, delta, N)
projMat = getProjMat(dim.astype(int), N)
findGoodN(df, eps, delta, dim,projMat)

"""**Test for normal distribution**"""

def checkNormal(data):
  fig = sm.qqplot(data, line='45')
  plt.show()
  #check if alpha>0.05
  stats.shapiro(data)
  stats.kstest(data, 'norm')

"""**Legacy code**"""

# Commented out IPython magic to ensure Python compatibility.
def writeToFile(data, filename):
    np.savetxt(filename, data, delimiter=',')



def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
#                                % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
#                                % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def run(data,eps,delta,N):
    distances =[]
    dim = minDimCalc(eps, delta, N)
    dim = dim.astype(int)
    projMat = getProjMat(dim, N)
    for index,vec in data.iterrows():
        projVec = (1/math.sqrt(dim)) * projectedVec(projMat, np.transpose(vec))
        distances.append(projVec)

    data = getDis(data)
    sketched = getDis(distances)
    assert(len(sketched) == len(data))
    assert(len(sketched[0])== len(data[0]))
    data = pd.DataFrame(data)
    sketched = pd.DataFrame(sketched)
    data.replace(0, np.nan, inplace = True)
    data.loc[:,:] = data.loc[:,:]**2
    sketched.replace(0, np.nan, inplace = True)
    sketched.loc[:,:] = sketched.loc[:,:]**2
    data.loc[:,:] = sketched.loc[:,:].div(data.loc[:,:])
    data = data.iloc[: , 1:]
    data = data.values.flatten()
    data = data[~np.isnan(data)]
    boxplot(data)
    tenth = np.percentile(data,10)
    ninety = np.percentile(data,90)
    mini = min(data)
    maxi = max(data)
    mean = np.mean(data)
    return mini, maxi, tenth, ninety, mean


start = time.time()
X_train, y_train = load_mnist('/content/fashion', kind='train')

df = pd.DataFrame(X_train)
smalltest = X_train[:3000]
# eps is the reduction factor
# delta is the confidence interval

eps = 0.3
delta = 0.05
setsize = 50
nSubsets = 50
N = df.shape[1]

minim = math.inf
maxim = -math.inf
tenth = math.inf
ninety = -math.inf
lower_conf = math.inf
upper_conf = -math.inf
mean = 1

for i in range(nSubsets):
  subsetData = getSubset(df, setsize)
  min_val,max_val, tenth_val, ninety_val, mean_val = run(subsetData, eps, delta, N)
  if min_val < minim:
    minim = min_val
  if max_val > maxim:
    maxim = max_val
  if tenth_val < tenth:
    tenth = tenth_val
  if ninety_val > ninety:
    ninety = ninety_val

  mean = mean+mean_val
mean = mean/nSubsets
end = time.time()
print(end - start)
print (minim, maxim, tenth, ninety, mean)