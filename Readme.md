# Face Recognition using PCA and LDA
## Names
### Marwan Khaled 7020
### Elhussein Sabri 6716
### Amr Yasser 6772



# References
https://www.adobe.com/mena_en/creativecloud/file-types/image/raster/pgm-file.html#:~:text=PGM%20(Portable%20Gray%20Map)%20files,shade%20of%20gray%20in%20between 

https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/ 

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

# Imports


```python
import os

# Import the necessary libraries
from PIL import Image
import numpy as np
import torch 
import pandas as pd
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import cv2
```

# Utils


```python
def project(D,Ur):
  mu = np.mean(D, axis=0)
  Z = D - mu
  
  return np.dot(Ur.T,Z.T).T

def knn(n_neighbors,train,train_y,test,test_y,con=False):
  knn =  KNeighborsClassifier(n_neighbors)
  knn.fit(train,train_y)
  y_pred = knn.predict(test)
  acc = accuracy_score(test_y,y_pred) 
  if con:
    print('\nAccuracy:',acc,'\n')
    confusion_matrix = metrics.confusion_matrix(test_y, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Non-face', 'Face'])
    cm_display.plot()
    plt.show()

  return acc

def display_data(D,y,title):
  print(f'\n{title}:')
  print(f'Data shape: {D.shape}')
  df = pd.DataFrame(D)
  df['Id'] = y

  print('\nValue counts:')
  plt.figure(figsize=(12,8))

  sns.countplot(x=df['Id'])
  plt.show()


  sample = D[np.random.choice(D.shape[0], 
                                  size=6, 
                                  replace=False),:]

  print('\nSample from data:')
  plt.figure(figsize=(12,8))
  i=1
  for s in sample:
    plt.subplot(2,3,i)
    plt.imshow(s.reshape(112,92),cmap='gray')
    i+=1
  plt.show()


def show_dimensions(U,train,test,alpha=False):
  if alpha:
    print(f'\n@ ɑlpha = {alpha}')
  print("\nReduction dimensions:",U.shape[1])
  print("Train reduced dimensions:",train.T.shape)
  print("Test reduced dimensions:",test.T.shape)


def plot(x,y,xl,yl):
  plt.plot(x,y,linestyle='--', marker='o', color='r', label='line with marker')
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.legend()
  plt.show()

def show_faces(imgs):
  plt.figure(figsize=(16,8))
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(imgs[:,i].reshape(112,92),cmap='gray')
  plt.show()
```

# Data Matrix and Label Vector


```python
# Run this cell only one time
D = np.zeros(shape=(10304, ), dtype= np.int8)
y = np.array([])

for i in range(1,41):
  for j in range(1,11):
    img = Image.open(f'/kaggle/input/att-database-of-faces/s{i}/{j}.pgm')
    data = np.asarray(img).reshape(-1)
    D = np.vstack((D,data))
    y = np.append(y,i)

y = y.astype('int8')
D = D[1:]

display_data(D,y,'Dataset')
```

    
    Dataset:
    Data shape: (400, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_7_1.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_7_3.png)
    


# Data Split


```python
train, test, train_y, test_y = train_test_split(D, y, test_size=0.5, random_state=42,stratify = y)

display_data(train,train_y,'Train data')
display_data(test,test_y,'Test data')
```

    
    Train data:
    Data shape: (200, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_9_1.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_9_3.png)
    


    
    Test data:
    Data shape: (200, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_9_5.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_9_7.png)
    


# Classification using PCA


```python
def PCA(D):
  # Step 1: Compute the mean of the data matrix X
  mu = np.mean(D, axis=0)

  # Step 2: Compute the centered data matrix Z
  Z = D - mu

  # Step 3: Compute the covariance matrix C
  C = np.cov(Z.T,bias=True)

  # Step 4: Compute the eigenvalues and eigenvectors of C
  eigenvalues, eigenvectors = np.linalg.eigh(C)

  # Step 5: Sort the eigenvalues and eigenvectors in decreasing order of eigenvalues
  indices = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[indices]
  eigenvectors = eigenvectors[:, indices]

  return eigenvectors,eigenvalues
```


```python
def find_R(eigenvectors,eigenvalues,ɑ):
  # Step 6: Compute dimensions of reduction corresponding to given ɑ value
  fr = 0
  sumEigens = sum(eigenvalues)
  r = 0

  while fr < ɑ:
    r+=1
    fr = sum(eigenvalues[:r]) / sumEigens

  # Step 7: Choose the reduced basis
  return eigenvectors[:,:r]
```


```python
# get the eigen faces
eigenvectors,eigenvalues = PCA(train)
```


```python
# a.Use the pseudo code below for computing the projection matrix U.
# Define the alpha = {0.8,0.85,0.9,0.95}

print('Top 5 eigenfaces:\n')
show_faces(eigenvectors[:,:5])

# compute reduction dimensions per alpha
U1 = find_R(eigenvectors,eigenvalues,0.8)

U2 = find_R(eigenvectors,eigenvalues,0.85)

U3 = find_R(eigenvectors,eigenvalues,0.9)

U4 = find_R(eigenvectors,eigenvalues,0.95)


```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_14_1.png)
    



```python
# b. Project the training set, and test sets separately using the same
# projection matrix
train1 = project(train,U1)
test1 = project(test,U1)
show_dimensions(U1,train1,test1,0.8)


train2 = project(train,U2)
test2 = project(test,U2)
show_dimensions(U2,train2,test2,0.85)


train3 = project(train,U3)
test3 = project(test,U3)
show_dimensions(U3,train3,test3,0.9)


train4 = project(train,U4)
test4 = project(test,U4)
show_dimensions(U4,train4,test4,0.95)

```

    
    @ ɑlpha = 0.8
    
    Reduction dimensions: 35
    Train reduced dimensions: (35, 200)
    Test reduced dimensions: (35, 200)
    
    @ ɑlpha = 0.85
    
    Reduction dimensions: 51
    Train reduced dimensions: (51, 200)
    Test reduced dimensions: (51, 200)
    
    @ ɑlpha = 0.9
    
    Reduction dimensions: 75
    Train reduced dimensions: (75, 200)
    Test reduced dimensions: (75, 200)
    
    @ ɑlpha = 0.95
    
    Reduction dimensions: 114
    Train reduced dimensions: (114, 200)
    Test reduced dimensions: (114, 200)
    


```python
# c. Use a simple classifier (first Nearest Neighbor to determine the class
# labels)
acc1 = knn(1,train1,train_y,test1,test_y)
acc2 = knn(1,train2,train_y,test2,test_y)
acc3 = knn(1,train3,train_y,test3,test_y)
acc4 = knn(1,train4,train_y,test4,test_y)

# d. Report Accuracy for every value of alpha separately
accs = [acc1,acc2,acc3,acc4]
print('Accuracy vector: ',accs,'\n\n')

plot([0.8,0.85,0.9,0.95],accs,'alpha','accuracy')
```

    Accuracy vector:  [0.935, 0.94, 0.945, 0.945] 
    
    
    


    
![png](face-recognition_files/face-recognition_16_1.png)
    


# Classification using LDA


```python
# D is the dataset
# m is number of classes

def LDA(D,m):  
  # Step 1: Calculate the Data matix for every class D1, D2, ..., Dm
  mu = np.mean(D,axis=0)
  Di = np.split(D,m)

  # Step 2: Calculate the mean vector for every class Mu1, Mu2, ..., Mum
  mui = np.mean(Di,axis=1)

  # Step 3: Calculate within-class and between-class scatter matrix for each class
  Sb = np.zeros((D.shape[1], D.shape[1]))
  S = np.zeros((D.shape[1], D.shape[1]))
  
  for i in range(m):
    # between-class scatter matrix
    nk = len(Di[i])
    cenMui = np.reshape(mui[i] - mu, (-1,D.shape[1]))
    Sb += nk * np.dot(cenMui.T,cenMui)
    
    # within-class scatter matrix
    Zi = Di[i] - mui[i]
    S += np.dot(Zi.T,Zi) 


  # Step 4: Compute dominant 39 eigenvectors
  S_inv = np.linalg.inv(S)
  eigenvalues,eigenvectors = np.linalg.eigh(np.dot(S_inv,Sb))

  indices = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[indices]
  eigenvectors = eigenvectors[:, indices]


  print('Top 5 eigenfaces:\n')
  show_faces(eigenvectors[:,:5])


  U = eigenvectors[:,:m-1]

  return U


```


```python
# Projection Matrix
U = LDA(train,40)
```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_19_1.png)
    



```python
# b. Project the training set, and test sets separately using the same
# projection matrix U. You will have 39 dimensions in the new space
trainP = project(train,U)
testP = project(test,U)

show_dimensions(U,trainP,testP)
```

    
    Reduction dimensions: 39
    Train reduced dimensions: (39, 200)
    Test reduced dimensions: (39, 200)
    


```python
# c. Use a simple classifier (first Nearest Neighbor to determine the class
# labels).
acc = knn(1,trainP,train_y,testP,test_y)

# d. Report accuracy for the multiclass LDA on the face recognition
# dataset
print('LDA classification with KNN@(n_neighbours = 1) Accuracy:',acc)

```

    LDA classification with KNN@(n_neighbours = 1) Accuracy: 0.92
    

# Classifier Tuning using K-NN =[1,3,5,7]

## PCA accuracy measure against number of neighbors


```python
for neighbors in range(1,9,2):
  print("At k=",neighbors,"\n")
  acc1 = knn(neighbors,train1,train_y,test1,test_y)
  acc2 = knn(neighbors,train2,train_y,test2,test_y)
  acc3 = knn(neighbors,train3,train_y,test3,test_y)
  acc4 = knn(neighbors,train4,train_y,test4,test_y)
  accs = [acc1,acc2,acc3,acc4]
  print('Accuracy vector: ',accs,'\n\n')


```

    At k= 1 
    
    Accuracy vector:  [0.935, 0.94, 0.945, 0.945] 
    
    
    At k= 3 
    
    Accuracy vector:  [0.89, 0.87, 0.85, 0.855] 
    
    
    At k= 5 
    
    Accuracy vector:  [0.795, 0.795, 0.785, 0.785] 
    
    
    At k= 7 
    
    Accuracy vector:  [0.76, 0.77, 0.755, 0.725] 
    
    
    


```python
#for alpha=0.8
fig,axs=plt.subplots(2,2,figsize=(10,10))
PCA_y8=np.array([0.935,0.89,0.795,0.76])
PCA_x8=np.array([1,3,5,7])
axs[0,0].set_title("Alpha = 0.8")
axs[0,0].plot(PCA_x8,PCA_y8,'tab:blue')
axs[0,0].set_xlabel('#neighbours')
axs[0,0].set_ylabel('Accuracy')

#for alpha=0.85

PCA_y85=np.array([0.94,0.87,0.795,0.77])
PCA_x85=np.array([1,3,5,7])
axs[0,1].set_title("Alpha = 0.85")
axs[0,1].plot(PCA_x85,PCA_y85,'tab:orange')
axs[0,1].set_xlabel('#neighbours')
axs[0,1].set_ylabel('Accuracy')

#for alpha=0.9

PCA_y9=np.array([0.945,0.85,0.785,0.755])
PCA_x9=np.array([1,3,5,7])
axs[1,0].set_title("Alpha = 0.9")
axs[1,0].plot(PCA_x9,PCA_y9,'tab:green')
axs[1,0].set_xlabel('#neighbours')
axs[1,0].set_ylabel('Accuracy')

#for alpha=0.95

PCA_y95=np.array([0.945,0.855,0.785,0.725])
PCA_x95=np.array([1,3,5,7])
axs[1,1].set_title("Alpha = 0.95")
axs[1,1].plot(PCA_x95,PCA_y95,'tab:red')
axs[1,1].set_xlabel('#neighbours')
axs[1,1].set_ylabel('Accuracy')

plt.show()
```


    
![png](face-recognition_files/face-recognition_25_0.png)
    


## LDA measure against K-NN



```python

for neighbors in range(1,9,2):
  print("\n At k=",neighbors)
  acc= knn(neighbors,trainP,train_y,testP,test_y)
  
  # d. Report accuracy for the multiclass LDA on the face recognition
  # dataset
  print('LDA classification Accuracy:',acc)

```

    
     At k= 1
    LDA classification Accuracy: 0.92
    
     At k= 3
    LDA classification Accuracy: 0.825
    
     At k= 5
    LDA classification Accuracy: 0.785
    
     At k= 7
    LDA classification Accuracy: 0.765
    


```python
LDA_y1=np.array([0.92,0.825,0.785,0.765])
LDA_x1=np.array([1,3,5,7])
plt.plot(LDA_x1,LDA_y1,'tab:blue')
plt.xlabel('#neighbours')
plt.ylabel('Accuracy')
plt.show()
```


    
![png](face-recognition_files/face-recognition_28_0.png)
    


# Compare vs Non faces


```python
faces2=[]
target2=[]
for i in range(1, 41):
  for j in range (1,11):
    with open('/kaggle/input/att-database-of-faces/s'+str(i)+'/'+str(j)+'.pgm','rb') as imgf:
      image = plt.imread(imgf)
      image=image.transpose()
      im=image.flatten()
      faces2.append(im)
      target2.append(1)
for i in range (1,401):
    with open('/kaggle/input/turtles/Turtle_Tortoise ('+str(i)+').jpg','rb') as imgn:
      image2=plt.imread(imgn)
      image_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
      resized_image = cv2.resize(image_rgb, (92, 112))
      image2=resized_image.transpose()
      im2=image2.flatten()
      faces2.append(im2)
      target2.append(0)
faces=np.array(faces2)
target=np.array(target2)
target=np.reshape(target,(800,1))

```


```python
def readd(R):
  faces2=[]
  target2=[]
  for i in range(1, 41):
    for j in range (1,11):
      with open('/kaggle/input/att-database-of-faces/s'+str(i)+'/'+str(j)+'.pgm','rb') as imgf:
       image = plt.imread(imgf)
       image=image.transpose()
       im=image.flatten()
       faces2.append(im)
       target2.append(1)
  for i in range (1,R+1):
    with open('/kaggle/input/turtles/Turtle_Tortoise ('+str(i)+').jpg','rb') as imgn:
      image2=plt.imread(imgn)
      image_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
      resized_image = cv2.resize(image_rgb, (92, 112))
      image2=resized_image.transpose()
      im2=image2.flatten()
      faces2.append(im2)
      target2.append(0)
  faces=np.array(faces2)
  target=np.array(target2)
  target=np.reshape(target,(R+400,1))
  return(faces,target) 

```

## 200 non-face images vs 400 face images (unbalanced)


```python
fn2,tn2=readd(200)
display(tn2.shape)

train_n2, test_n2, train_yn2, test_yn2 = train_test_split(fn2, tn2, test_size=0.5, random_state=42,stratify = tn2)
```


    (600, 1)


##### PCA


```python
# get the eigen faces
eigenvectors,eigenvalues = PCA(train_n2)
```


```python
U1n2 = find_R(eigenvectors,eigenvalues,0.8)

U2n2 = find_R(eigenvectors,eigenvalues,0.85)

U3n2 = find_R(eigenvectors,eigenvalues,0.9)

U4n2 = find_R(eigenvectors,eigenvalues,0.95)
```


```python
train1n2 = project(train_n2,U1n2)
test1n2 = project(test_n2,U1n2)
show_dimensions(U1n2,train1n2,test1n2,0.8)


train2n2 = project(train_n2,U2n2)
test2n2 = project(test_n2,U2n2)
show_dimensions(U2n2,train2n2,test2n2,0.85)


train3n2 = project(train_n2,U3n2)
test3n2 = project(test_n2,U3n2)
show_dimensions(U3n2,train3n2,test3n2,0.9)


train4n2 = project(train_n2,U2n2)
test4n2 = project(test_n2,U2n2)
show_dimensions(U4n2,train4n2,test4n2,0.95)
```

    
    @ ɑlpha = 0.8
    
    Reduction dimensions: 40
    Train reduced dimensions: (40, 300)
    Test reduced dimensions: (40, 300)
    
    @ ɑlpha = 0.85
    
    Reduction dimensions: 60
    Train reduced dimensions: (60, 300)
    Test reduced dimensions: (60, 300)
    
    @ ɑlpha = 0.9
    
    Reduction dimensions: 90
    Train reduced dimensions: (90, 300)
    Test reduced dimensions: (90, 300)
    
    @ ɑlpha = 0.95
    
    Reduction dimensions: 143
    Train reduced dimensions: (60, 300)
    Test reduced dimensions: (60, 300)
    


```python
acc1n2 = knn(1,train1n2,train_yn2.reshape(-1),test1n2,test_yn2.reshape(-1),True)
acc2n2 = knn(1,train2n2,train_yn2.reshape(-1),test2n2,test_yn2.reshape(-1),True)
acc3n2 = knn(1,train3n2,train_yn2.reshape(-1),test3n2,test_yn2.reshape(-1),True)
acc4n2 = knn(1,train4n2,train_yn2.reshape(-1),test4n2,test_yn2.reshape(-1),True)

# d. Report Accuracy for every value of alpha separately
accsn2 = [acc1n2,acc2n2,acc3n2,acc4n2]
print('\nAccuracy vector: ',accsn2,'\n')

plot([0.8,0.85,0.9,0.95],accsn2,'alpha','accuracy')
```

    
    Accuracy: 0.9766666666666667 
    
    


    
![png](face-recognition_files/face-recognition_38_1.png)
    


    
    Accuracy: 0.9766666666666667 
    
    


    
![png](face-recognition_files/face-recognition_38_3.png)
    


    
    Accuracy: 0.9766666666666667 
    
    


    
![png](face-recognition_files/face-recognition_38_5.png)
    


    
    Accuracy: 0.9766666666666667 
    
    


    
![png](face-recognition_files/face-recognition_38_7.png)
    


    
    Accuracy vector:  [0.9766666666666667, 0.9766666666666667, 0.9766666666666667, 0.9766666666666667] 
    
    


    
![png](face-recognition_files/face-recognition_38_9.png)
    


##### LDA


```python
Un2 = LDA(train_n2,2)
```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_40_1.png)
    



```python
trainn2 = project(train_n2,Un2)
testn2 = project(test_n2,Un2)

show_dimensions(Un2,trainn2,testn2)
```

    
    Reduction dimensions: 1
    Train reduced dimensions: (1, 300)
    Test reduced dimensions: (1, 300)
    


```python
knn(1,trainn2,train_yn2.reshape(-1),testn2,test_yn2.reshape(-1),True);
```

    
    Accuracy: 0.5566666666666666 
    
    


    
![png](face-recognition_files/face-recognition_42_1.png)
    


## 400 non face images vs 400 face images(balanced)


```python
fn4,tn4=readd(400)
display(tn4.shape)

train_n4, test_n4, train_yn4, test_yn4 = train_test_split(fn4, tn4, test_size=0.5, random_state=42,stratify = tn4)
```


    (800, 1)


##### PCA


```python
# get the eigen faces
eigenvectors,eigenvalues = PCA(train_n4)
```


```python
U1n4 = find_R(eigenvectors,eigenvalues,0.8)

U2n4 = find_R(eigenvectors,eigenvalues,0.85)

U3n4 = find_R(eigenvectors,eigenvalues,0.9)

U4n4 = find_R(eigenvectors,eigenvalues,0.95)
```


```python
train1n4 = project(train_n4,U1n4)
test1n4 = project(test_n4,U1n4)
show_dimensions(U1n4,train1n4,test1n4,0.8)


train2n4 = project(train_n4,U2n4)
test2n4 = project(test_n4,U2n4)
show_dimensions(U2n4,train2n4,test2n4,0.85)


train3n4 = project(train_n4,U3n4)
test3n4 = project(test_n4,U3n4)
show_dimensions(U3n4,train3n4,test3n4,0.9)


train4n4 = project(train_n4,U4n4)
test4n4 = project(test_n4,U4n4)
show_dimensions(U4n4,train4n4,test4n4,0.95)
```

    
    @ ɑlpha = 0.8
    
    Reduction dimensions: 47
    Train reduced dimensions: (47, 400)
    Test reduced dimensions: (47, 400)
    
    @ ɑlpha = 0.85
    
    Reduction dimensions: 74
    Train reduced dimensions: (74, 400)
    Test reduced dimensions: (74, 400)
    
    @ ɑlpha = 0.9
    
    Reduction dimensions: 116
    Train reduced dimensions: (116, 400)
    Test reduced dimensions: (116, 400)
    
    @ ɑlpha = 0.95
    
    Reduction dimensions: 189
    Train reduced dimensions: (189, 400)
    Test reduced dimensions: (189, 400)
    


```python
acc1n4 = knn(1,train1n4,train_yn4.reshape(-1),test1n4,test_yn4.reshape(-1),True)
acc2n4 = knn(1,train2n4,train_yn4.reshape(-1),test2n4,test_yn4.reshape(-1),True)
acc3n4 = knn(1,train3n4,train_yn4.reshape(-1),test3n4,test_yn4.reshape(-1),True)
acc4n4 = knn(1,train4n4,train_yn4.reshape(-1),test4n4,test_yn4.reshape(-1),True)

# d. Report Accuracy for every value of alpha separately
accsn4 = [acc1n4,acc2n4,acc3n4,acc4n4]
print('\nAccuracy vector: ',accsn4,'\n')

plot([0.8,0.85,0.9,0.95],accsn4,'alpha','accuracy')
```

    
    Accuracy: 0.9825 
    
    


    
![png](face-recognition_files/face-recognition_49_1.png)
    


    
    Accuracy: 0.9825 
    
    


    
![png](face-recognition_files/face-recognition_49_3.png)
    


    
    Accuracy: 0.985 
    
    


    
![png](face-recognition_files/face-recognition_49_5.png)
    


    
    Accuracy: 0.985 
    
    


    
![png](face-recognition_files/face-recognition_49_7.png)
    


    
    Accuracy vector:  [0.9825, 0.9825, 0.985, 0.985] 
    
    


    
![png](face-recognition_files/face-recognition_49_9.png)
    


##### LDA


```python
Un4 = LDA(train_n4,2)
```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_51_1.png)
    



```python
trainn4 = project(train_n4,Un4)
testn4 = project(test_n4,Un4)

show_dimensions(Un4,trainn4,testn4)
```

    
    Reduction dimensions: 1
    Train reduced dimensions: (1, 400)
    Test reduced dimensions: (1, 400)
    


```python
knn(1,trainn4,train_yn4.reshape(-1),testn4,test_yn4.reshape(-1),True);
```

    
    Accuracy: 0.515 
    
    


    
![png](face-recognition_files/face-recognition_53_1.png)
    


## 600 non face images vs 400 face images(unbalanced)


```python
fn6,tn6=readd(600)
display(tn6.shape)

train_n6, test_n6, train_yn6, test_yn6 = train_test_split(fn6, tn6, test_size=0.5, random_state=42,stratify = tn6)
```


    (1000, 1)


##### PCA


```python
# get the eigen faces
eigenvectors,eigenvalues = PCA(train_n6)
```


```python
U1n6 = find_R(eigenvectors,eigenvalues,0.8)

U2n6 = find_R(eigenvectors,eigenvalues,0.85)

U3n6 = find_R(eigenvectors,eigenvalues,0.9)

U4n6 = find_R(eigenvectors,eigenvalues,0.95)
```


```python
train1n6 = project(train_n6,U1n6)
test1n6 = project(test_n6,U1n6)
show_dimensions(U1n6,train1n6,test1n6,0.8)


train2n6 = project(train_n6,U2n6)
test2n6 = project(test_n6,U2n6)
show_dimensions(U2n6,train2n6,test2n6,0.85)


train3n6 = project(train_n6,U3n6)
test3n6 = project(test_n6,U3n6)
show_dimensions(U3n6,train3n6,test3n6,0.9)


train4n6 = project(train_n6,U4n6)
test4n6 = project(test_n6,U4n6)
show_dimensions(U4n6,train4n6,test4n6,0.95)
```

    
    @ ɑlpha = 0.8
    
    Reduction dimensions: 51
    Train reduced dimensions: (51, 500)
    Test reduced dimensions: (51, 500)
    
    @ ɑlpha = 0.85
    
    Reduction dimensions: 82
    Train reduced dimensions: (82, 500)
    Test reduced dimensions: (82, 500)
    
    @ ɑlpha = 0.9
    
    Reduction dimensions: 134
    Train reduced dimensions: (134, 500)
    Test reduced dimensions: (134, 500)
    
    @ ɑlpha = 0.95
    
    Reduction dimensions: 226
    Train reduced dimensions: (226, 500)
    Test reduced dimensions: (226, 500)
    


```python
acc1n6 = knn(1,train1n6,train_yn6.reshape(-1),test1n6,test_yn6.reshape(-1),True)
acc2n6 = knn(1,train2n6,train_yn6.reshape(-1),test2n6,test_yn6.reshape(-1),True)
acc3n6 = knn(1,train3n6,train_yn6.reshape(-1),test3n6,test_yn6.reshape(-1),True)
acc4n6 = knn(1,train4n6,train_yn6.reshape(-1),test4n6,test_yn6.reshape(-1),True)

# d. Report Accuracy for every value of alpha separately
accsn6 = [acc1n6,acc2n6,acc3n6,acc4n6]
print('Accuracy vector: ',accsn6,'\n\n')

plot([0.8,0.85,0.9,0.95],accsn6,'alpha','accuracy')
```

    
    Accuracy: 0.992 
    
    


    
![png](face-recognition_files/face-recognition_60_1.png)
    


    
    Accuracy: 0.994 
    
    


    
![png](face-recognition_files/face-recognition_60_3.png)
    


    
    Accuracy: 0.996 
    
    


    
![png](face-recognition_files/face-recognition_60_5.png)
    


    
    Accuracy: 0.994 
    
    


    
![png](face-recognition_files/face-recognition_60_7.png)
    


    Accuracy vector:  [0.992, 0.994, 0.996, 0.994] 
    
    
    


    
![png](face-recognition_files/face-recognition_60_9.png)
    


##### LDA


```python
Un6 = LDA(train_n6,2)
```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_62_1.png)
    



```python
trainn6 = project(train_n6,Un6)
testn6 = project(test_n6,Un6)

show_dimensions(Un6,trainn6,testn6)
```

    
    Reduction dimensions: 1
    Train reduced dimensions: (1, 500)
    Test reduced dimensions: (1, 500)
    


```python
knn(1,trainn6,train_yn6.reshape(-1),testn6,test_yn6.reshape(-1),True);
```

    
    Accuracy: 0.576 
    
    


    
![png](face-recognition_files/face-recognition_64_1.png)
    


## Accuracy vs the number of non-faces images while fixing the number of face images

### PCA


```python
#for alpha=0.8
fig,axs=plt.subplots(2,2,figsize=(10,10))
PCAn_y8=np.array([0.9766666666666667,0.9825,0.99])
PCAn_x8=np.array([200,400,600])
axs[0,0].set_title("Alpha = 0.8")
axs[0,0].plot(PCAn_x8,PCAn_y8,'tab:blue')
axs[0,0].set_xlabel('#Non-Face')
axs[0,0].set_ylabel('Accuracy')

#for alpha=0.85

PCAn_y85=np.array([0.9766666666666667,0.9825,0.996])
PCAn_x85=np.array([200,400,600])
axs[0,1].set_title("Alpha = 0.85")
axs[0,1].plot(PCAn_x85,PCAn_y85,'tab:orange')
axs[0,1].set_xlabel('#Non-Face')
axs[0,1].set_ylabel('Accuracy')

#for alpha=0.9

PCAn_y9=np.array([0.9766666666666667,0.985,0.996])
PCAn_x9=np.array([200,400,600])
axs[1,0].set_title("Alpha = 0.9")
axs[1,0].plot(PCAn_x9,PCAn_y9,'tab:green')
axs[1,0].set_xlabel('#Non-Face')
axs[1,0].set_ylabel('Accuracy')

#for alpha=0.95

PCAn_y95=np.array([0.9766666666666667,0.985,0.996])
PCAn_x95=np.array([200,400,600])
axs[1,1].set_title("Alpha = 0.95")
axs[1,1].plot(PCAn_x95,PCAn_y95,'tab:red')
axs[1,1].set_xlabel('#Non-Face')
axs[1,1].set_ylabel('Accuracy')

plt.show()
```


    
![png](face-recognition_files/face-recognition_67_0.png)
    


### LDA



```python
LDAn_y1=np.array([0.5566666666666666 ,0.52,0.62])
LDAn_x1=np.array([200,400,600])
plt.plot(LDAn_x1,LDAn_y1,'tab:blue')
plt.xlabel('#Non-Face')
plt.ylabel('Accuracy')
plt.show()
```


    
![png](face-recognition_files/face-recognition_69_0.png)
    


# Bonus

## Data split


```python
# a. Use different Training and Test splits. Change the number of instances per subject to be 7 and keep 3 instances per subject for testing. compare the results you have with the ones you got earlier with 50% split. 
trainB, testB, trainB_y, testB_y = train_test_split(D, y, test_size=0.3, random_state=42,stratify = y)


display_data(trainB,trainB_y,'Train data')
display_data(testB,testB_y,'Test data')



```

    
    Train data:
    Data shape: (280, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_72_1.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_72_3.png)
    


    
    Test data:
    Data shape: (120, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_72_5.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_72_7.png)
    


## PCA


```python
eigenvectors,eigenvalues = PCA(trainB)
```


```python
# a.Use the pseudo code below for computing the projection matrix U.
# Define the alpha = {0.8,0.85,0.9,0.95}

print('Top 5 eigenfaces:\n')
show_faces(eigenvectors[:,:5])

# compute reduction dimensions per alpha
U1B = find_R(eigenvectors,eigenvalues,0.8)

U2B = find_R(eigenvectors,eigenvalues,0.85)

U3B = find_R(eigenvectors,eigenvalues,0.9)

U4B = find_R(eigenvectors,eigenvalues,0.95)

```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_75_1.png)
    



```python
# b. Project the training set, and test sets separately using the same
# projection matrix
train1B = project(trainB,U1B)
test1B = project(testB,U1B)
show_dimensions(U1B,train1B,test1B,0.8)

train2B = project(trainB,U2B)
test2B = project(testB,U2B)
show_dimensions(U2B,train2B,test2B,0.85)

train3B = project(trainB,U3B)
test3B = project(testB,U3B)
show_dimensions(U3B,train3B,test3B,0.9)

train4B = project(trainB,U4B)
test4B = project(testB,U4B)
show_dimensions(U4B,train4B,test4B,0.95)

```

    
    @ ɑlpha = 0.8
    
    Reduction dimensions: 40
    Train reduced dimensions: (40, 280)
    Test reduced dimensions: (40, 120)
    
    @ ɑlpha = 0.85
    
    Reduction dimensions: 59
    Train reduced dimensions: (59, 280)
    Test reduced dimensions: (59, 120)
    
    @ ɑlpha = 0.9
    
    Reduction dimensions: 91
    Train reduced dimensions: (91, 280)
    Test reduced dimensions: (91, 120)
    
    @ ɑlpha = 0.95
    
    Reduction dimensions: 148
    Train reduced dimensions: (148, 280)
    Test reduced dimensions: (148, 120)
    


```python
# c. Use a simple classifier (first Nearest Neighbor to determine the class
# labels)
acc1B = knn(1,train1B,trainB_y,test1B,testB_y)
acc2B = knn(1,train2B,trainB_y,test2B,testB_y)
acc3B = knn(1,train3B,trainB_y,test3B,testB_y)
acc4B = knn(1,train4B,trainB_y,test4B,testB_y)

# d. Report Accuracy for every value of alpha separately
accsB = [acc1B,acc2B,acc3B,acc4B]
print('Accuracy vector: ',accsB,'\n\n')

plot([0.8,0.85,0.9,0.95],accsB,'alpha','accuracy')
```

    Accuracy vector:  [0.9583333333333334, 0.9416666666666667, 0.9583333333333334, 0.95] 
    
    
    


    
![png](face-recognition_files/face-recognition_77_1.png)
    


## LDA


```python
# Projection Matrix
UB = LDA(trainB,40)

```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_79_1.png)
    



```python
# b. Project the training set, and test sets separately using the same
# projection matrix U. You will have 39 dimensions in the new space
trainPB = project(trainB,UB)
testPB = project(testB,UB)

show_dimensions(UB,trainPB,testPB)
```

    
    Reduction dimensions: 39
    Train reduced dimensions: (39, 280)
    Test reduced dimensions: (39, 120)
    


```python
# c. Use a simple classifier (first Nearest Neighbor to determine the class
# labels).
accB = knn(1,trainPB,trainB_y,testPB,testB_y)

# d. Report accuracy for the multiclass LDA on the face recognition
# dataset
print('LDA classification with KNN@(n_neighbours = 1) Accuracy:',accB)

```

    LDA classification with KNN@(n_neighbours = 1) Accuracy: 0.95
    

## sklearn PCA


```python
from sklearn.decomposition import IncrementalPCA as IPCA
pca = IPCA(n_components=91)
 
pca_train = pca.fit_transform(trainB)
pca_test = pca.transform(testB)

```


```python
knn(1,pca_train,trainB_y,pca_test,testB_y)
```




    0.9583333333333334



## sklearn LDA


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(trainB, trainB_y)

lda_trainB = clf.transform(trainB)
lda_testB = clf.transform(testB)

print(lda_trainB.shape,lda_testB.shape)
```

    (280, 39) (120, 39)
    


```python
knn(1,lda_trainB,trainB_y,lda_testB,testB_y)
```




    0.9666666666666667



# Scaling


```python
scaler = MinMaxScaler()
Dscaled = scaler.fit_transform(D)

display_data(Dscaled,y,'Scaled dataset')
```

    
    Scaled dataset:
    Data shape: (400, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_89_1.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_89_3.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split(Dscaled, y, test_size=0.3, random_state=42,stratify = y)

display_data(X_train,y_train,'Train data')
display_data(X_test,y_test,'Test data')
```

    
    Train data:
    Data shape: (280, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_90_1.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_90_3.png)
    


    
    Test data:
    Data shape: (120, 10304)
    
    Value counts:
    


    
![png](face-recognition_files/face-recognition_90_5.png)
    


    
    Sample from data:
    


    
![png](face-recognition_files/face-recognition_90_7.png)
    



```python
eigenvectors,eigenvalues = PCA(X_train)
```


```python
Us = find_R(eigenvectors,eigenvalues,0.9)
```


```python
trainsca = project(X_train,Us)
testsca = project(X_test,Us)
show_dimensions(Us,trainsca,testsca,0.85)
```

    
    @ ɑlpha = 0.85
    
    Reduction dimensions: 88
    Train reduced dimensions: (88, 280)
    Test reduced dimensions: (88, 120)
    


```python
knn(1,trainsca,y_train,testsca,y_test)
```




    0.9666666666666667




```python
# Projection Matrix
UBs = LDA(X_train,40)
```

    Top 5 eigenfaces:
    
    


    
![png](face-recognition_files/face-recognition_95_1.png)
    



```python
trainPs = project(X_train,UBs)
testPs = project(X_test,UBs)

show_dimensions(UBs,trainPs,testPs)
```

    
    Reduction dimensions: 39
    Train reduced dimensions: (39, 280)
    Test reduced dimensions: (39, 120)
    


```python
knn(1,trainPs,y_train,testPs,y_test)
```




    0.9583333333333334


