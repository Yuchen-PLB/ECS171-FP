# ECS171-FP

Project from UC Davis Fall 2022 ECS171 - Introduction to Machine Learning

By: Keer (Nicole) Ni, Tingwei Liu, Yuchen Liu, and Kehuan Wang.

## Link to the dataset (from Kaggle)
https://www.kaggle.com/datasets/michaelbryantds/cpu-and-gpu-product-data

## Information about the dataset being used
This dataset contains features including: Product (name of the product), Type (CPU or GPU) Release Date (release date of the product), Process Size (this is the physical size of the components that makes up a chip), TDP (thermal design point), Die Size (physical dimension of a bare die), Transistors (number of transistors), Freq (frequency in MHz), Foundry (company that makes the CPU/GPU), Vendor (seller of the CPU/GPU).

In this dataset, the Type could be considered as a label/category for other features (e.g. we could predict the Type (is it a CPU/GPU) given other features (Process Size, TDP, Die Size, Freq)). i.e. Which architecture has better performances. Other categorical features such as Foundry and Vendor could also be considered as labels/categories (using other numerical features to predict the Foundry or Vendor for example). In addition, the release date can also be considered as a label. According to Moore's law, there is a relationship between releasing date and number of transistors, which affect the performance of CPU/GPUs.

## Introduction

### Objectives and Goals
1. We will do visualized analysis on the data set to discover the trend of CPU and GPU  development.
2. We are expected to use the visualized data to explain simple questions such as: which company is producing the better CPUs and which company is producing the better GPUs.
3. Using neural networks to discuss the relationship between different variables, we are expected to find a general trend of CPU & GPU development. We will focus on the building up models about the relationship between the performance and the different categorical variables and the release date. 
4. We will look deeper into the model and explain :
    1. Will Moore's Law still hold in the next 5-8 years?
    2. Will Dannard Scaling be still valid in the next 5-8 years?
    3. What is most related to CPU/GPU performance? Is it the transistors size, die size, or frequency? What is the relationship between them?
    4. What would be the tread of future CPU/GPU development? Are their sizes getting smaller? Is their frequency going to increase? What would be the limit of chip performance?

### Steps to Take
1. Set up a Github repository to track updates and progress of the final project.
2. Data exploration (e.g. Histogram, Pair Plot, Q-Q Plot, Line Plot, t-SNE Plot…).
3. Data preprocessing (e.g. remove rows with null data, encode categorical data into numerical values if needed, data scaling (normalization, standardization), data imputation (consider to impute the missing data if don’t want to remove rows with null value), …).
4. Choose appropriate machine learning models and train the model (linear regression, polynomial regression, neural network, logistic regression, …).
5. Construct clear write-ups (include goals, results, and discussion) to report our findings. 

## ToDos for more Data Preprocessing and more Model Training:
1. We decided to apply missing value imputation (MissForest) to impute the missing values in the dataset. This is because the there are too many missing values in the dataset and removing them will result in having a dataset not large enough for training and testing the model.
2. We decided to apply logistic regression to make prediction for 'Vendor' feature.
3. We decided to split the final project into different objectives (e.g. one of the objectives is to predict whether a given trial/sample is a CPU or GPU using the 'Type' label as y for prediction results). This will require us to preprocess the dataset differently (i.e. scale and transform only the feature values X, but separate the predicting values y without using any scaling or transforming method to it). We will look more into the dataset and decide to use other models (logistic regression, neural network, decision tree, SVM).
4. We are trying to do clusting and PCA on the data to discuss the impact of different variables and use them to reduce the model

## Figures and Data exploration

### Part 1. Data exploration
NOTE: We are using Google Colab as the coding environment. To read the dataset correctly, please upload the **chip_dataset.csv** file to the folder in Google Colab and progress from there.
1. We took at look at the different features and their maximum & minimum values.
2. We transformed the time values of the 'Date' feature to make it easier to plot and analysis.
3. We splitted the dataset into two main categories: CPU and GPU. Since the GPU category/class has more feature than the CPU feature. 
4. We look at seaborn pairplots and correlation coefficients to see the data distribution.

## Methods

### Part 3.1 Data Preprocessing and Train First Model (Linear Regression and Polynomial Regression)
1. We first clean the dataframe by deleting unused column 'Unnamed: 0' (only shows indecies of the dataframe).
2. We normalize the dataset using Minmax Normalization.
3. We encode the categorical column/features name 'Vendor' which will be used in other models later. 
4. We build and train the Linear Regression model and Polynomial Regression model (up to degree 3).
5. We report R^2 which is the percentage of variation explain/reveal by the regression model. 
6. We run a logistic regression on the categorical data, trying to explain which Vendor is tending to produce higher quality CPU/GPU.


### Part 3.2 Vendor Prediction with ANN Model
This model predict the brand of CPU/GPU with expect to its 'Release Date Process Size (nm) TDP (W) Die Size (mm^2)	Transistors (million)	Freq (MHz)'. We enumerate the vendor to 0, 1, 2 as our y-label and normalize the properties mentioned above as our X. 
#### 3.2.1 Method
We build a 3-layer ANN model with activation relu and signmoid, train for 1000 epoch. The activation function is chosen based on the previous observation of the logistic regression, which shows clear linear relationship for the previous 5 years from 2000 and more polinomial shape for the recent years.
We tried categorical NN and sequential NN to explore the relationship between chip performances and the Vendor. At the end, we also build the comfusion matrix to try to evaluate the performance of our model. The code works like below: 
```
from sklearn.naive_bayes import CategoricalNB
model = CategoricalNB()
model.fit(X_train, y_train)
yhat_test = model.predict(X_test)
```
Here, X_train is size, tdp, die size,transistors and y_train is the enumerated vendor (for CPU is 0,1 since there is only AMD and Intel; whereas for GPUs are 0,1,2,3 since there are four known Venders in the data).

After building the model, we create confusion matrix with respect to the prediction:
```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
yhat_test = model.predict(X_test)
myconfusionmatrix = confusion_matrix(y_test, yhat_test)
display(myconfusionmatrix)
mycmdisp = ConfusionMatrixDisplay(confusion_matrix=myconfusionmatrix, display_labels=model.classes_)
mycmdisp.plot()
```
The sequential NN takes the time as a sequential argument, which helps to identify the performance of chips of different brands within 20 years. 
```
import tensorflow
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units =6, activation = 'relu', input_dim = 6))
model.add(Dense(units = 6, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 1, epochs = 1000)
```

#### 3.2.2 Result
CPU and GPU shows different performance after training. For the categorical model, the CPU shows 0 wrong prediction on Vendor 1 and 137 wrong prediction on Vendor 0.  
![alt text](pictures/ANNconfusion_matrix_CPU.png)

For the sequential model, it reaches around 0.85 accuracy after 700 epochs.

The model below shows the model accuracy in respect to the epochs. 
![alt text](pictures/ANN-CPUmodel.png)

 However, with respect to the GPU sequential model, the model didn't provide accurate prediction.

#### 3.2.3 Discussion and Application
Based on the four model built above(sequential CPU, sequential GPU, categorical CPU and categorical GPU), the sequential CPU model provides best performance. It shows 0.85 accuracy, which proves our assumption that the perfomance of the chips grows more faster in the recent years for both vendors. This model can also further predict the performance of the chips in future for different venders. 
The categorical CPU model, however, didn't shows good performance on predict different venders. This might happen because when ignoring the release date veriable, one vender might produce a similar chip after another vender produce it few years later. Further exploration is needed to find out the reason why it is harder to predict the vendors.
The sequential model for GPU does not shows only 10% accuracy. This might happen because there are more venders involved in the GPU industries, which increase the complexity of the model, 100 epouch is not enough. There also need further exploration to figure out the relationship between different GPU venders to the development of GPUS.

### Part 3.3 Clustering

### 3.3.1 Method 1: Agglomerative Hierarchical Clustering
Hierarchical clustering is an unsupervised clustering algorithm used to create clusters with a tree-like hierarchy. In this clustering method, there is no need to give the number of clusters to the algorithm.  In contrast to this, the other algorithm like K-Mean produces flat clusters where there is no hierarchy and we also have to choose the number of clusters, to begin with.

Here, we first draw a Hierarchical Dendrogram to have a general overview of the CPU data to decide the number of clusters we need for the K-Mean clustering algorithm.

The hierarchical clustering algorithm can be of two types –
Divisive Clustering – It takes a top-down approach where the entire data observation is considered to be one big cluster at the start. Then subsequently it is split into two clusters, then three clusters, and so on until each data ends up as a separate cluster.
Agglomerative Clustering – It takes a bottom-up approach where it assumes individual data observation to be one cluster at the start. Then it starts merging the data points into clusters till it creates one final cluster at the end with all data points.

### 3.1.1 Method
Parameters of Agglomerative Clustering
The agglomeration hierarchical clustering can have multiple variations depending on affinity and linkage.
Affinity
Affinity denotes the method using which the distance or similarity between data points or clusters is calculated. Which include –

K-nearest neighbor (KNN) 
K-nearest neighbor (KNN) is a non-parametric classifier. The prediction of the label of a test point is assigned according to the vote of its K nearest neighbors’ labels, where K is a user-defined parameter. KNN is a simple technique, and could work well when given a good distance metric and sufficient training dataset. It can be shown that the KNN classifier can come within a factor of 2 of the best possible performance if N → ∞ . For a test point x, the probability that its class label y=c is defined as $$p\left(y=c|x,D,K\right)=\frac{1}{K}\sum_{_{i\in}N_{k}\left(x,D\right)}^{}\left(y_{i}\right)=c_{i}$$

Where $$N_{k}\left( x,D \right)$$ are the K nearest neighbors of the test point. The estimate class label would then be defined as $$\hat{y}\left( x \right)=argmax_{c}p(y=c|x,D,K)$$


Euclidean-straight line distance between 2 data points in a plane:
Manhattan-distance between two strings, a and b is denoted as d(a,b).
Cosine-Cos θ distance between the two data points
The equation is:
$$\left( \sum_{i=1}^{n}|x_{i}-y_{i}|^{p} \right)^{\frac{1}{p}}$$
Where: 
p = 1, Manhattan Distance
p = 2, Euclidean Distance
p = infinity, Chebychev Distance

Linkage
The clusters are formed by using different types of criteria or known as linkage functions. Linkage methods use the affinity that we discussed above.

The different linkage methods produce different results of hierarchical clustering, they are listed below :

Single-merge in each step the two clusters whose two closest members have the smallest distance 
Complete-merge in each step the two clusters whose merger has the smallest diameter
Average-compromise between the sensitivity of complete-link clustering to outliers and the tendency of single-link clustering to form long chains that do not correspond to the intuitive notion of clusters as compact, spherical object
Wards-increase in the "error sum of squares" (ESS) after fusing two clusters into a single cluster
Error Sum of Squares: $$ESS=\sum_{i}^{}\sum_{j}^{}\sum_{k}^{}|x_{ijk}-\bar{x}_{i\cdot k}|^{2}$$

### 3.1.2 Application

In this case we are interested in the development trend of chip’s  Process Size, Die Size, Transistors and Freq. So we use these four variables as our input, we generate a Hierarchical Dendrogram which is show below:

In the above dendrogram graph, such a vertical line is the blue line. We now draw a horizontal line across this vertical line as shown below. This horizontal line cuts the vertical line at two places, and this means the optimal number of clusters is 4.

We are then able to run the AgglomerativeClustering module of sklearn.cluster package to create flat clusters by passing no. of clusters as 4 (determined in the above section). Again we use euclidean and ward as the parameters.
By the cluster method stated as above , we are able to obtain the four clusters from the Agglomerative Clustering as below:

![alt text](https://github.com/Yuchen-PLB/ECS171-FP/blob/main/pictures/Tree.png)

#### 3.1.3 Interpretation

From the above Clusters we are able separate the CPU data into 3 subset-

1. Early developed CPU
	
2. AMD High perfermance CPU
	
3. Advanced CPU
	
By looking into this subset we found that-
	
The development of chip manufacture technology has greatly decreased the Process Size of the chip, and brought revolutionary improvement on the CPU. But the process size is reaching the limit of Silicon's atomic size is about 0.2 nanometers. Although, the regression model we discussed before comes with the idea that this thing is not likely to happen in the coming 30-40 years, after the process size reaches that limit, there would not be any breakthrough on chip performance. 
	
So by this model, we can predict that in the next 5~8 years, the Moore's will Law still holds, but there will be gradually more and more differences between AMD and Intel chip manufacturing techniques. We can also foresee when Samsung comes up with their 3-nm manufacturing technique there will be a big improvement of CPU and GPU performance.



### 3.3.4 Method 2: Principal component analysis (PCA)

PCA is an unsupervised pre-processing task that is carried out before applying any ML algorithm. PCA is based on “orthogonal linear transformation” which is a mathematical technique to project the attributes of a data set onto a new coordinate system. The attribute which describes the most variance is called the first principal component and is placed at the first coordinate.

Similarly, the attribute which stands second in describing variance is called a second principal component and so on. In short, the complete dataset can be expressed in terms of principal components. Usually, more than 90% of the variance is explained by two/three principal components.

Principal component analysis, or PCA, thus converts data from high dimensional space to low dimensional space by selecting the most important attributes that capture maximum information about the dataset.

### 3.3.5 Method

Principal component analysis (PCA) is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation, increasing the interpretability of data while preserving the maximum amount of information, and enabling the visualization of multidimensional data. Formally, PCA is a statistical technique for reducing the dimensionality of a dataset. This is accomplished by linearly transforming the data into a new coordinate system where (most of) the variation in the data can be described with fewer dimensions than the initial data.

The principal components of a collection of points in a real coordinate space are a sequence of p unit vectors, where the i-th vector is the direction of a line that best fits the data while being orthogonal to the first i-1 vectors. Here, a best-fitting line is defined as one that minimizes the average squared perpendicular distance from the points to the line. These directions constitute an orthonormal basis in which different individual dimensions of the data are linearly uncorrelated. Principal component analysis (PCA) is the process of computing the principal components and using them to perform a change of basis on the data

![68850f0e63c154dd348b9f3ff18e26b](https://user-images.githubusercontent.com/118629117/205845268-8d0e8ebd-b473-4b64-93ff-4dc26ba83f1a.png)

The k-th component can be found by subtracting the first k − 1 principal components from X-

![689b2449539019c66c0dbfa4e31d4e9](https://user-images.githubusercontent.com/118629117/205845328-0f55a50f-056a-481d-82db-1f46029051d9.png)

### 3.3.6 Application

It is clear that the dataset has 1543  data items with 4 input attributes. There are Three output classes-benign and malignant. Due to 4 input features, it is impossible to visualize this data. while the dimension of actual data is (1543,4). Thus, it is clear that with PCA, the number of dimensions has reduced to 3 from 4.
 
Plot the principal components for better data visualization.  Though we had taken n_components =3, here we are plotting a 2d graph as well as 3d using first two principal components and 3 principal components respectively. For three principal components, we need to plot a 3d graph. The colors show the 2 output classes of the original dataset-benign and malignant. It is clear that principal components show clear separation between two output classes. 
 
For three principal components, we need to plot a 3d graph. x[:,0] signifies the first principal component. Similarly, x[:,1] and x[:,2] represent the second and the third principal component.
	
![alt text](https://github.com/Yuchen-PLB/ECS171-FP/blob/main/pictures/pca1.png)

![alt text](https://github.com/Yuchen-PLB/ECS171-FP/blob/main/pictures/pca2.png)

### 3.3.7 Interpretation:

![alt text](https://github.com/Yuchen-PLB/ECS171-FP/blob/main/pictures/pca3.png)

From the graph above, we found that the two variables that are most related to the chip performance are Process Size and Freq, which means most other factors are corresponding with the chip’s process size and chip frequency, the advancement in the chip process size and frequency lead to the development of chip performance. 

By the PCA, we also discover that the factor  Process Size and Freq are not perfectly perpendicular to each other, which means that the development in  Process Size is not always corresponding with the Freq. Actually the angle between them is slightly over 90^{。}That is because as we decrease the Process Size, the processor actually becomes more fragile under high voltage, which restricts the optimized Freq for the chip.

From this perspective, the future trend of chip development in the next 5-7 years will still be the competition of more advanced processor manufacturing techniques and higher frequency. The limit of chip performance will not be reached until they reach the bottleneck of manufacturing technique and chip frequency.
	
## Result

## Discussion
1. According to the result showed with the linear regression, we can clearly see that the CPU grows linearly for the first 10 years since 2000, this matches what 
moore's law indecated: "number of transistors doubles every year". However, recently, number of transistors grows much faster, which can be proved by the polynomial regression graph. This also shows that the technology develops much faster.

2. The ANN model did a fair job in predicting the brands of the CPUs, which shows that there exist distinct differences between difference CPU vendors. This model can be used to further explore the sepcialities for a specific chip vendor and evaluate the overall performance of the chip in this vendor.

## Conclusion

The accelerating speed of business operations paired with constantly rising customer expectations means that for many organizations, decision making must be progressively devolved away from headquarters.
	
In some cases, these decisions may need to be entirely automated. Increasingly, decisions are being made based on data generated at the edge.  Putting compute capabilities closer to this relies on the effective combination of three technologies: edge computing, the cloud and artificial intelligence (AI).  While all three already add value individually, which means in the foreseeable future, there will be a sharp rise in the demand of hashrate. Apparently those enterprises are in urgent need of cheap and high performance chips. But the question is: will Moore's Law still hold? What will be the limit of chip performance?
	
In our project, we perform Multiple Linear Regression (MLR) and polynomial regression to build up a basic regression model of CPU/GPU development. We have built up a categorical Neural Network (NN) on the CPU to predict the chip’s Vendor. By applying the sequential model, we are able to reach around 0.85 accuracy, and proves our assumption that the performance of the chips grows faster in the recent years for both vendors. Our neural network has determined the developmental trend of the major vendors and we are able to predict how the next generation CPU/GPU will be in the near future.

We also applied Clustering and PCA on both CPU & GPU. By the model,we believe the future trend of chip development in the next 5-7 years will still be the competition of more advanced processor manufacturing techniques and higher frequency. The limit of chip performance will not be reached until they reach the bottleneck of manufacturing technique and chip frequency.

## Reference:
[1]kmeans clustering algorithm - Python. (n.d.). https://pythonprogramminglanguage.com/kmeans-clustering-algorithm/

[2]Kumar, B. (2021, October 21). *Pandas Delete Column. Python Guides*. https://pythonguides.com/pandas-delete-column/

[3]Taskesen, E. (2022, September 27). A practical guide for getting the most out of Principal Component Analysis. | *Towards Data Science.* Medium. https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559

[4]pca. (2022, November 1). PyPI. https://pypi.org/project/pca/

[5]Real Python. (2022, September 1).* K-Means Clustering in Python: A Practical Guide*. https://realpython.com/k-means-clustering-python/

[6]*How to Do Hierarchical Clustering in Python ? 5 Easy Steps Only.* (2021, April 11). Data Science Learner. https://www.datasciencelearner.com/how-to-do-hierarchical-clustering-in-python/

[7]Extracting specific rows from a data frame. (2017, August 7). Stack Overflow. https://stackoverflow.com/questions/45552952/extracting-specific-rows-from-a-data-frame

[8]Comparing Kmeans and Agglomerative Clustering. (2022, May 17). Stack Overflow. https://stackoverflow.com/questions/72272929/comparing-kmeans-and-agglomerative-clustering

Especially thanks for Dr.Sebastian Kühnert in STA 141 provide additional explanation on PCA (in R)

## Contribution

Tingwei Liu: Write code for data preprocessing, build correlation chart on dataset, write code for data normalization, build and explain ANN models, introduce the group members to each other.

Yuchen Liu : Write code for data transformation and data formating, build and explain Clustering models, build and explain PCA models, reserve meeting rooms for the group.

Keer (Nicole) Ni: Organize the Github folders and files, delte repeative codes, separate data preprocessing and model training codes, tidy the Readme section on Github, debug and correct the data transformation for 'Release Date' label, organize the code files to subsections for readability, build and explain Regression (Linear, Polynomial, Logistic) models.

Kehuan Wang: Check the code, query relevant code cases as an aid, supplement the required regional model and transfer the code to Yuchen Liu for sorting, and standardize the overall code structure.
