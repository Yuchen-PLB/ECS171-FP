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

## Figures and Data exploration

### Part 1. Data exploration
NOTE: We are using Google Colab as the coding environment. To read the dataset correctly, please upload the **chip_dataset.csv** file to the folder in Google Colab and progress from there.
1. We took at look at the different features and their maximum & minimum values.
2. We transformed the time values of the 'Date' feature to make it easier to plot and analysis.
3. We splitted the dataset into two main categories: CPU and GPU. Since the GPU category/class has more feature than the CPU feature. 
4. We look at seaborn pairplots and correlation coefficients to see the data distribution.

## Methods

### 1. Data Preprocessing and Train First Model (Linear Regression and Polynomial Regression)
1. We first clean the dataframe by deleting unused column 'Unnamed: 0' (only shows indecies of the dataframe).
2. We normalize the dataset using Minmax Normalization.
3. We encode the categorical column/features name 'Vendor' which will be used in other models later. 
4. We build and train the Linear Regression model and Polynomial Regression model (up to degree 3).
5. We report R^2 which is the percentage of variation explain/reveal by the regression model. 
6. We run a logistic regression on the categorical data, trying to explain which Vendor is tending to produce higher quality CPU/GPU.


### 2. Vendor Prediction with ANN Model
1. This model predict the brand of CPU/GPU with expect to its 'Release Date	Process Size (nm)	TDP (W)	Die Size (mm^2)	Transistors (million)	Freq (MHz)'.
2. We enumerate the vendor to 0, 1, 2 as our y-label and normalize the properties mentioned above as our X.
3. We build a 3-layer ANN model with activation relu and signmoid, train for 1000 epoch.
4. The model reaches 0.85 accuracy at the end.

##Clustering
#3.1Agglomerative Hierarchical Clustering
3.1.1 General introduction to Hierarchical Clustering
Hierarchical clustering is an unsupervised clustering algorithm used to create clusters with a tree-like hierarchy. In this clustering method, there is no need to give the number of clusters to the algorithm.  In contrast to this, the other algorithm like K-Mean produces flat clusters where there is no hierarchy and we also have to choose the number of clusters, to begin with.

Here, we first draw a Hierarchical Dendrogram to have a general overview of the CPU data to decide the number of clusters we need for the K-Mean clustering algorithm.

The hierarchical clustering algorithm can be of two types –
Divisive Clustering – It takes a top-down approach where the entire data observation is considered to be one big cluster at the start. Then subsequently it is split into two clusters, then three clusters, and so on until each data ends up as a separate cluster.
Agglomerative Clustering – It takes a bottom-up approach where it assumes individual data observation to be one cluster at the start. Then it starts merging the data points into clusters till it creates one final cluster at the end with all data points.



3.1.2 Method
Parameters of Agglomerative Clustering
The agglomeration hierarchical clustering can have multiple variations depending on affinity and linkage.
Affinity
Affinity denotes the method using which the distance or similarity between data points or clusters is calculated. Which include –

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

3.1.3 Application
In this case we are interested in the development trend of chip’s  Process Size, Die Size, Transistors and Freq. So we use these four variables as our input, we generate a Hierarchical Dendrogram which is show below:

In the above dendrogram graph, such a vertical line is the blue line. We now draw a horizontal line across this vertical line as shown below. This horizontal line cuts the vertical line at two places, and this means the optimal number of clusters is 4.

We are then able to run the AgglomerativeClustering module of sklearn.cluster package to create flat clusters by passing no. of clusters as 4 (determined in the above section). Again we use euclidean and ward as the parameters.
By the cluster method stated as above , we are able to obtain the four clusters from the Agglomerative Clustering as below:

3.1.4 Interpretation:
From the above Clusters we are able to identify few thing:
	1.


## Result

## Discussion
1. According to the result showed with the linear regression, we can clearly see that the CPU grows linearly for the first 10 years since 2000, this matches what 
moore's law indecated: "number of transistors doubles every year". However, recently, number of transistors grows much faster, which can be proved by the polynomial regression graph. This also shows that the technology develops much faster.

2. The ANN model did a fair job in predicting the brands of the CPUs, which shows that there exist distinct differences between difference CPU vendors. This model can be used to further explore the sepcialities for a specific chip vendor and evaluate the overall performance of the chip in this vendor.

## Conclusion

## ToDos for more Data Preprocessing and more Model Training:
1. We decided to apply missing value imputation (MissForest) to impute the missing values in the dataset. This is because the there are too many missing values in the dataset and removing them will result in having a dataset not large enough for training and testing the model.
2. We decided to apply logistic regression to make prediction for 'Vendor' feature.
3. We decided to split the final project into different objectives (e.g. one of the objectives is to predict whether a given trial/sample is a CPU or GPU using the 'Type' label as y for prediction results). This will require us to preprocess the dataset differently (i.e. scale and transform only the feature values X, but separate the predicting values y without using any scaling or transforming method to it). We will look more into the dataset and decide to use other models (logistic regression, neural network, decision tree, SVM).
4. We are trying to do clusting and PCA on the data to discuss the impact of different variables and use them to reduce the model


