# ECS171-FP

Project from UC Davis Fall 2022 ECS171 - Introduction to Machine Learning

By: Keer (Nicole) Ni, Tingwei Liu, Yuchen Liu, and Kehuan Wang.

## Link to the dataset (from Kaggle)
https://www.kaggle.com/datasets/michaelbryantds/cpu-and-gpu-product-data

## Information about the dataset being used
This dataset contains features including: Product (name of the product), Type (CPU or GPU) Release Date (release date of the product), Process Size (this is the physical size of the components that makes up a chip), TDP (thermal design point), Die Size (physical dimension of a bare die), Transistors (number of transistors), Freq (frequency in MHz), Foundry (company that makes the CPU/GPU), Vendor (seller of the CPU/GPU).

In this dataset, the Type could be considered as a label/category for other features (e.g. we could predict the Type (is it a CPU/GPU) given other features (Process Size, TDP, Die Size, Freq)). i.e. Which architecture has better performances. Other categorical features such as Foundry and Vendor could also be considered as labels/categories (using other numerical features to predict the Foundry or Vendor for example). In addition, the release date can also be considered as a label. According to Moore's law, there is a relationship between releasing date and number of transistors, which affect the performance of CPU/GPUs.

## Objectives and Goals
1. We will do visualized analysis on the data set to discover the trend of CPU and GPU  development.
2. We are expected to use the visualized data to explain simple questions such as: which company is producing the better CPUs and which company is producing the better GPUs.
3. Using neural networks to discuss the relationship between different variables, we are expected to find a general trend of CPU & GPU development. We will focus on the building up models about the relationship between the performance and the different categorical variables and the release date. 
4. We will look deeper into the model and explain :
    1. Will Moore's Law still hold in the next 5-8 years?
    2. Will Dannard Scaling be still valid in the next 5-8 years?
    3. What is most related to CPU/GPU performance? Is it the transistors size, die size, or frequency? What is the relationship between them?
    4. What would be the tread of future CPU/GPU development? Are their sizes getting smaller? Is their frequency going to increase? What would be the limit of chip performance?

## Steps to Take
1. Set up a Github repository to track updates and progress of the final project.
2. Data exploration (e.g. Histogram, Pair Plot, Q-Q Plot, Line Plot, t-SNE Plot…).
3. Data preprocessing (e.g. remove rows with null data, encode categorical data into numerical values if needed, data scaling (normalization, standardization), data imputation (consider to impute the missing data if don’t want to remove rows with null value), …).
4. Choose appropriate machine learning models and train the model (linear regression, polynomial regression, neural network, logistic regression, …).
5. Construct clear write-ups (include goals, results, and discussion) to report our findings. 

## Part 1. Data exploration
NOTE: We are using Google Colab as the coding environment. To read the dataset correctly, please upload the **chip_dataset.csv** file to the folder in Google Colab and progress from there.
1. We took at look at the different features and their maximum & minimum values.
2. We transformed the time values of the 'Date' feature to make it easier to plot and analysis.
3. We splitted the dataset into two main categories: CPU and GPU. Since the GPU category/class has more feature than the CPU feature. 
4. We look at seaborn pairplots and correlation coefficients to see the data distribution.


## Part 2. Data Preprocessing and Train First Model (Linear Regression and Polynomial Regression)
1. We first clean the dataframe by deleting unused column 'Unnamed: 0' (only shows indecies of the dataframe).
2. We normalize the dataset using Minmax Normalization.
3. We encode the categorical column/features name 'Vendor' which will be used in other models later. 
4. We build and train the Linear Regression model and Polynomial Regression model (up to degree 3).
5. We report R^2 which is the percentage of variation explain/reveal by the regression model. 
6. We run a logistic regression on the categorical data, trying to explain which Vendor is tending to produce higher quality CPU/GPU.

## ToDos for more Data Preprocessing and more Model Training:
1. We decided to apply missing value imputation (MissForest) to impute the missing values in the dataset. This is because the there are too many missing values in the dataset and removing them will result in having a dataset not large enough for training and testing the model.
2. We decided to apply logistic regression to make prediction for 'Vendor' feature.
3. We decided to split the final project into different objectives (e.g. one of the objectives is to predict whether a given trial/sample is a CPU or GPU using the 'Type' label as y for prediction results). This will require us to preprocess the dataset differently (i.e. scale and transform only the feature values X, but separate the predicting values y without using any scaling or transforming method to it). We will look more into the dataset and decide to use other models (logistic regression, neural network, decision tree, SVM).
4. We are trying to do clusting and PCA on the data to discuss the impact of different variables and use them to reduce the model


