import os
import __root__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# STEP 1
# importing dataset
resource = os.path.join(__root__.PROJECT_ROOT, 'data/data_preprocessing/')
csv = os.path.join(resource, 'Data.csv')
# reading dataset with read.csv
dataset = pd.read_csv(csv)
# selecting columns and row with iloc
# first index is row and second index is column
# independent varialble
# because we need to work with numpy array so we must separate the dataset
x = dataset.iloc[:, :-1].values
# dependent variable
y = dataset.iloc[:, 3].values

# STEP 2 (optional)
# taking care of missing data
# missing value signifies missing cell identifiers
# mean strategy is default
imputer = SimpleImputer()
# fitting imputer object to the columns where missing values are present
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
# DEPCRICATION WARNING THERE IS NO NEED TO USE LABEL ENCODER FIRST AND THEN ONEHOTENCODER
# YOU CAN DIRECTLY USE ONEHOTENCODER WITH COLUMN TANSFORMER
# Encoding categorical data
# converting categories in integers
# labelencoder_x = LabelEncoder()
# fitting labelencoder object to Country column of dataset
# x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

# STEP 3 (optional)
# dummy encoding of categories
# because the encoded categories might contains relational order with number
# to avoid this we need to convert this encoded column into separate columns
# using OneHotEncoder to create dummy variable
# specify the which column to onehot encode
# DEPCRICATION WARNING THERE IS NO NEED TO USE LABEL ENCODER FIRST AND THEN ONEHOTENCODER
# YOU CAN DIRECTLY USE ONEHOTENCODER WITH COLUMN TANSFORMER
# onhotencoder = OneHotEncoder(categorical_features=[0])
# x = onhotencoder.fit_transform(x).toarray()
# THIS IS HOW YOU CAN USE ONEHOTENCODER DIRECTOLY
ct = ColumnTransformer(
    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    # CATEGORICAL_FEATURES KEYWORD IS DEPRICATED
    # keywords: Transformer name, transformer, column
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    # Leave the rest of the columns untouched
    remainder='passthrough'
)
x = ct.fit_transform(x)
# label encoding to integer for y vector
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# STEP 4(IMPORTANT)
# spliting the dataset into the training set and test set
# this is splitting of dataset for training of model and testing of model
# test size define what percent of dataset goes into testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# STEP 5 (Depends on context)
# feature scaling
sc_x = StandardScaler()
# train set needs to be fit and transform
x_train = sc_x.fit_transform(x_train)
# we dont need to fit and transform for test set
x_test = sc_x.transform(x_test)
# dependent vectors do not need scaling
