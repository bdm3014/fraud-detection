# %%
import pandas as pd
import gc
import numpy as np
import math
import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as stats
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime
import re
import shap

from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import catboost
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
%matplotlib inline

# Standard plotly imports
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)


import warnings
warnings.filterwarnings("ignore")

import gc
gc.enable()

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import os

# %% [markdown]
# ##### REFERENCE NOTEBOOKS
# https://www.kaggle.com/code/kabure/extensive-eda-and-modeling-xgb-hyperopt  
# https://www.kaggle.com/code/yw6916/lgb-xgb-ensemble-stacking-based-on-fea-eng/notebook  
# https://www.kaggle.com/code/iasnobmatsu/xgb-model-with-feature-engineering/notebook     
# https://www.kaggle.com/code/davidcairuz/feature-engineering-lightgbm#About-this-kernel

# %% [markdown]
# #### Functions

# %%
def make_corr(variables, data, figsize=(10, 15)):
    if isinstance(variables, pd.DataFrame):
        variables = variables['Column Name'].tolist()

    cols = variables

    corr_matrix = data[cols].corr()

    # Create a heatmap with the specified figsize
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap='RdBu_r', annot=True, center=0.0)

    plt.title('Correlation Heatmap for Columns Starting with C')
    plt.show()
    
# We will focus on each column in detail
# Uniqe Values, DTYPE, NUNIQUE, NULL_RATE
def column_details(regex, df):
  
    global columns
    columns=[col for col in df.columns if re.search(regex, col)]

    from colorama import Fore, Back, Style

    print('Unique Values of the Features:\nfeature: DTYPE, NUNIQUE, NULL_RATE\n')
    for i in df[columns]:
        color = Fore.RED if df[i].dtype =='float64' else Fore.BLUE if df[i].dtype =='int64' else Fore.GREEN
        print(f'{i}: {color} {df[i].dtype}, {df[i].nunique()}, %{round(df[i].isna().sum()/len(df[i])*100,2)}\n{Style.RESET_ALL}{pd.Series(df[i].unique()).sort_values().values}\n')

def null_values(df, rate=0):
    """a function to show null values with percentage"""
    nv=pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/df.shape[0]],axis=1).rename(columns={0:'Missing_Records', 1:'Percentage (%)'})
    return nv[nv['Percentage (%)']>rate].sort_values('Percentage (%)', ascending=False)

#Plot Functions

def plot_col(col, df, figsize=(20, 6)):
    """
    Function to create a pair of subplots containing two graphs based on a specified column.

    Left Graph (First Subplot):
    - Draws a bar graph representing the percentage of Fraud cases with respect to the specified column.
    - Uses two colors (0 and 1) to represent Fraud and Non-Fraud cases.
    - Adds a second line graph on the same column, representing the percentage of Fraud cases.

    Right Graph (Second Subplot):
    - Draws a bar graph representing the number of unique values in the dataset based on the specified column.

    The purpose of this function is to visualize the relationship between Fraud status, unique values, and missing values in a specific column.

    :param col: Name of the column to be visualized.
    :param df: Dataset.
    :param figsize: Size of the created figure.
    """

    # Create a copy of the DataFrame to ensure the original DataFrame is not modified
    df_copy = df.copy()

    # Handle NaN values by filling them with 'Missing' in the copied DataFrame
    df_copy[col] = df_copy[col].fillna('Missing')

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Left Graph: Bar graph and line graph for Fraud percentages
    plt.subplot(121)
    tmp = pd.crosstab(df_copy[col], df_copy['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    ax[0] = sns.countplot(x=col, data=df_copy, hue='isFraud', order=np.sort(df_copy[col].unique()))
    ax[0].tick_params(axis='x', rotation=90)

    ax_twin = ax[0].twinx()
    ax_twin = sns.pointplot(x=col, y='Fraud', data=tmp, color='black', order=np.sort(df_copy[col].unique()))

    ax[0].grid()

    # Right Graph: Bar graph for the number of unique values in the column
    plt.subplot(122)
    ax[1] = sns.countplot(x=df_copy[col], order=np.sort(df_copy[col].unique()))

    plt.show()


#correlation functions
#for xgboost : For perfectly correlated variables(100%), there is no impact on model performance — neither on train and nor on validation dataset. 
# Also, there is no change in variable importance and rank order 
# In case of partially correlated features, the output of XGBoost is slightly impacted. 
# We see a marginal change in the performance of the model, suggesting the robustness of XGBoost when dealing with correlated variables.
# However, one may note that the partially correlated variables in the model are affecting the variable importance.
# reference link: https://vishesh-gupta.medium.com/correlation-in-xgboost-8afa649bd066 

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Check distinct values for each correlated pair
                distinct_values_col = len(x[col[0]].unique())
                distinct_values_row = len(x[row[0]].unique())

                # Keep the one with more distinct values
                if distinct_values_col > distinct_values_row:
                    drop_cols.append(row.values[0])
                else:
                    drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return drops

# References:
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

def cramers_v(x, y):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = sts.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

#outlier functions
def simplify_column(col, df, threshold=0.005, value='mode'):
  df[col] = df[col].replace(df[col].value_counts(dropna=True)[df[col].value_counts(dropna=True, normalize=True)<threshold].index,df[col].mode()[0] if value=='mode' else 'other')
  return df[col]

def identify_collinear_categorical_features(df, columns, threshold):
    """
    Objective:
        Identify collinear categorical features in a dataframe with Cramér's V greater than the threshold.

    Inputs:
        df: dataframe
        columns: list of column names to check for collinearity
        threshold: features with Cramér's V greater than this value are considered collinear

    Output:
        list of columns to drop
    """
    # Create an empty DataFrame to store the results
    cramers_v_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    # Fill in the Cramér's V values for each pair of columns
    for col1 in columns:
        for col2 in columns:
            cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

    # Identify columns to drop based on Cramér's V threshold
    drop_cols = set()
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i < j and cramers_v_matrix.loc[col1, col2] > threshold:
                drop_cols.add(col2)

    return list(drop_cols)

def remove_collinear_categorical_features(df, drop_cols):
    """
    Objective:
        Remove collinear categorical features from a dataframe.

    Inputs:
        df: dataframe
        drop_cols: list of columns to drop

    Output:
        dataframe that contains only the non-highly-collinear features
    """
    # Drop the identified columns
    df = df.drop(columns=drop_cols)

    return df

#Encoders
# Frequency Encoding

def frequency_encoding(train, test, columns, self_encoding=False):
    for col in columns:
        df = pd.concat([train[[col]], test[[col]]])
        fq_encode = df[col].value_counts(dropna=False, normalize=True).to_dict()
        if self_encoding:
            train[col] = train[col].map(fq_encode)
            test[col]  = test[col].map(fq_encode)            
        else:
            train[col+'_freq_encoded'] = train[col].map(fq_encode)
            test[col+'_freq_encoded']  = test[col].map(fq_encode)
    return train, test

#Modeling
def plot_feature_importances(model, num=10, figsize=(20,10)):
    feature_imp = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)[:num]
    plt.figure(figsize=figsize)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title("Feature Importance")
    plt.show()

# %% [markdown]
# * Data is separated into two datasets: customer identity information and transaction information. 
# * Not all transactions are associated with available identities. 
# * Unique key for both tables is TransactionID. It is duplicated in transaction table, it is unique in identity table.

# %% [markdown]
# #### Transaction Dataset

# %%
# Importing transaction data
# We are standardizing the column types in accordance with the data definition.

# Define column names for the dataset
cols_t = ['TransactionID', 'TransactionDT', 'TransactionAmt',
   'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
   'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
   'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
   'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
   'M5', 'M6', 'M7', 'M8', 'M9']

# Generate column names for the 'V' features (V1 to V339)
cols_v = ['V'+str(x) for x in range(1, 340)]

# Define data types for the 'V' features as float32
types_v = {c: 'float32' for c in cols_v}

# Specify the columns that need to be converted to the 'object' data type
columns_to_convert_to_object = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                                'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 
                                'M5', 'M6', 'M7', 'M8', 'M9']

# Read the data from the CSV file into a DataFrame (train)
transaction = pd.read_csv(r'C:\Fraud_Data\data\train_transaction.csv',
                    usecols=cols_t+['isFraud']+cols_v, 
                    dtype={**types_v, **{col: 'object' for col in columns_to_convert_to_object}}, index_col='TransactionID')



# %%
# The `TransactionDT` feature represents a timedelta from a specific reference datetime, rather than an actual timestamp. 
# It measures the time elapsed since the reference datetime in a timedelta format.

# %%
# Getting a real format of transaction date

# Predefined start date
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

# Define the date range
dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

# Create 'DT' column using the 'TransactionDT' column
transaction['DT'] = transaction['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))

# Convert 'DT' column to 'DatetimeIndex' object
transaction['DT'] = pd.to_datetime(transaction['DT'])


# %%
# Sorting the DataFrame based on the 'DT' column
transaction = transaction.sort_values(by='DT')

# %% [markdown]
# #### TRAIN-TEST SPLIT BASED ON TRANSACTION DATE IN TRANSACTION DATA

# %%
# Splitting train-test (TRAIN 75% TEST 25%) I will then merge these tables with identity table by using TransactionID.
train_index = transaction.index[:3 * len(transaction) // 4]
test_index = transaction.index[3 * len(transaction) // 4:]

# Splitting train_transaction and test_transaction based on indices
train_transaction = transaction.loc[train_index]
test_transaction = transaction.loc[test_index]

# %%
train_transaction.head()

# %%
test_transaction['DT'].isnull().sum()

# %% [markdown]
# #### Identity Dataset

# %%
# Importing identity data
# Define column names for the dataset
cols_t = ['TransactionID','DeviceInfo','DeviceType','id_38','id_37','id_36','id_35','id_34','id_33','id_32','id_31','id_30','id_29','id_28',
          'id_27','id_26','id_25','id_24','id_23','id_22','id_21','id_20','id_19','id_18','id_17','id_16','id_15','id_14','id_13',
          'id_12','id_11','id_10','id_09','id_08','id_07','id_06','id_05','id_04','id_03','id_02','id_01']

# Specify the columns that need to be converted to the 'object' data type
columns_to_convert_to_object = ['DeviceInfo','DeviceType','id_38','id_37','id_36','id_35','id_34','id_33','id_32','id_31','id_30','id_29','id_28',
          'id_27','id_26','id_25','id_24','id_23','id_22','id_21','id_20','id_19','id_18','id_17','id_16','id_15','id_14','id_13',
          'id_12']

# Read the data
identity = pd.read_csv(
    r'C:\Fraud_Data\data\train_identity.csv',
    usecols=cols_t,
    dtype=dict.fromkeys(columns_to_convert_to_object, 'object'), 
    index_col='TransactionID'
)

# %%
identity.head()

# %%
# Check for duplicated Transaction IDs when TransactionID is the index - all transaction ids unique in identity table, this will be elobrated while merging the datasets
print('Length of Transaction IDs:', identity.index.shape[0])
print('Number of unique Transaction IDs:', identity.index.nunique())


# %%
# Check if the index of train_transaction is present in the identity index
is_in_transaction = identity.index.isin(transaction.index)

# Filter out the indices that are not present
not_in_transaction = identity.index[~is_in_transaction]

# Display the indices that are not present and their count
print("Indices not present and their count:")
print(not_in_transaction)
print("Total count of indices not present:", len(not_in_transaction))


# %% [markdown]
# #### Merging Transaction and Identity Data

# %%
# Merging datas

# Merge train_transaction and identity
train = pd.merge(train_transaction, identity, how='left', left_index=True, right_index=True)

# Merge test_transaction and identity
test = pd.merge(test_transaction, identity, how='left', left_index=True, right_index=True)

print("Train: ", train.shape)
print("Test: ", test.shape)

# Delete transaction, train_transaction, test_transaction, identity
del transaction, train_transaction, test_transaction, identity


# %%
# Train start-end date
print('min Transaction Date: ',min(train['DT'].values))
print('max Transaction Date: ',max(train['DT'].values))

# %%
# Test start-end date
print('min Transaction Date: ',min(test['DT'].values))
print('max Transaction Date: ',max(test['DT'].values))

# %%
# Check for the duplicated dates-train
print('length of Transaction Date',train['DT'].shape[0] )
print('length of unique Transaction Date', train['DT'].nunique())

# %%
# Check for the duplicated dates-test
print('length of Transaction Date',test['DT'].shape[0] )
print('length of unique Transaction Date', test['DT'].nunique())

# %%
train.head()

# %%
# Performing garbage collection to release memory occupied by unused objects
gc.collect()


# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_1.pkl')

#save 'test' data to a pickle file named 'test_1.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_1.pkl')


# %%
# Read the 'train_1.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_1.pkl')

# Read the 'test_1.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_1.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %%
print(f'There are {train.isnull().any().sum()} columns in train have null values.')

# %%
print(f'There are {test.isnull().any().sum()} columns in test have null values.')

# %% [markdown]
# #### **Target Variable Distribution**

# %% [markdown]
# The original dataset is characterized by a significant class imbalance, predominantly consisting of non-fraudulent transactions. This imbalance, where only 3.5% of transactions are labeled as fraud, poses a challenge for developing accurate predictive models and conducting analyses. Utilizing such an imbalanced dataset as the basis for machine learning models may lead to substantial errors and the risk of overfitting.
# 
# The risk of overfitting arises from the potential for algorithms to incorrectly assume that the majority of transactions are not fraudulent. In contrast to making assumptions, the primary objective is to develop models capable of identifying patterns indicative of fraudulent activities.
# 
# In the machine learning context, class imbalance denotes a considerable disparity in the number of data points representing different classes. Addressing this imbalance is paramount to ensure that models are not misled by the prevalence of non-fraudulent instances. The goal is to enable models to accurately identify patterns associated with fraudulent transactions.
# 
# In the training set, approximately 96.49% of transactions are labeled as non-fraudulent (isFraud==0), while around 3.51% are identified as fraudulent (isFraud==1). Similarly, in the test set, approximately 96.55% of transactions are non-fraudulent, and about 3.45% are labeled as fraudulent.
# 
# This class distribution indicates a high prevalence of non-fraudulent transactions in both the training and test sets, with a relatively small percentage of transactions being classified as fraudulent. This reinforces the presence of a significant class imbalance, which should be considered when developing and evaluating predictive models.
# 

# %%
# Train Data
# Count the occurrences of each class in the 'isFraud' column
class_counts = train['isFraud'].value_counts()

# Calculate the percentage distribution
class_percentages = class_counts / len(train) * 100

# Plot the class distribution using Matplotlib
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="mako")

# Adding percentages above the bars
for i, value in enumerate(class_counts.values):
    plt.text(i, value + 50, f'{class_percentages[i]:.2f}%', ha='center', va='bottom', fontsize=10, color='black')

plt.title("Train Data Imbalance - isFraud")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# %%
# Test Data
# Count the occurrences of each class in the 'isFraud' column
class_counts = test['isFraud'].value_counts()

# Calculate the percentage distribution
class_percentages = class_counts / len(test) * 100

# Plot the class distribution using Matplotlib
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="mako")

# Adding percentages above the bars
for i, value in enumerate(class_counts.values):
    plt.text(i, value + 50, f'{class_percentages[i]:.2f}%', ha='center', va='bottom', fontsize=10, color='black')

plt.title("Test Data Imbalance - isFraud")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# %%
del class_counts
gc.collect()

# %% [markdown]
# #### Handling Missing Values, Feature Elimination By 3 Criterias

# %% [markdown]
# Datasets have too much missing values.

# %%
# Total missing values of the train data
missing_count = train.isnull().sum()
cell_counts = np.product(train.shape)
missing_sum = missing_count.sum()
print ("%",(round(missing_sum/cell_counts,2)) * 100)

# %%
# Total missing values of the test data
missing_count = test.isnull().sum()
cell_counts = np.product(test.shape)
missing_sum = missing_count.sum()
print ("%",(round(missing_sum/cell_counts,2)) * 100)

# %%
# columns with no nulls of train (20 columns have no nulls, rest have)
null_counts = train.isnull().sum()

columns_with_no_null = null_counts[null_counts == 0].index

print("\nColumns with No Null Values-train:")
print(columns_with_no_null)


# %%
# columns with no nulls of test
null_counts = test.isnull().sum()

columns_with_no_null = null_counts[null_counts == 0].index

print("\nColumns with No Null Values-test:")
print(columns_with_no_null)

# %%
del missing_count, cell_counts, missing_sum, null_counts, columns_with_no_null

# %% [markdown]
# 
# It has been observed that a significant number of columns exhibit a pattern of "correlated missing values," where certain rows share missing values in corresponding positions across multiple columns.
# Most of them are in the same variable group, this will be a clue for highly correlated variables in the same groups to eliminate later.
# 

# %%
# Check for null percentage correlated columns-train
# Create a dictionary to store columns with null percentages > 10%
null_percentage_dict = {}
nan_counts = train.isnull().sum()

for col in train.columns:
    null_percent = (nan_counts[col] / len(train)) * 100
    if null_percent >= 10:
        null_percentage_dict[null_percent] = null_percentage_dict.get(null_percent, []) + [col]

# Sort the dictionary by null percentages
sorted_null_percentage_dict = {k: v for k, v in sorted(null_percentage_dict.items(), reverse=True)}

for null_percent, columns in sorted_null_percentage_dict.items():
    print(f"####### Null Percentage = {null_percent:.2f}%")
    print(columns)


# %%
# Check for null percentage correlated columns-test
# Create a dictionary to store columns with null percentages > 10%
null_percentage_dict = {}
nan_counts = test.isnull().sum()

for col in test.columns:
    null_percent = (nan_counts[col] / len(test)) * 100
    if null_percent >= 10:
        null_percentage_dict[null_percent] = null_percentage_dict.get(null_percent, []) + [col]

# Sort the dictionary by null percentages
sorted_null_percentage_dict = {k: v for k, v in sorted(null_percentage_dict.items(), reverse=True)}

for null_percent, columns in sorted_null_percentage_dict.items():
    print(f"####### Null Percentage = {null_percent:.2f}%")
    print(columns)

# %%
# Drop columns by using 3 criteria:
# 1. If a column only has only one distinct value
# 2. If a column has more than 90% null values
# 3. If one of the categories in a column dominates more than 90% of the column
# we are looking for these criterias in train, then dropping the columns from both train and test

# Initialize lists to store columns to be dropped based on different criteria
one_value_cols, many_null_cols, big_top_value_cols = [], [], []

# Iterate through only the train DataFrame
for df in [train]:
    # Identify columns with only one distinct value
    one_value_cols += [col for col in df.columns if df[col].nunique() == 1]
    
    # Identify columns with more than 90% null values
    many_null_cols += [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]
    
    # Identify columns where a single value dominates more than 90%
    big_top_value_cols += [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

# Combine the lists of columns to be dropped, removing duplicates using set
cols_to_drop = list(set(one_value_cols + many_null_cols + big_top_value_cols))

# Check if 'isFraud' is in the list of columns to be dropped, and remove it if present
if 'isFraud' in cols_to_drop:
    cols_to_drop.remove('isFraud')

# Drop the identified columns from the train DataFrame
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

# Print the number of features that are going to be dropped for being considered useless
print(f'{len(cols_to_drop)} features are going to be dropped for being useless')


# %% [markdown]
# #### **Transaction Date**

# %%
# TransactionDT Dist for Train Transaction Data - Whole Data
time_val_whole = train['DT'].values

# Create a figure
plt.figure(figsize=(18, 5))

# Create a distribution plot for TransactionDT column - Train Data
plt.subplot(1, 3, 1)  # 1 row, 3 columns, plot at position 1
sns.kdeplot(time_val_whole, color='r', fill=True, edgecolor='black')
plt.title('Distribution of DT - Train Data', fontsize=14)
plt.xlabel('TransactionDT')

# TransactionDT Dist for Train - isFraud=0
time_val_no_fraud = train['DT'][train['isFraud'] == 0]

# TransactionDT Dist for Train - isFraud=1
time_val_fraud = train['DT'][train['isFraud'] == 1]

# Create subplots for isFraud=0 and isFraud=1
for i, time_val in enumerate([time_val_no_fraud, time_val_fraud], start=2):
    plt.subplot(1, 3, i)
    sns.kdeplot(time_val, color='r', fill=True, edgecolor='black')
    plt.title(f'Distribution of DT - isFraud={i-2}', fontsize=14)
    plt.xlabel('DT')
    

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# %%
# TransactionDT Dist for Train Transaction Data - Test Data
time_val_whole = test['DT'].values

# Create a figure
plt.figure(figsize=(18, 5))

# Create a distribution plot for TransactionDT column - Whole Data
plt.subplot(1, 3, 1)  # 1 row, 3 columns, plot at position 1
sns.kdeplot(time_val_whole, color='r', fill=True, edgecolor='black')
plt.title('Distribution of DT - Train Data', fontsize=14)
plt.xlabel('TransactionDT')

# TransactionDT Dist for Train Transaction Data - isFraud=0
time_val_no_fraud = test['DT'][test['isFraud'] == 0]

# TransactionDT Dist for Train Transaction Data - isFraud=1
time_val_fraud = test['DT'][test['isFraud'] == 1]

# Create subplots for isFraud=0 and isFraud=1
for i, time_val in enumerate([time_val_no_fraud, time_val_fraud], start=2):
    plt.subplot(1, 3, i)
    sns.kdeplot(time_val, color='r', fill=True, edgecolor='black')
    plt.title(f'Distribution of DT - isFraud={i-2}', fontsize=14)
    plt.xlabel('DT')
    

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# %% [markdown]
# Transaction_day_of_week

# %%
#creating new columns named "Transaction_day_of_week" in both the train and test.
#The resulting values will be integers representing the day of the week (Monday is 0 and Sunday is 6).
train['Transaction_day_of_week'] = train['DT'].dt.dayofweek
test['Transaction_day_of_week'] = test['DT'].dt.dayofweek

# %%
# Calculate fraud rates for each day of the week
fraud_rates = train.groupby('Transaction_day_of_week')['isFraud'].mean()

# Plot settings
plt.figure(figsize=(12, 6))
sns.countplot(x='Transaction_day_of_week', hue='isFraud', data=train, palette='viridis')

# Plot fraud rates
for i, rate in enumerate(fraud_rates):
    plt.text(i, rate * train['Transaction_day_of_week'].value_counts(dropna=False).max(), f'{rate:.2%}', ha='center', va='bottom')

# Plot labels and title
plt.title('Transaction Day of Week Distribution by Fraud Status')
plt.xlabel('Transaction Day of Week')
plt.ylabel('Count')

# Show the plot
plt.show()

# %%
# Calculate fraud rates for each day of the week
fraud_rates = test.groupby('Transaction_day_of_week')['isFraud'].mean()

# Plot settings
plt.figure(figsize=(12, 6))
sns.countplot(x='Transaction_day_of_week', hue='isFraud', data=train, palette='viridis')

# Plot fraud rates
for i, rate in enumerate(fraud_rates):
    plt.text(i, rate * test['Transaction_day_of_week'].value_counts(dropna=False).max(), f'{rate:.2%}', ha='center', va='bottom')

# Plot labels and title
plt.title('Transaction Day of Week Distribution by Fraud Status')
plt.xlabel('Transaction Day of Week')
plt.ylabel('Count')

# Show the plot
plt.show()

# %%
# Perform one-hot encoding for Transaction_day_of_week column on train
TransactionDay_OHE = pd.get_dummies(train['Transaction_day_of_week'], prefix='Transaction_day', drop_first=False, dtype=int)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, TransactionDay_OHE], axis=1)

# %%
# Perform one-hot encoding for Transaction_day_of_week column on train
TransactionDay_OHE = pd.get_dummies(test['Transaction_day_of_week'], prefix='Transaction_day', drop_first=False, dtype=int)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, TransactionDay_OHE], axis=1)

# %%
# original 'Transaction_day_of_week' column dropped
train.drop('Transaction_day_of_week', axis=1, inplace=True)
test.drop('Transaction_day_of_week', axis=1, inplace=True)

# %% [markdown]
# Transaction_hour_of_day

# %%
# extracting the hour information from the "DT" column (ranging from 0 to 23)
train['Transaction_hour_of_day'] = train['DT'].dt.hour
test['Transaction_hour_of_day'] = test['DT'].dt.hour

# %%
# Ploting transaction hour with fraud info
# Calculate fraud rates for each hour of the day
fraud_rates_hour = train.groupby('Transaction_hour_of_day')['isFraud'].mean()

# Plot settings
plt.figure(figsize=(12, 6))
sns.countplot(x='Transaction_hour_of_day', hue='isFraud', data=train, palette='viridis')

# Plot fraud rates
for i, rate in enumerate(fraud_rates_hour):
    plt.text(i, rate * train['Transaction_hour_of_day'].value_counts().max(), f'{rate:.2%}', ha='center', va='bottom')

# Plot labels and title
plt.title('Transaction Hour of Day Distribution by Fraud Status')
plt.xlabel('Transaction Hour of Day')
plt.ylabel('Count')

# Show the plot
plt.show()

# %%
# Perform one-hot encoding for Transaction_hour_of_day-train
one_hot_encoded = pd.get_dummies(train['Transaction_hour_of_day'], prefix='Transaction_hour', drop_first=False, dtype=int)

# Add the resulting one-hot encoded data to the original DataFrame
train = pd.concat([train, one_hot_encoded], axis=1)

# %%
# Perform one-hot encoding for Transaction_hour_of_day-test
one_hot_encoded = pd.get_dummies(test['Transaction_hour_of_day'], prefix='Transaction_hour', drop_first=False, dtype=int)

# Add the resulting one-hot encoded data to the original DataFrame
test = pd.concat([test, one_hot_encoded], axis=1)

# %%
# original 'Transaction_hour_of_day' column dropped
train.drop('Transaction_hour_of_day', axis=1, inplace=True)
test.drop('Transaction_hour_of_day', axis=1, inplace=True)

# %% [markdown]
# #### **Transaction Amount**

# %%
column_details(regex='TransactionAmt', df=train)

# %%
plt.figure(figsize=(12,8))
plt.subplot(121)
sns.stripplot(y='TransactionAmt', x='isFraud', data=train)
plt.axhline(6500, color='red')
plt.title('Train')

plt.subplot(122)
sns.stripplot(y='TransactionAmt', x='isFraud', data=test)
plt.axhline(6500, color='red')
plt.title('Test')

# %%
max_transaction_amount = train['TransactionAmt'][train['TransactionAmt'] < 30000].max()
print(f"The maximum transaction amount below 30000 is: {max_transaction_amount}")


# %% [markdown]
# There are outliers in Transaction_Amount with isFraud == 0. These amounts are above 30k. 
# I find the maximum amount below 30k ( 6450.97 ). I will capped the Train transaction with 6450.97 below code.

# %%
# Identify transactions with isFraud == 0 and TransactionAmt above 6450.97
condition = train['TransactionAmt'] > 6450.97

# Cap the TransactionAmt at 5368 for the identified transactions
train.loc[condition, 'TransactionAmt'] = 6450.97

# %%
train['TransactionAmt'].max()

# %%
print('Avg Transaction Amount by Frauds-Train', train[train.isFraud==1]['TransactionAmt'].mean())
print('Avg Transaction Amount by Non-Frauds-Train', train[train.isFraud==0]['TransactionAmt'].mean())
print('Avg Transaction Amount-Train',train['TransactionAmt'].mean())
print('Avg Transaction Amount-Test', test['TransactionAmt'].mean() )

# %% [markdown]
# The averages of TransactionAmt of train and test datasets are nearly same. The average of the fraud transactions(147.32) is bigger than the average of the non-fraud transactions(133.73).
# The average transaction amount for fraudulent transactions appears to be higher than that for legitimate transactions. Statistically, the average transaction amount for fraudulent transactions is 147.32 units, whereas the average amount for legitimate transactions is 133.74 units. This observation indicates that fraudulent transactions tend to involve higher amounts

# %%
#Distribution of TransactionAmt-Train
time_val = train['TransactionAmt'].values

plt.figure(figsize=(18, 4))
sns.distplot(time_val, color='r')
plt.title('Distribution of TransactionAmt-Train', fontsize=14)
plt.xlim([min(time_val)-1000, max(time_val)])  

# Show the plot
plt.show()

# %%
#Distribution of TransactionAmt-Train
time_val = test['TransactionAmt'].values

plt.figure(figsize=(18, 4))
sns.distplot(time_val, color='r')
plt.title('Distribution of TransactionAmt-Train', fontsize=14)
plt.xlim([min(time_val)-1000, max(time_val)])  

# Show the plot
plt.show()

# %%
plt.figure(figsize=(15,8))
plt.suptitle('Time of Transaction vs Amount by isFraud')
fraud_mean, nonfraud_mean = train[train.isFraud=='1']['TransactionAmt'].mean(), train[train.isFraud=='0']['TransactionAmt'].mean()
sns.scatterplot(x=train['DT'], y=train['TransactionAmt'], data=train, hue='isFraud', size="isFraud", sizes=(200, 20))
plt.axhline(y=fraud_mean ,color='red',label=f'fraud mean:{round(fraud_mean,2)}')
plt.axhline(y=nonfraud_mean, color='green',label=f'non-froud mean:{round(nonfraud_mean,2)}')
plt.legend()

plt.yscale('log')
plt.show()

# %%
# Extracting the decimal point of the transaction amount as a new variable
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

# %% [markdown]
# #### ProductCD : product code, the product for each transaction (nominal categorical)

# %%
for df in [train, test]:
  column_details(regex='ProductCD', df=df)

# %%
# Distribution of ProductCD column-Train
plot_col('ProductCD', df=train)


# %%
train.groupby('ProductCD')['isFraud'].mean()

# %%
test.groupby('ProductCD')['isFraud'].mean()

# %% [markdown]
# * C, H, R, S, W values are unknown from data definition.
# 
# Probably their meaning:
# * C (Credit): Credit card transactions
# * H (Debit): Debit card or ATM card transactions
# * R (Charge Card): Charge card transactions
# * S (Cash): Cash transactions
# * W (Wallet): Transactions made with digital wallets or payment applications
# 
# 'W' has the highest frequency, while 'S' has the lowest.
# Most fraud activities realized by C product code. (11%) Least fraud activities with W product ccode. (2%) This could indicate that a C product category is more strongly associated with fraud. The probabilities of fraud for other categories (H, R, S, W) are lower, but the contribution of these categories may still be significant. 
# 

# %%
train['ProductCD'].value_counts(dropna=False)

# %%
test['ProductCD'].value_counts(dropna=False)

# %%
# Perform one-hot encoding for ProductCD column on train
ProductCD_OHE = pd.get_dummies(train['ProductCD'], prefix='ProductCD', drop_first=False, dtype=int)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, ProductCD_OHE], axis=1)


# %%
# Perform one-hot encoding for ProductCD column on train
ProductCD_OHE = pd.get_dummies(test['ProductCD'], prefix='ProductCD', drop_first=False, dtype=int)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, ProductCD_OHE], axis=1)

# %%
# original 'ProductCD' column dropped
train.drop('ProductCD', axis=1, inplace=True)
test.drop('ProductCD', axis=1, inplace=True)

# %%
gc.collect()

# %% [markdown]
# #### Card1-Card6 : payment card information, such as card type, card category, issue bank, country (nominal categorical)

# %%
column_details(regex='^card\d', df=train)

# %%
cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
for i in cards:
    print ("Unique ",i, " = ",train[i].nunique())

# %% [markdown]
# Card4 and Card6 has 4 distinct value, I will observe the distributions of fraud activities in these columns.

# %%
# card4 seems to be credit card company
# to show the most used Brand of electronic cards for fraud transactions in the train dataset
plot_col("card4", df=train)

# %%
train.groupby('card4')['isFraud'].mean()

# %%
test.groupby('card4')['isFraud'].mean()

# %%
train['card4'].value_counts(dropna=False)

# %%
test['card4'].value_counts(dropna=False)

# %%
# Perform one-hot encoding for card4 column on train
card4_OHE = pd.get_dummies(train['card4'], prefix='card4', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, card4_OHE], axis=1)


# %%
# Perform one-hot encoding for card4 column on test
card4_OHE = pd.get_dummies(test['card4'], prefix='card4', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, card4_OHE], axis=1)

# %%
# original 'card4' column dropped
train.drop('card4', axis=1, inplace=True)
test.drop('card4', axis=1, inplace=True)

# %%
# card6 seems to be type of card  
# to show the most used method of transaction with cards for fraud activities in the train dataset
plot_col("card6", df=train)

# %%
train['card6'].value_counts(dropna=False)

# %%
# Because number of instances so low in two cats, i will map them into the category with highest freq. 
# Map 'debit or credit' and 'charge card' values to 'debit'
train['card6'] = train['card6'].map({'debit or credit': 'debit', 'charge card': 'debit'}).fillna(train['card6'])

# %%
train['card6'].value_counts(dropna=False)

# %%
test['card6'].value_counts(dropna=False)

# %%
# Perform one-hot encoding for card6 column on train
card6_OHE = pd.get_dummies(train['card6'], prefix='card6', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, card6_OHE], axis=1)


# %%
# Perform one-hot encoding for card6 column on test
card6_OHE = pd.get_dummies(test['card6'], prefix='card6', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, card6_OHE], axis=1)

# %%
# original 'card6' column dropped
train.drop('card6', axis=1, inplace=True)
test.drop('card6', axis=1, inplace=True)

# %% [markdown]
# Card1: The 'Card 1' column, originally designated as categorical, exhibits behavior akin to continuous data, featuring a substantial '12485' unique values.

# %%
tmp = pd.crosstab(train['card1'], train['isFraud'], normalize='index') * 100
tmp = tmp.reset_index()
tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

plt.figure(figsize=(14, 5))
g = sns.distplot(train[train['isFraud'] == 1]['card1'], label='Fraud')
g = sns.distplot(train[train['isFraud'] == 0]['card1'], label='NoFraud')
g.legend()
g.set_title("Card 1 Values Distribution by Target", fontsize=20)
g.set_xlabel("Card 1 Values", fontsize=18)
g.set_ylabel("Probability", fontsize=18)

plt.show()

# %%
pd.to_numeric(train['card1'], errors='coerce').min()

# %%
pd.to_numeric(train['card1'], errors='coerce').max()

# %%
pd.to_numeric(test['card1'], errors='coerce').min()

# %%
pd.to_numeric(test['card1'], errors='coerce').max()

# %%
# I will replace the type of the column object to numerical because the variable seems to be continious one.
train['card1'] = pd.to_numeric(train['card1'], errors='coerce')
test['card1'] = pd.to_numeric(test['card1'], errors='coerce')

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_2.pkl')

#save 'test' data to a pickle file named 'test_1.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_2.pkl')


# %%
# Read the 'train_1.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_2.pkl')

# Read the 'test_1.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_2.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# Card2, Card3, Card5

# %%
# For card2, card3 and card5 there are many categories with frequency below 200 in train. I change these categories' names to 'Other'

threshold = 200

#Card2
# Identify 'other' categories in the training set
train_card2_counts = train['card2'].value_counts()
train_unique_card2_other = set(train_card2_counts.index[train_card2_counts < threshold])
train['card2'] = train['card2'].apply(lambda x: 'other' if not pd.isna(x) and train_card2_counts[x] < threshold else x)

#Card3
# Identify 'other' categories in the training set
train_card3_counts = train['card3'].value_counts()
train_unique_card3_other = set(train_card3_counts.index[train_card3_counts < threshold])
train['card3'] = train['card3'].apply(lambda x: 'other' if not pd.isna(x) and train_card3_counts[x] < threshold else x)

#Card5
# Identify 'other' categories in the training set
train_card5_counts = train['card5'].value_counts()
train_unique_card5_other = set(train_card5_counts.index[train_card5_counts < threshold])
train['card5'] = train['card5'].apply(lambda x: 'other' if not pd.isna(x) and train_card5_counts[x] < threshold else x)


# %%
# Apply the same transformation to the test set as in the training set, where categories with counts below the threshold are assigned as 'other' in train ( use the categories assigned 'other' in train).
# If the test set has a category that was not present in the training set, set that category to NaN

#Card2
# Apply the 'other' categories identified in the training set to the 'card2' column in the test set
test['card2'] = test['card2'].apply(lambda x: 'other' if not pd.isna(x) and x in train_unique_card2_other else x)

# Identify new categories in the test set and set them to NaN
new_categories_in_test = set(test['card2'].dropna().unique()) - set(train['card2'].dropna().unique())
test['card2'] = test['card2'].apply(lambda x: np.nan if x in new_categories_in_test else x)

#Card3
# Apply the 'other' categories identified in the training set to the 'card3' column in the test set
test['card3'] = test['card3'].apply(lambda x: 'other' if not pd.isna(x) and x in train_unique_card3_other else x)

# Identify new categories in the test set and set them to NaN
new_categories_in_test = set(test['card3'].dropna().unique()) - set(train['card3'].dropna().unique())
test['card3'] = test['card3'].apply(lambda x: np.nan if x in new_categories_in_test else x)

#Card5
# Apply the 'other' categories identified in the training set to the 'card5' column in the test set
test['card5'] = test['card5'].apply(lambda x: 'other' if not pd.isna(x) and x in train_unique_card5_other else x)

# Identify new categories in the test set and set them to NaN
new_categories_in_test = set(test['card5'].dropna().unique()) - set(train['card5'].dropna().unique())
test['card5'] = test['card5'].apply(lambda x: np.nan if x in new_categories_in_test else x)

# %%
# after changing some categories to other, the unique value in each column as follows;
cards = ['card2', 'card3', 'card5']
for i in cards:
    print ("Unique ",i, " = ",train[i].nunique())

# %% [markdown]
# One hot encoding is a common technique used to handle nominal categorical data by converting each category into a binary column. While label encoding can be suitable for ordinal categorical data, it's not the best choice for nominal categorical data due to its inherent ordering. Target encoding provides an effective solution for nominal categorical data.
# In here from card2 to card6 we will use target encoding.

# %%
# Target Encoding
# List of columns to process (Number of unique values below 200) (taking into account the average of the target feature in train set)
columns_to_process = ['card2', 'card3', 'card5']

# Loop through each column
for col in columns_to_process:
    # Calculate target encoding for the current column
    temp_dict = train.groupby([col])['isFraud'].agg(['mean']).to_dict()['mean']
    
    # Create a new column with the target encoding values
    train[f'{col}_target_encoded'] = train[col].replace(temp_dict)
    test[f'{col}_target_encoded'] = test[col].replace(temp_dict)

# %%
# List of original columns to drop
columns_to_drop = ['card2', 'card3', 'card5']

# Drop the original columns
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)


# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_3.pkl')

#save 'test' data to a pickle file named 'test_1.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_3.pkl')


# %%
# Read the 'train_1.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_3.pkl')

# Read the 'test_1.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_3.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# #### addr1 and addr2: Address information related to the transaction (nominal categorical)
# Probably addr1 - subzone / add2 - Country

# %% [markdown]
# addr1

# %%
for df in [train, test]:
  column_details(regex='addr', df=df)

# %%
# First ten most frequent adddress1 values in train
print ("Unique Subzones = ",train['addr1'].nunique())

print('\nFirst Ten Address-1')
print('--------------------')
train.addr1.value_counts().head(9)

# %%
# before changing some categories to other, the unique value in addr1 column as follows;
print ("Unique addr1 = ", train['addr1'].nunique())

# %%
# check addr1 column to define the threshold of other category
result = train.groupby('addr1')['isFraud'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
result.to_clipboard()

# %%
# For addr1 there are many categories with frequency below 1000 in train. I change these categories' names to 'Other'
threshold = 1000

# Identify 'other' categories in the training set
train_addr1_counts = train['addr1'].value_counts()
train_unique_addr1_other = set(train_addr1_counts.index[train_addr1_counts < threshold])
train['addr1'] = train['addr1'].apply(lambda x: 'other' if not pd.isna(x) and train_addr1_counts[x] < threshold else x)

# %%
# Apply the same transformation to the test set as in the training set, where categories with counts below the threshold are assigned as 'other' in train ( use the categories assigned 'other' in train).
# If the test set has a category that was not present in the training set, set that category to NaN

#addr1
# Apply the 'other' categories identified in the training set to the 'addr1' column in the test set
test['addr1'] = test['addr1'].apply(lambda x: 'other' if not pd.isna(x) and x in train_unique_addr1_other else x)

# Identify new categories in the test set and set them to NaN
new_categories_in_test_addr1 = set(test['addr1'].dropna().unique()) - set(train['addr1'].dropna().unique())
test['addr1'] = test['addr1'].apply(lambda x: np.nan if x in new_categories_in_test_addr1 else x)


# %%
# after changing some categories to other, the unique value in addr1 column as follows;
print ("Unique addr1 = ", train['addr1'].nunique())

# %% [markdown]
# addr2

# %%
# check addr2 column to define the threshold of other category
result = train.groupby('addr2')['isFraud'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
result.to_clipboard()

# %%
print ("Unique Countries = ",train['addr2'].nunique())

print('\nFirst Ten Address-2')
print('--------------------')
train.addr2.value_counts().head(9)

# %%
# For addr2 there are many categories with frequency below 50 in train. I change these categories' names to 'Other'
threshold_addr2 = 50

# Identify 'other' categories in the training set
train_addr2_counts = train['addr2'].value_counts()
train_unique_addr2_other = set(train_addr2_counts.index[train_addr2_counts < threshold_addr2])
train['addr2'] = train['addr2'].apply(lambda x: 'other' if not pd.isna(x) and train_addr2_counts[x] < threshold_addr2 else x)

# %%
# Apply the 'other' categories identified in the training set to the 'addr2' column in the test set
test['addr2'] = test['addr2'].apply(lambda x: 'other' if not pd.isna(x) and x in train_unique_addr2_other else x)

# Identify new categories in the test set and set them to NaN
new_categories_in_test_addr2 = set(test['addr2'].dropna().unique()) - set(train['addr2'].dropna().unique())
test['addr2'] = test['addr2'].apply(lambda x: np.nan if x in new_categories_in_test_addr2 else x)


# %%
# after changing some categories to other, the unique value in addr2 column as follows;
print ("Unique addr2 = ", train['addr2'].nunique())

# %% [markdown]
# new addr column

# %%
# By combining addr1 and addr2 , I create a new addr column
for df in [train, test]:
    # Combine 'addr2' and 'addr1' columns as strings, separated by an underscore
    df['addr'] = (df['addr2'].astype(str) + '_' + df['addr1'].astype(str)).replace({'nan_nan': np.nan})

# %%
# check addr column to define the threshold of other category
result = train.groupby('addr')['isFraud'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
result.to_clipboard()

# %%
print ("Unique Adresses = ",train['addr'].nunique())

print('\nFirst Ten Addresses')
print('--------------------')
train.addr.value_counts().head(9)

# %%
# For new addr column there are many categories with frequency below 50 in train. I change these categories' names to 'Other'
threshold_addr = 50

# Identify 'other' categories in the training set
train_addr_counts = train['addr'].value_counts()
train_unique_addr_other = set(train_addr_counts.index[train_addr_counts < threshold_addr])
train['addr'] = train['addr'].apply(lambda x: 'other' if not pd.isna(x) and train_addr_counts[x] < threshold_addr else x)


# %%
# Apply the 'other' categories identified in the training set to the 'addr' column in the test set
test['addr'] = test['addr'].apply(lambda x: 'other' if not pd.isna(x) and x in train_unique_addr_other else x)

# Identify new categories in the test set and set them to NaN
new_categories_in_test_addr = set(test['addr'].dropna().unique()) - set(train['addr'].dropna().unique())
test['addr'] = test['addr'].apply(lambda x: np.nan if x in new_categories_in_test_addr else x)

# %%
# after changing some categories to other, the unique value in addr2 column as follows;
print ("Unique addr = ", train['addr'].nunique())

# %% [markdown]
# Target Encoding for Address columns

# %%
# List of columns to process (Number of unique values below 200) (taking into account the average of the target feature in the train set)
columns_to_process = ['addr1', 'addr2', 'addr']

# Loop through each column
for col in columns_to_process:
    # Calculate target encoding for the current column
    temp_dict = train.groupby([col])['isFraud'].agg(['mean']).to_dict()['mean']
    
    # Create a new column with the target encoding values
    train[f'{col}_target_encoded'] = train[col].replace(temp_dict)
    test[f'{col}_target_encoded'] = test[col].replace(temp_dict)

# %%
# List of original columns to drop
columns_to_drop = ['addr1', 'addr2', 'addr']

# Drop the original columns
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

# %% [markdown]
# **Top 'addr1' Values by Fraud Rate:**
# The 'addr1' values with the highest fraud rates are 305.0, 466.0, 471.0, 483.0, and 501.0.
# For example, transactions with 'addr1' equal to 305.0 have a fraud rate of 66.67%.
# 
# **Top 'addr2' Values by Fraud Rate:**
# The 'addr2' values with the highest fraud rates are 10.0, 82.0, 46.0, 92.0, and 75.0.
# For example, transactions with 'addr2' equal to 10.0 have a fraud rate of 100%.
# 
# **Top Combined 'addr' Values by Fraud Rate:**
# The combined 'addr' values (created by concatenating 'addr2' and 'addr1') with the highest fraud rates are 10.0_296.0, 46.0_296.0, 60.0_296.0, 92.0_296.0, and 82.0_296.0.
# For example, transactions with 'addr' equal to 10.0_296.0 have a fraud rate of 100%.
# 
# **Insights:**
# Certain combinations of 'addr1' and 'addr2' or the combined 'addr' exhibit higher fraud rates. For instance, the combination 10.0_296.0 appears to have a consistent fraud rate of 100% across 'addr1' and 'addr2'.
# These patterns may indicate potential areas of interest for further investigation or feature engineering. High fraud rates in specific 'addr' combinations could be indicative of fraudulent behavior or anomalies in those locations.

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_4.pkl')

#save 'test' data to a pickle file named 'test_1.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_4.pkl')


# %%
# Read the 'train_1.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_4.pkl')

# Read the 'test_1.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_4.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# #### dist1 : The distance (numeric)

# %%
column_details(regex='^dist', df=train)

# %%
plt.figure(figsize=(12,8))
plt.subplot(121)
sns.stripplot(y='dist1', x='isFraud', data=train)
plt.axhline(5000, color='red')
plt.title('Train')

plt.subplot(122)
sns.stripplot(y='dist1', x='isFraud', data=test)
plt.axhline(5000, color='red')
plt.title('Test')

# %%
max_dist1 = train['dist1'][train['dist1'] < 5500].max()
print(f"The maximum dist1 below 5500 is: {max_dist1}")


# %% [markdown]
# There are outliers in dist1 with isFraud == 0. These are above 5500. 
# I find the maximum amount below 5500 ( 5431 ). I will capped the Train with 5432 below code.

# %%
# Identify transactions with isFraud == 0 and dist1 5431
condition = train['dist1'] > 5431

# Cap the dist1 at 5432 for the identified transactions
train.loc[condition, 'dist1'] = 5431

# %%
train['dist1'].max()

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_5.pkl')

#save 'test' data to a pickle file named 'test_1.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_5.pkl')


# %%
# Read the 'train_1.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_5.pkl')

# Read the 'test_1.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_5.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# #### P_emaildomain&R_emaildomain

# %% [markdown]
# P_emaildomain : categoric, 56 uniques
# It's possible to make subgroup feature from it or general group
# 

# %% [markdown]
# R_emaildomain : categoric, 59 uniques
# It's possible to make subgroup feature from it or general group

# %%
# R_emaildomain
column_details(regex='R_emaildomain', df=train)

# %%
# P_emaildomain
column_details(regex='P_emaildomain', df=train)

# %% [markdown]
# Mapping Emails

# %%
#I produced 2 new columns with mail server and domain for 
# The emails dictionary maps email domain suffixes to their corresponding email providers (e.g., 'gmail.com' to 'google', 'att.net' to 'att', etc.).
emails = {'gmail': 'google', 'att.net': 'other', 'twc.com': 'spectrum', 
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'other',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'earthlink', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
          'protonmail.com': 'protonmail', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'other', 
          'prodigy.net.mx': 'other', 'frontier.com': 'yahoo', 'anonymous.com': 'anonymous', 
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'other', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'mail', 
          'bellsouth.net': 'other', 'embarqmail.com': 'other', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

# The us_emails list contains a few strings representing common email domain suffixes associated with the United States ('gmail', 'net', 'edu').
us_emails = ['gmail', 'net', 'edu']

# Iterating over two columns, 'P_emaildomain' and 'R_emaildomain', in two DataFrames, train and test.
for c in ['P_emaildomain', 'R_emaildomain']:
    # Mapping email domains to their corresponding providers and creating new columns with the suffix '_bin' appended.
    train[c + '_bin'] = train[c].map(emails)
    test[c + '_bin'] = test[c].map(emails)
    
    # Creating new columns with the suffix '_suffix' appended, containing the last part of the email domain after the last period.
    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
    # Mapping these suffixes to 'us' if they are in the us_emails list, indicating that the email is associated with the United States.
    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

# %%
# Convert 'nan' strings to actual np.nan values in the specified suffix columns for both train and test DataFrames
suffix_columns = ['P_emaildomain_bin', 'R_emaildomain_bin', 'P_emaildomain_suffix', 'R_emaildomain_suffix']

for column in suffix_columns:
    train[column].replace('nan', np.nan, inplace=True)
    test[column].replace('nan', np.nan, inplace=True)

# %% [markdown]
# R_emaildomain_bin train-test sync

# %%
# Check the frequencies
value_counts = train['R_emaildomain_bin'].value_counts()

# Identify categories with less than 1000 occurrences
threshold = 1000
rare_categories = value_counts[value_counts < threshold].index

# Assign 'other' category to rows with rare categories
train.loc[train['R_emaildomain_bin'].isin(rare_categories), 'R_emaildomain_bin'] = 'other'


# %%
# Assign 'other' category to rows with rare categories in the test data frame
test.loc[test['R_emaildomain_bin'].isin(rare_categories), 'R_emaildomain_bin'] = 'other'

# %% [markdown]
# P_emaildomain_bin train-test sync

# %%
# Check the frequencies
value_counts = train['P_emaildomain_bin'].value_counts()

# Identify categories with less than 5000 occurrences
threshold = 5000
rare_categories = value_counts[value_counts < threshold].index

# Assign 'other' category to rows with rare categories
train.loc[train['P_emaildomain_bin'].isin(rare_categories), 'P_emaildomain_bin'] = 'other'

# %%
# Assign 'other' category to rows with rare categories in the test data frame
test.loc[test['P_emaildomain_bin'].isin(rare_categories), 'P_emaildomain_bin'] = 'other'

# %% [markdown]
# R_emaildomain_suffix train-test sync

# %%
# Check the frequencies
value_counts = train['R_emaildomain_suffix'].value_counts()

# Identify categories with less than 500 occurrences
threshold = 500
rare_categories = value_counts[value_counts < threshold].index

# Assign 'other' category to rows with rare categories
train.loc[train['R_emaildomain_suffix'].isin(rare_categories), 'R_emaildomain_suffix'] = 'other'

# %%
# Assign 'other' category to rows with rare categories in the test data frame
test.loc[test['R_emaildomain_suffix'].isin(rare_categories), 'R_emaildomain_suffix'] = 'other'

# %% [markdown]
# P_emaildomain_suffix train-test sync

# %%
# Check the frequencies
value_counts = train['P_emaildomain_suffix'].value_counts()

# Identify categories with less than 500 occurrences
threshold = 500
rare_categories = value_counts[value_counts < threshold].index

# Assign 'other' category to rows with rare categories
train.loc[train['P_emaildomain_suffix'].isin(rare_categories), 'P_emaildomain_suffix'] = 'other'

# %%
# Assign 'other' category to rows with rare categories in the test data frame
test.loc[test['P_emaildomain_suffix'].isin(rare_categories), 'P_emaildomain_suffix'] = 'other'

# %% [markdown]
# Control of cats between train and test

# %%
# Control of whether any incompatible categories between train and test
# Columns to be examined
columns_to_check = ['P_emaildomain_bin', 'R_emaildomain_bin', 'P_emaildomain_suffix', 'R_emaildomain_suffix']

# Create a dictionary to store columns with different categories
different_categories = {}

# Check each column
for column in columns_to_check:
    train_categories = set(train[column].unique())
    test_categories = set(test[column].unique())
    
    # If they have different categories, add to the dictionary
    if train_categories != test_categories:
        different_categories[column] = {
            'train': train_categories,
            'test': test_categories
        }

# Print columns with different categories
if different_categories:
    print("Columns with different categories:")
    for column, categories in different_categories.items():
        print(f"{column}:")
        print(f"  Train Categories: {categories['train']}")
        print(f"  Test Categories:  {categories['test']}\n")
else:
    print("There are no columns with different categories between the train and test datasets.")


# %%
for col in ['R_emaildomain_bin', 'P_emaildomain_bin']:
  plot_col(col, df=train)

# %%
for col in ['R_emaildomain_suffix', 'P_emaildomain_suffix']:
  plot_col(col, df=train)

# %%
# es has 10% fraud rate
fraud_rates = train.groupby('R_emaildomain_suffix')['isFraud'].mean().reset_index()
fraud_rates.rename(columns={'isFraud': 'FraudRate'}, inplace=True)
fraud_rates.sort_values(by='FraudRate', ascending=False).head()

# %% [markdown]
# Two highest fraud activity domains:
# * es (Spain) Category: The category associated with Spain (es) has a higher fraud rate compared to other categories (%10). This suggests that transactions originating from Spain may require closer scrutiny in terms of fraud detection.
# 
# * com (United States) Category: The category associated with the United States (com) has a significantly higher fraud rate compared to other categories (%8.32). This may imply the need for careful examination of transactions coming from this geographical region.

# %%
# Protonmail has 95% fraud rate
fraud_rates = train.groupby('R_emaildomain_bin')['isFraud'].mean().reset_index()
fraud_rates.rename(columns={'isFraud': 'FraudRate'}, inplace=True)
fraud_rates.sort_values(by='FraudRate', ascending=False).head()

# %%
# Protonmail has 46% fraud rate
fraud_rates = train.groupby('P_emaildomain_bin')['isFraud'].mean().reset_index()
fraud_rates.rename(columns={'isFraud': 'FraudRate'}, inplace=True)
fraud_rates.sort_values(by='FraudRate', ascending=False).head()

# %% [markdown]
# Two highest fraud activity domains:
# 
# * ProtonMail Category: The ProtonMail category has a significantly higher fraud rate compared to other categories (%95). This indicates that transactions originating from ProtonMail may pose a higher risk of fraud. Careful scrutiny of transactions associated with such accounts could be crucial.
# 
# * Mail Category: This general 'mail' category exhibits a higher fraud rate compared to other categories (%36). This suggests that transactions from general email providers should be examined with caution.

# %%
# Because protonmail has the highest fraud rate I will create a boolen column for it
#train['P_isproton']=(train['P_emaildomain']=='protonmail.com')
#train['R_isproton']=(train['R_emaildomain']=='protonmail.com')
#test['P_isproton']=(test['P_emaildomain']=='protonmail.com')
#test['R_isproton']=(test['R_emaildomain']=='protonmail.com')

# %% [markdown]
# one-hot encoding for R_emaildomain_bin

# %%
# Perform one-hot encoding for R_emaildomain_bin column on train
R_emaildomain_bin_OHE = pd.get_dummies(train['R_emaildomain_bin'], prefix='R_emaildomain_bin', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, R_emaildomain_bin_OHE], axis=1)


# %%
# Perform one-hot encoding for R_emaildomain_bin column on test
R_emaildomain_bin_OHE = pd.get_dummies(test['R_emaildomain_bin'], prefix='R_emaildomain_bin', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, R_emaildomain_bin_OHE], axis=1)

# %%
# original 'R_emaildomain_bin' column dropped
train.drop('R_emaildomain_bin', axis=1, inplace=True)
test.drop('R_emaildomain_bin', axis=1, inplace=True)

# %%
# I dont want to use other category because it is not homogenous in terms of fraud_rate
train.drop('R_emaildomain_bin_other', axis=1, inplace=True)
test.drop('R_emaildomain_bin_other', axis=1, inplace=True)

# %% [markdown]
# one-hot encoding for R_emaildomain_suffix

# %%
# Perform one-hot encoding for R_emaildomain_suffix column on train
R_emaildomain_suffix_OHE = pd.get_dummies(train['R_emaildomain_suffix'], prefix='R_emaildomain_suffix', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, R_emaildomain_suffix_OHE], axis=1)


# %%
# Perform one-hot encoding for R_emaildomain_suffix column on test
R_emaildomain_suffix_OHE = pd.get_dummies(test['R_emaildomain_suffix'], prefix='R_emaildomain_suffix', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, R_emaildomain_suffix_OHE], axis=1)

# %%
# original 'R_emaildomain_suffix' column dropped
train.drop('R_emaildomain_suffix', axis=1, inplace=True)
test.drop('R_emaildomain_suffix', axis=1, inplace=True)

# %%
# I dont want to use other category because it is not homogenous in terms of fraud_rate
train.drop('R_emaildomain_suffix_other', axis=1, inplace=True)
test.drop('R_emaildomain_suffix_other', axis=1, inplace=True)

# %% [markdown]
# one-hot encoding for P_emaildomain_bin

# %%
# Perform one-hot encoding for P_emaildomain_bin column on train
P_emaildomain_bin_OHE = pd.get_dummies(train['P_emaildomain_bin'], prefix='P_emaildomain_bin', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, P_emaildomain_bin_OHE], axis=1)


# %%
# Perform one-hot encoding for P_emaildomain_bin column on test
P_emaildomain_bin_OHE = pd.get_dummies(test['P_emaildomain_bin'], prefix='P_emaildomain_bin', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, P_emaildomain_bin_OHE], axis=1)

# %%
# original 'R_emaildomain_bin' column dropped
train.drop('P_emaildomain_bin', axis=1, inplace=True)
test.drop('P_emaildomain_bin', axis=1, inplace=True)

# %%
# I dont want to use other category because it is not homogenous in terms of fraud_rate
train.drop('P_emaildomain_bin_other', axis=1, inplace=True)
test.drop('P_emaildomain_bin_other', axis=1, inplace=True)

# %% [markdown]
# one-hot encoding for P_emaildomain_suffix

# %%
# Perform one-hot encoding for P_emaildomain_suffix column on train
P_emaildomain_suffix_OHE = pd.get_dummies(train['P_emaildomain_suffix'], prefix='P_emaildomain_suffix', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, P_emaildomain_suffix_OHE], axis=1)

# %%
# Perform one-hot encoding for P_emaildomain_suffix column on test
P_emaildomain_suffix_OHE = pd.get_dummies(test['P_emaildomain_suffix'], prefix='P_emaildomain_suffix', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, P_emaildomain_suffix_OHE], axis=1)

# %%
# original 'P_emaildomain_suffix' column dropped
train.drop('P_emaildomain_suffix', axis=1, inplace=True)
test.drop('P_emaildomain_suffix', axis=1, inplace=True)

# %%
# I dont want to use other category because it is not homogenous in terms of fraud_rate
train.drop('P_emaildomain_suffix_other', axis=1, inplace=True)
test.drop('P_emaildomain_suffix_other', axis=1, inplace=True)

# %%
# dropping original columns
train = train.drop('P_emaildomain', axis=1)
test = test.drop('P_emaildomain', axis=1)

# %%
# dropping original columns
train = train.drop('R_emaildomain', axis=1)
test = test.drop('R_emaildomain', axis=1)

# %%
train

# %%
# I want to remove collinear features in 'email' columns with a threshold of 0.75
mail_cols = ['P_emaildomain_suffix', 'R_emaildomain_suffix', 'P_emaildomain_bin', 'R_emaildomain_bin']
filtered_columns = [col for col in train.columns if any(pattern in col for pattern in mail_cols)]
drop_columns = identify_collinear_categorical_features(train, filtered_columns, threshold=0.75)

# Display the columns to drop
print("Columns to drop:", drop_columns)

# %%
selected_columns = filtered_columns

# Create an empty DataFrame to store the results
cramers_v_matrix = pd.DataFrame(index=selected_columns, columns=selected_columns, dtype=float)

# Fill in the Cramers V values for each pair of columns
for col1 in selected_columns:
    for col2 in selected_columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(train[col1], train[col2])

# Heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(cramers_v_matrix, cmap='RdBu_r', annot=True, fmt=".2f", linewidths=.5)
plt.title('Cramers V Matrix Heatmap')
plt.show()

# %%
# dropping highly correlated columns
train = train.drop(drop_columns, axis=1)
test = test.drop(drop_columns, axis=1)

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_6.pkl')

#save 'test' data to a pickle file named 'test_1.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_6.pkl')


# %%
# Read the 'train_1.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_6.pkl')

# Read the 'test_1.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_6.pkl')

# %% [markdown]
# #### C1 .... C14 : Counting variables such as how many addresses are found to be associated with the payment card (numeric)

# %%
column_details(regex='^C\d', df=train)

# %%
# Finding highly correlated columns to drop
columns=[col for col in train.columns if re.search('^C\d.*', col)]
corr_treshold = 0.75
drop_col = remove_collinear_features(train[columns],corr_treshold)
drop_col

# %%
# I m removing C13 instead of C1, beucase C13 has many outliers. 
drop_col = {'C13','C10', 'C11', 'C12', 'C14', 'C2', 'C4', 'C6', 'C7', 'C8', 'C9'}

# %%
# Select columns starting with 'C'
columns = [col for col in train.columns if re.search('^C\d.*', col)]

# Create a correlation heatmap using the make_corr function
make_corr(columns, train)


# %%
# Only C1, C5 remained.
train = train.drop(drop_col, axis=1)
test = test.drop(drop_col, axis=1)

# %% [markdown]
# #### D1...D15 : Timedelta variables, such as days between previous transaction (numeric)

# %%
for df in [train, test]:
  column_details(regex='^D\d.*', df=df)

# %%
columns=[col for col in train.columns if re.search('^D\d.*', col)]

corr_treshold = 0.75
drop_col = remove_collinear_features(train[columns],corr_treshold)
drop_col

# %%
[col for col in train.columns if re.search('^D\d.*', col)]

# %%
# Create a correlation heatmap using the make_corr function
make_corr(columns, train)

# %%
# The correlated columns having the most missing values are dropped. So, I replaced some columns in the dropping column list below.
drop_col={'D11', 'D12', 'D2', 'D4'}
for df in [train, test]:
  df = df.drop(drop_col, axis=1)

# %% [markdown]
# #### M1 ... M9 : Match variables, used to verify information such as names on the card and address. (nominal categoric)

# %%
column_details(regex='^M\d*', df=train)

# %%
for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
  plot_col(col, df=train)

# %% [markdown]
# In the case of the 'M4' and 'M5 columnS, missing values do not have the highest percentage of fraud.

# %%
# converting 'nan' to np.nan
regex_pattern = '^M\d*'
columns_to_replace = [col for col in df.columns if col.startswith(tuple(regex_pattern))]
df[columns_to_replace] = df[columns_to_replace].replace('nan', np.nan)

# %%
# Replace 'F' values with NaN in the 'M1' column (because in train there is only 20 F rows, in test only 5 rows)
train['M1'].replace('F', np.nan, inplace=True)
test['M1'].replace('F', np.nan, inplace=True)

# %%
# One Hot Encoding for M columns-train
# Select the column names
columns_to_encode = train.columns[train.columns.str.match(r'^M[1-9]$')]

# Perform one-hot encoding for each column
for column in columns_to_encode:
    one_hot_encoded = pd.get_dummies(train[column], prefix=column, drop_first=False, dtype=int, dummy_na=True)
    train = pd.concat([train, one_hot_encoded], axis=1)

# %%
# One Hot Encoding for M columns-test
# Select the column names
columns_to_encode = test.columns[test.columns.str.match(r'^M[1-9]$')]

# Perform one-hot encoding for each column
for column in columns_to_encode:
    one_hot_encoded = pd.get_dummies(test[column], prefix=column, drop_first=False, dtype=int, dummy_na=True)
    test = pd.concat([test, one_hot_encoded], axis=1)

# %%
# Drop the original columns that were one-hot encoded
train.drop(columns=columns_to_encode, inplace=True)
test.drop(columns=columns_to_encode, inplace=True)

# %%
# I want to remove collinear features in 'M' columns with a threshold of 0.75
m_cols = [f'M{i}' for i in range(1, 10)]
filtered_columns = [col for col in train.columns if any(pattern in col for pattern in m_cols)]
drop_columns = identify_collinear_categorical_features(train, filtered_columns, threshold=0.75)

# Display the columns to drop
print("Columns to drop:", drop_columns)

# %%
m_columns = train.loc[:, filtered_columns]

# Create an empty DataFrame to store the results
cramers_v_matrix = pd.DataFrame(index=m_columns.columns, columns=m_columns.columns, dtype=float)

# Fill in the Cramers V values for each pair of columns
for col1 in m_columns.columns:
    for col2 in m_columns.columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(m_columns[col1], m_columns[col2])

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cramers_v_matrix, cmap='RdBu_r', annot=True, fmt=".2f", linewidths=.5)
plt.title('Cramers V Matrix Heatmap')
plt.show()

# %%
#dropping highly correlated features 
train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_7.pkl')

#save 'test' data to a pickle file named 'test_1.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_7.pkl')

# %%
# Read the 'train_1.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_7.pkl')

# Read the 'test_1.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_7.pkl')

# %% [markdown]
# #### V1-V339 : Vesta-engineered features that encompass ranking, counting, and various entity relationships.(numeric)

# %% [markdown]
# We identified redundancy and correlation among the 'V' columns, and dropped correlated columns with a correlation coefficient (r) greater than 0.75. This process resulted in retaining only 69 independent 'V' columns.

# %%
column_details(regex='V\d*', df=train)

# %%
# removing high correlated variables (222 eliminated)
corr_treshold = 0.75
drop_col = remove_collinear_features(train[columns],corr_treshold)
len(drop_col)

# %%
# dropping redundant Vs
train = train.drop(drop_col, axis=1)
test = test.drop(drop_col, axis=1)

# %%
# remaining Variables' length (64 vars)
columns=[col for col in train.columns if re.search('^V\d*', col)]
len(columns)

# %%
plt.figure(figsize=(10,10))
sns.heatmap(train[columns+['isFraud']].sample(frac=0.2).corr(),annot=False, cmap="RdBu_r")

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_2.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_8.pkl')

#save 'test' data to a pickle file named 'test_2.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_8.pkl')


# %%
# Read the 'train_2.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_8.pkl')

# Read the 'test_2.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_8.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# #### id_1 ... id_11 (numeric)

# %%
column_details(regex='id_(1|2|3|4|5|6|7|8|9|10|11)$', df=train)

# %%
# removing high correlated variables 
corr_treshold = 0.75
drop_col = remove_collinear_features(train[columns],corr_treshold)
len(drop_col)

# %%
'''
# dropping redundant Vs
train = train.drop(drop_col, axis=1)
test = test.drop(drop_col, axis=1)
'''

# %% [markdown]
# There is no correlation between these two variables.

# %% [markdown]
# #### id_12...id_38 (nominal categoric)

# %%
column_details(regex='id_(12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38)', df=train)

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_2.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_9.pkl')

#save 'test' data to a pickle file named 'test_2.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_9.pkl')


# %%
# Read the 'train_2.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_9.pkl')

# Read the 'test_2.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_9.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# id_30

# %%
# Some browser codes with '_' others have '.' .  I replace '_' with '.' in the 'id_30' column 
train['id_30'] = train['id_30'].str.replace('_', '.')
test['id_30'] = test['id_30'].str.replace('_', '.')

# %%
# creating OS_id_30 column
train['OS_id_30'] = train['id_30'].str.split(' ', expand=True)[0]
test['OS_id_30'] = test['id_30'].str.split(' ', expand=True)[0]

# %%
# Replace values in 'OS_id_30' column with np.nan if they are not in the allowed list
allowed_values = ['Windows', 'iOS', 'Mac', 'Android', 'Linux']
train['OS_id_30'] = np.where(train['OS_id_30'].isin(allowed_values), train['OS_id_30'], np.nan)
test['OS_id_30'] = np.where(test['OS_id_30'].isin(allowed_values), test['OS_id_30'], np.nan)

# %%
test['OS_id_30'].value_counts(dropna=False)

# %%
# Perform one-hot encoding for OS_id_30 column on train
OS_id_30_OHE = pd.get_dummies(train['OS_id_30'], prefix='OS_id_30', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, OS_id_30_OHE], axis=1)

# %%
# Perform one-hot encoding for OS_id_30 column on test
OS_id_30_OHE = pd.get_dummies(test['OS_id_30'], prefix='OS_id_30', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, OS_id_30_OHE], axis=1)

# %%
# original 'OS_id_30' column dropped
train.drop('OS_id_30', axis=1, inplace=True)
test.drop('OS_id_30', axis=1, inplace=True)

# %% [markdown]
# Some of the browsers dont have the version. I will assign these to the browsers' common version ( mode )

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_2.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_10.pkl')

#save 'test' data to a pickle file named 'test_2.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_10.pkl')


# %%
# Read the 'train_2.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_10.pkl')

# Read the 'test_2.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_10.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# id_31

# %%
# create a new column that represents lastest browser
a = np.zeros(train.shape[0])
train["lastest_browser"] = a
a = np.zeros(test.shape[0])
test["lastest_browser"] = a
def setbrowser(df):
    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
    return df
train=setbrowser(train)
test=setbrowser(test)

# %%
# Capitalize the first letter of categories in the id_31 column
train['id_31'] = train['id_31'].str.capitalize()
test['id_31'] = test['id_31'].str.capitalize()

# %%
# Creating Browser_id_31 column by mapping id_31 - train

# Create a boolean array for missing values
train['id_31'].fillna('', inplace=True)
missing_values = train['id_31'].isnull()

# Update the 'Browser_id_31' column based on conditions
train.loc[missing_values, 'Browser_id_31'] = np.nan
train.loc[train['id_31'].str.contains('Chrome', case=False) & ~train['id_31'].str.contains('Android|iOS', case=False), 'Browser_id_31'] = 'Chrome'
train.loc[train['id_31'].str.contains('Chrome', case=False) & train['id_31'].str.contains('Android', case=False), 'Browser_id_31'] = 'Chrome Android'
train.loc[train['id_31'].str.contains('Chrome', case=False) & train['id_31'].str.contains('iOS', case=False), 'Browser_id_31'] = 'Chrome iOS'
train.loc[train['id_31'].str.contains('Safari', case=False) & ~train['id_31'].str.contains('Mobile', case=False), 'Browser_id_31'] = 'Safari'
train.loc[train['id_31'].str.contains('Safari', case=False) & train['id_31'].str.contains('Mobile', case=False), 'Browser_id_31'] = 'Mobile Safari'
train.loc[train['id_31'].str.contains('Firefox', case=False), 'Browser_id_31'] = 'Firefox'
train.loc[train['id_31'].str.startswith('Ie') & ~train['id_31'].str.contains('tablet', case=False), 'Browser_id_31'] = 'Internet Explorer'
train.loc[train['id_31'].str.contains('Edge', case=False), 'Browser_id_31'] = 'Internet Explorer'
train.loc[train['id_31'].str.startswith('Ie') & train['id_31'].str.contains('tablet', case=False), 'Browser_id_31'] = 'Internet Explorer Mobile'
train.loc[train['id_31'].str.startswith('Samsu'), 'Browser_id_31'] = 'Samsung Browser'

# %%
# Creating Browser_id_31 column by mapping id_31 -test

# Create a boolean array for missing values
test['id_31'].fillna('', inplace=True)
missing_values = test['id_31'].isnull()

# Update the 'Browser_id_31' column based on conditions
test.loc[missing_values, 'Browser_id_31'] = np.nan
test.loc[test['id_31'].str.contains('Chrome', case=False) & ~test['id_31'].str.contains('Android|iOS', case=False), 'Browser_id_31'] = 'Chrome'
test.loc[test['id_31'].str.contains('Chrome', case=False) & test['id_31'].str.contains('Android', case=False), 'Browser_id_31'] = 'Chrome Android'
test.loc[test['id_31'].str.contains('Chrome', case=False) & test['id_31'].str.contains('iOS', case=False), 'Browser_id_31'] = 'Chrome iOS'
test.loc[test['id_31'].str.contains('Safari', case=False) & ~test['id_31'].str.contains('Mobile', case=False), 'Browser_id_31'] = 'Safari'
test.loc[test['id_31'].str.contains('Safari', case=False) & test['id_31'].str.contains('Mobile', case=False), 'Browser_id_31'] = 'Mobile Safari'
test.loc[test['id_31'].str.contains('Firefox', case=False), 'Browser_id_31'] = 'Firefox'
test.loc[test['id_31'].str.startswith('Ie') & ~test['id_31'].str.contains('tablet', case=False), 'Browser_id_31'] = 'Internet Explorer'
test.loc[test['id_31'].str.contains('Edge', case=False), 'Browser_id_31'] = 'Internet Explorer'
test.loc[test['id_31'].str.startswith('Ie') & test['id_31'].str.contains('tablet', case=False), 'Browser_id_31'] = 'Internet Explorer Mobile'
test.loc[test['id_31'].str.startswith('Samsu'), 'Browser_id_31'] = 'Samsung Browser'

# %% [markdown]
# One Hot Encoding for Browser_id_31

# %%
# Perform one-hot encoding for Browser_id_31 column on train
Browser_id_31_OHE = pd.get_dummies(train['Browser_id_31'], prefix='Browser_id_31', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, Browser_id_31_OHE], axis=1)

# %%
# Perform one-hot encoding for Browser_id_31 column on test
Browser_id_31_OHE = pd.get_dummies(test['Browser_id_31'], prefix='Browser_id_31', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, Browser_id_31_OHE], axis=1)

# %%
# original 'Browser_id_31' column dropped
train.drop('Browser_id_31', axis=1, inplace=True)
test.drop('Browser_id_31', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For Lastest Browser

# %%
# Perform one-hot encoding for lastest_browser column on train
lastest_browser_OHE = pd.get_dummies(train['lastest_browser'], prefix='lastest_browser', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, lastest_browser_OHE], axis=1)

# %%
# Perform one-hot encoding for lastest_browser column on test
lastest_browser_OHE = pd.get_dummies(test['lastest_browser'], prefix='lastest_browser', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, lastest_browser_OHE], axis=1)

# %%
# original 'lastest_browser' column dropped
train.drop('lastest_browser', axis=1, inplace=True)
test.drop('lastest_browser', axis=1, inplace=True)

# %%
train

# %%
# original 'id_30' column dropped
train.drop('id_30', axis=1, inplace=True)
test.drop('id_30', axis=1, inplace=True)

# %%
# original 'id_31' column dropped
train.drop('id_31', axis=1, inplace=True)
test.drop('id_31', axis=1, inplace=True)

# %% [markdown]
# Remaining id columns

# %%
# some id columns will drop because of inconsistency based on time
columns_to_drop = ['id_13', 'id_14', 'id_17', 'id_19', 'id_20', 'id_33']

# Drop the specified columns from the dataframe
train.drop(columns=columns_to_drop, inplace=True)
test.drop(columns=columns_to_drop, inplace=True)

# %% [markdown]
# id_12

# %% [markdown]
# One Hot Encoding For id_12

# %%
# Perform one-hot encoding for id_12 column on train
id_12_OHE = pd.get_dummies(train['id_12'], prefix='id_12', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_12_OHE], axis=1)

# %%
# Perform one-hot encoding for id_12 column on test
id_12_OHE = pd.get_dummies(test['id_12'], prefix='id_12', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_12_OHE], axis=1)

# %%
# original 'id_12' column dropped
train.drop('id_12', axis=1, inplace=True)
test.drop('id_12', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_15

# %%
# Perform one-hot encoding for id_15 column on train
id_15_OHE = pd.get_dummies(train['id_15'], prefix='id_15', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_15_OHE], axis=1)

# %%
# Perform one-hot encoding for id_15 column on test
id_15_OHE = pd.get_dummies(test['id_15'], prefix='id_15', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_15_OHE], axis=1)

# %%
# original 'id_15' column dropped
train.drop('id_15', axis=1, inplace=True)
test.drop('id_15', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_16

# %%
# Perform one-hot encoding for id_16 column on train
id_16_OHE = pd.get_dummies(train['id_16'], prefix='id_16', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_16_OHE], axis=1)

# %%
# Perform one-hot encoding for id_16 column on test
id_16_OHE = pd.get_dummies(test['id_16'], prefix='id_16', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_16_OHE], axis=1)

# %%
# original 'id_16' column dropped
train.drop('id_16', axis=1, inplace=True)
test.drop('id_16', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_28

# %%
# Perform one-hot encoding for id_28 column on train
id_28_OHE = pd.get_dummies(train['id_28'], prefix='id_28', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_28_OHE], axis=1)

# %%
# Perform one-hot encoding for id_28 column on test
id_28_OHE = pd.get_dummies(test['id_28'], prefix='id_28', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_28_OHE], axis=1)

# %%
# original 'id_28' column dropped
train.drop('id_28', axis=1, inplace=True)
test.drop('id_28', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_29

# %%
# Perform one-hot encoding for id_29 column on train
id_29_OHE = pd.get_dummies(train['id_29'], prefix='id_29', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_29_OHE], axis=1)

# %%
# Perform one-hot encoding for id_29 column on test
id_29_OHE = pd.get_dummies(test['id_29'], prefix='id_29', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_29_OHE], axis=1)

# %%
# original 'id_29' column dropped
train.drop('id_29', axis=1, inplace=True)
test.drop('id_29', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_32

# %%
# Check the frequencies in the 'id_32' column below 1000 , then drop low freq cats
value_counts = train['id_32'].value_counts()

# Set values with frequencies below 1000 to NaN
low_frequency_values = value_counts[value_counts < 1000].index
train.loc[train['id_32'].isin(low_frequency_values), 'id_32'] = np.nan
test.loc[test['id_32'].isin(low_frequency_values), 'id_32'] = np.nan

# %%
# Perform one-hot encoding for id_32 column on train
id_32_OHE = pd.get_dummies(train['id_32'], prefix='id_32', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_32_OHE], axis=1)

# %%
# Perform one-hot encoding for id_32 column on test
id_32_OHE = pd.get_dummies(test['id_32'], prefix='id_32', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_32_OHE], axis=1)

# %%
# original 'id_32' column dropped
train.drop('id_32', axis=1, inplace=True)
test.drop('id_32', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_34

# %%
# Check the frequencies in the 'id_34' column below 1000 , then drop low freq cats
value_counts = train['id_34'].value_counts()

# Set values with frequencies below 1000 to NaN
low_frequency_values = value_counts[value_counts < 1000].index
train.loc[train['id_34'].isin(low_frequency_values), 'id_34'] = np.nan
test.loc[test['id_34'].isin(low_frequency_values), 'id_34'] = np.nan

# %%
# Perform one-hot encoding for id_34 column on train
id_34_OHE = pd.get_dummies(train['id_34'], prefix='id_34', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_34_OHE], axis=1)

# %%
# Perform one-hot encoding for id_34 column on test
id_34_OHE = pd.get_dummies(test['id_34'], prefix='id_34', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_34_OHE], axis=1)

# %%
# original 'id_34' column dropped
train.drop('id_34', axis=1, inplace=True)
test.drop('id_34', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_35

# %%
# Perform one-hot encoding for id_35 column on train
id_35_OHE = pd.get_dummies(train['id_35'], prefix='id_35', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_35_OHE], axis=1)

# %%
# Perform one-hot encoding for id_35 column on test
id_35_OHE = pd.get_dummies(test['id_35'], prefix='id_35', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_35_OHE], axis=1)

# %%
# original 'id_35' column dropped
train.drop('id_35', axis=1, inplace=True)
test.drop('id_35', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_36

# %%
# Perform one-hot encoding for id_36 column on train
id_36_OHE = pd.get_dummies(train['id_36'], prefix='id_36', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_36_OHE], axis=1)

# %%
# Perform one-hot encoding for id_36 column on test
id_36_OHE = pd.get_dummies(test['id_36'], prefix='id_36', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_36_OHE], axis=1)

# %%
# original 'id_36' column dropped
train.drop('id_36', axis=1, inplace=True)
test.drop('id_36', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_37

# %%
# Perform one-hot encoding for id_37 column on train
id_37_OHE = pd.get_dummies(train['id_37'], prefix='id_37', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_37_OHE], axis=1)

# %%
# Perform one-hot encoding for id_37 column on test
id_37_OHE = pd.get_dummies(test['id_37'], prefix='id_37', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_37_OHE], axis=1)

# %%
# original 'id_37' column dropped
train.drop('id_37', axis=1, inplace=True)
test.drop('id_37', axis=1, inplace=True)

# %% [markdown]
# One Hot Encoding For id_38

# %%
# Perform one-hot encoding for id_38 column on train
id_38_OHE = pd.get_dummies(train['id_38'], prefix='id_38', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, id_38_OHE], axis=1)

# %%
# Perform one-hot encoding for id_38 column on test
id_38_OHE = pd.get_dummies(test['id_38'], prefix='id_38', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, id_38_OHE], axis=1)

# %%
# original 'id_38' column dropped
train.drop('id_38', axis=1, inplace=True)
test.drop('id_38', axis=1, inplace=True)

# %%
matching_columns = [col for col in train.columns if (col.startswith('id_') and 12 <= int(col.split('_')[1]) <= 38)
                                                  or col.startswith(('OS_id_30', 'lastest_browser', 'Browser_id_31'))]

print(matching_columns)

# %%
id_columns = train.loc[:, matching_columns]

# Create an empty DataFrame to store the results
cramers_v_matrix = pd.DataFrame(index=id_columns.columns, columns=id_columns.columns, dtype=float)

# Fill in the Cramers V values for each pair of columns
for col1 in id_columns.columns:
    for col2 in id_columns.columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(id_columns[col1], id_columns[col2])

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cramers_v_matrix, cmap='RdBu_r', annot=True, fmt=".2f", linewidths=.5)
plt.title('Cramers V Matrix Heatmap')
plt.show()

# %%
# i want to remove collinear features in 'id_' columns with a threshold of 0.75
id_columns = matching_columns
drop_columns = identify_collinear_categorical_features(train, id_columns, threshold=0.75)

# Display the columns to drop
print("Columns to drop:", drop_columns)


# %%
# Remove the collinear features
train = remove_collinear_categorical_features(train, drop_columns)
test = remove_collinear_categorical_features(test, drop_columns)

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_2.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_11.pkl')

#save 'test' data to a pickle file named 'test_2.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_11.pkl')


# %%
# Read the 'train_2.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_11.pkl')

# Read the 'test_2.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_11.pkl')

# %%
test['isFraud'].value_counts(dropna=False)

# %% [markdown]
# #### DeviceType (nominal categoric)

# %%
for df in [train, test]:
  column_details(regex='DeviceType', df=df)

# %%
plot_col('DeviceType', df=train)

# %% [markdown]
# One Hot Encoding For DeviceType

# %%
# Perform one-hot encoding for DeviceType column on train
DeviceType_OHE = pd.get_dummies(train['DeviceType'], prefix='DeviceType', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, DeviceType_OHE], axis=1)

# %%
# Perform one-hot encoding for DeviceType column on test
DeviceType_OHE = pd.get_dummies(test['DeviceType'], prefix='DeviceType', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, DeviceType_OHE], axis=1)

# %%
# original 'DeviceType' column dropped
train.drop('DeviceType', axis=1, inplace=True)
test.drop('DeviceType', axis=1, inplace=True)

# %%
test['isFraud'].value_counts(dropna=False)

# %%
[col for col in df.columns if col.startswith('DeviceType')]

# %%
# i want to remove collinear features in 'DeviceType' columns with a threshold of 0.75
id_columns = [col for col in train.columns if col.startswith('DeviceType')]
drop_columns = identify_collinear_categorical_features(train, id_columns, threshold=0.75)

# Display the columns to drop
print("Columns to drop:", drop_columns)


# %%
# Remove the collinear features
train = remove_collinear_categorical_features(train, drop_columns)
test = remove_collinear_categorical_features(test, drop_columns)

# %% [markdown]
# #### DeviceInfo(nominal categoric)

# %%
for df in [train, test]:
  column_details(regex='DeviceInfo', df=df)

# %%
# Adding new column derived from DeviceInfo ( new column: device_name)
def setDevice(df):
    
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]

    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 500].index), 'device_name'] = "Others"
    gc.collect()
    
    return df

train=setDevice(train)
test=setDevice(test)

# %%
train['device_name'].value_counts(dropna=False)

# %%
test['device_name'].value_counts(dropna=False)

# %% [markdown]
# One Hot Encoding For Device Name

# %%
# Perform one-hot encoding for device_name column on train
device_name_OHE = pd.get_dummies(train['device_name'], prefix='device_name', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
train = pd.concat([train, device_name_OHE], axis=1)

# %%
# Perform one-hot encoding for device_name column on test
device_name_OHE = pd.get_dummies(test['device_name'], prefix='device_name', drop_first=False, dtype=int, dummy_na=True)

# Add the resulting one-hot encoded data to the original DataFrames
test = pd.concat([test, device_name_OHE], axis=1)

# %%
# original 'device_name' column dropped
train.drop('device_name', axis=1, inplace=True)
test.drop('device_name', axis=1, inplace=True)

# %%
# Drop the original column
train.drop('DeviceInfo', axis=1, inplace=True)
test.drop('DeviceInfo', axis=1, inplace=True)

# %%
# i want to remove collinear features in 'device_name' columns with a threshold of 0.75
id_columns = [col for col in train.columns if col.startswith('device_name')]
drop_columns = identify_collinear_categorical_features(train, id_columns, threshold=0.75)

# Display the columns to drop
print("Columns to drop:", drop_columns)


# %%
# Remove the collinear features
train = remove_collinear_categorical_features(train, drop_columns)
test = remove_collinear_categorical_features(test, drop_columns)

# %%
gc.collect()

# %% [markdown]
# #### Pickling Final Train and Test

# %%
#pickling datasets
#Save 'train' data to a pickle file named 'train_1=3.pkl'
train.to_pickle(r'C:\Fraud_Data\data\train_12.pkl')

#save 'test' data to a pickle file named 'test_3.pkl'
test.to_pickle(r'C:\Fraud_Data\data\test_12.pkl')


# %%
# Read the 'train_3.pkl' pickle file and load it into the 'train' DataFrame
train = pd.read_pickle('./train_12.pkl')

# Read the 'test_3.pkl' pickle file and load it into the 'test' DataFrame
test = pd.read_pickle('./test_12.pkl')


