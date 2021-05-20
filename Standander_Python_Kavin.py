import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings('ignore')

#Set working directory
os.chdir("D:\Kavin\Edwisor\Dataset\Santander")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

## Basic Structure of Train and Test Dataset
print(df_train.info())
df_train.describe()
print(df_test.info())
df_test.describe()
print('   \n    ')
#########################################Missing value analysis#########################################
## Missing Value Analysis
print('Missing Value Analysis:')
# train data
if df_train.isnull().sum().sum()>0:
    Print('Training data set have missing Values')
else:
    print("Training data set doesn't have missing Values")

# train data
if df_train.isnull().sum().sum()>0:
    Print('Training data set have missing Values')
else:
    print("Training data set doesn't have missing Values")
#########################################Outlier Analysis#########################################
## Outlier Analysis
print('   \n    ')
print('Outlier Analysis :')
Independent_Variables=list(df_train.columns[2:])

#seperating the Column in to Group to vizulaize the outliers in boxplot
n = 10
chunk_set = []
for x in range(0, len(Independent_Variables), n):
    #chunks.append(xx + n)
    chunk_set.append(Independent_Variables[x:x+n])

#Show bos plot in set group
for i in chunk_set:
    plt.show(df_train.boxplot(column =i,figsize=(10,5)))

## Removing Ouliers using chauvenet Criterion
#Removing the outliers using the chauvenet way:
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.    
    return prob < criterion       # Use boolean array outside this function

from scipy.special import erfc
Outilers = dict()
for col in [col for col in Independent_Variables]:
    Outilers[col] = df_train[chauvenet(df_train[col].values)].shape[0]
df_train_Outilers = pd.Series(Outilers)

from scipy.special import erfc
Outilers1 = dict()
for col in [col for col in Independent_Variables]:
    Outilers1[col] = df_test[chauvenet(df_test[col].values)].shape[0]
df_test_Outilers = pd.Series(Outilers1)

print('Total number of outliers in Training dataset: {} ({:.2f}%)'.format(sum(df_train_Outilers.values), (sum(df_train_Outilers.values) / df_train.shape[0]) * 100))
print('Total number of outliers in Test dataset: {} ({:.2f}%)'.format(sum(df_test_Outilers.values), (sum(df_test_Outilers.values) / df_test.shape[0]) * 100))


from scipy.special import erfc
#remove these outliers in train and test data
for col in Independent_Variables:
    df_train=df_train.loc[(~chauvenet(df_train[col].values))]
for col in Independent_Variables:
    df_test=df_test.loc[(~chauvenet(df_test[col].values))]
print('    \n         ')
print('No of rows in Training Data set has reduced to:{} Rows and with {} Column count'.format(df_train.shape[0],df_train.shape[1]))
print('No of rows in Test     Data set has reduced to:{} Rows and with {} Columns'.format(df_test.shape[0],df_test.shape[1]))
print('    \n         ')
#########################################Feature Selection#########################################

## Feature Selection
print('Feature Selection :')
df_corr = df_train.loc[:,Independent_Variables]

#Function to find the correaltion values between Variables
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=3):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
print('     \n         ')
print("Top 10 Absolute Correlations in Training Dataset")
print(get_top_abs_correlations(df_train[Independent_Variables], 10))
print('     \n         ')
print('Maximum correlation between independent variables in Training DataSet : {}'.format(get_top_abs_correlations(df_train[Independent_Variables], 1)))

print('     \n         ')
print("Top 10 Absolute Correlations in Test Dataset")
print(get_top_abs_correlations(df_test[Independent_Variables], 10))
print('    \n         ')
print('Maximum correlation between independent variables in Test Dataset : {}'.format(get_top_abs_correlations(df_test[Independent_Variables], 1)))	


print('    \n         ')
## Heat Map for Training Dataset
print('  Heat Map for Training Dataset  ')
#Heat Map for Train Dataset
#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
kot = corr[corr>=.5]
#f,ax=plt.figure(figsize=(12,8))
f, ax = plt.subplots(figsize=(40, 40))
sns.set(font_scale=0.5)
sns.heatmap(kot, cmap="Greens",square=True, ax=ax,annot_kws={'size': 3},annot = True,fmt='.4g')
ax.set_yticklabels(kot.columns,  rotation=360, fontsize=8)
ax.set_xticklabels(kot.columns,  rotation=90,fontsize=8)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
print('    \n         ')
######################################### Distribution of Data#########################################

print(' Cheking the Distribution of Data :')
##Cheking the Distribution of Data
print('Distribution of mean per row in the train and test dataset')
plt.figure(figsize=(15,5))
sns.distplot(df_train[Independent_Variables].mean(axis=1),color="green", kde=True,bins=100, label='train')
sns.distplot(df_test[Independent_Variables].mean(axis=1),color="Red", kde=True,bins=100, label='test')
plt.title('Distribution of mean per row in the train and test dataset')
plt.legend()
plt.show()
print('     \n         ')
print('Distribution of std per row in the train and test dataset')
plt.figure(figsize=(15,5))
sns.distplot(df_train[Independent_Variables].std(axis=1),color="green", kde=True,bins=100, label='train')
sns.distplot(df_test[Independent_Variables].std(axis=1),color="Red", kde=True,bins=100, label='test')
plt.title('Distribution of std per row in the train and test dataset')
plt.legend()
plt.show()
print('     \n         ')
#########################################Handling  Imbalanced Dataset#########################################

## Handling  Imbalanced Dataset
print(' Handling  Imbalanced Dataset :      ')
print('Display Imbalance class counts \n',df_train.target.value_counts())

print(' Display Imbalance class counts \n')
sns.set(font_scale=0.9)
sns.factorplot('target', data=df_train, kind='count')

## Using Down sampling method to hadble imbalanced Dataset

from sklearn.utils import resample
# Separate majority and minority classes
df_majority = df_train[df_train.target==0]
df_minority = df_train[df_train.target==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=19908,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
print('Display new class counts \n',df_downsampled.target.value_counts())

print(' Display New class counts After Down Sampling\n ')
sns.set(font_scale=0.9)
sns.factorplot('target', data=df_downsampled, kind='count')
print('     \n         ')

#Seperating the  independent variables and dependednt variables 
X=df_downsampled.iloc[:,2:]
y=df_downsampled.iloc[:,1]

#########################################Information Gain#########################################

## Information Gain
from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(X,y)


#Find the independent variables that caries information about Target Variable
mutual_data=pd.DataFrame(mutual_info,index=X.columns)
#Reset index
mutual_data = mutual_data.reset_index()
#Rename variable
mutual_data = mutual_data.rename(columns = {'index': 'Variables', 0: 'infogain'})
mutual_data.sort_values(by=['infogain'],ascending=False)

#### Picking the Independent Variables that carries information about the Dependent Variables

#mutual_data[mutual_data['infogain']>0.000000].columns
finalColumn=[]
mc=mutual_data.loc[mutual_data['infogain']>0.000000]['Variables'].values
for i,p in enumerate(mc):
    #print(p)
    finalColumn.append(p)
#finalColumn

#Assigning only requied variables to new dataset
XX=X[finalColumn]

##################################################################################################################

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(XX,y,random_state=100,test_size=0.2)

##Standardizing the Train and Test Data

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#########################################LogisticRegression Algorithm#########################################


## LogisticRegression Algorithm
print('LogisticRegression Model')
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(n_jobs=-1)

model.fit(X_train,y_train)

LR_pred=model.predict(X_test)

#Performance Metrics for LogisticRegression Algorithm
print('   \n     ')
print('Performance Metrics for LogisticRegression Algorithm')
from sklearn.metrics import accuracy_score
CM_LR=confusion_matrix(y_test, LR_pred)
TN = CM_LR[0,0]
FN = CM_LR[1,0]
TP = CM_LR[1,1]
FP = CM_LR[0,1]
#print(CM_DT_WO_outliers)
print("Dataset Without Outliers")
print("F1 Score :",f1_score(LR_pred,y_test,average = "weighted"))
print('Report:\n',classification_report(y_test, LR_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test, LR_pred))
print('Accuracy_score% = ',(accuracy_score(y_test, LR_pred)*100))
print('Number of correctly classified samples = ',(accuracy_score(y_test, LR_pred, normalize=False)))
print('FalseNegativeRate =',(FN*100)/(FN+TP))
print('FalsePositiveRate =',(FP*100)/(FP+TN))
print('ROC_AUC_SCORE :', roc_auc_score(y_test, LR_pred))

#########################################RandomForest Algorithm#########################################

# ## RandomForest Algorithm
print('RandomForest Model')
# from sklearn.ensemble import RandomForestClassifier
# classifier=RandomForestClassifier(n_estimators=1500,random_state=123)
# classifier.fit(X_train,y_train)


# RF_Pred=classifier.predict(X_test)

# #Performance Metrics for RandomForestClassifier Algorithm
# print('   \n     ')
# print('Performance Metrics for RandomForestClassifier Algorithm')
# CM_RF=confusion_matrix(y_test, RF_Pred)
# TN = CM_RF[0,0]
# FN = CM_RF[1,0]
# TP = CM_RF[1,1]
# FP = CM_RF[0,1]
# #print(CM_DT_WO_outliers)
# print("Dataset Without Outliers")
# print("F1 Score :",f1_score(RF_Pred,y_test,average = "weighted"))
# print('Report:\n',classification_report(y_test, RF_Pred))
# print('Confusion Matrix: \n',confusion_matrix(y_test, RF_Pred))
# print('Accuracy_score% = ',(accuracy_score(y_test, RF_Pred)*100))
# print('Number of correctly classified samples = ',(accuracy_score(y_test, RF_Pred, normalize=False)))
# print('FalseNegativeRate =',(FN*100)/(FN+TP))
# print('FalsePositiveRate =',(FP*100)/(FP+TN))
# print('ROC_AUC_SCORE :', roc_auc_score(y_test, RF_Pred))

#########################################XGboost Classifer#########################################

## XGboost Classifer
print('XGboost Classifer Model')
from xgboost import XGBClassifier
xgboost_model = XGBClassifier()
xgboost_model.fit(X_train,y_train)

XG_pred = xgboost_model.predict(X_test)

#Performance Metrics for XGboot  Algorithm
print('   \n     ')
print('Performance Metrics for XGboot  Algorithm')
from sklearn.metrics import accuracy_score
CM_XG=confusion_matrix(y_test, XG_pred)
TN = CM_XG[0,0]
FN = CM_XG[1,0]
TP = CM_XG[1,1]
FP = CM_XG[0,1]
#print(CM_DT_WO_outliers)
print("Dataset Without Outliers")
print("F1 Score :",f1_score(XG_pred,y_test,average = "weighted"))
print('Report:\n',classification_report(y_test, XG_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test, XG_pred))
print('Accuracy_score% = ',(accuracy_score(y_test, XG_pred)*100))
print('Number of correctly classified samples = ',(accuracy_score(y_test, XG_pred, normalize=False)))
print('FalseNegativeRate =',(FN*100)/(FN+TP))
print('FalsePositiveRate =',(FP*100)/(FP+TN))
print('ROC_AUC_SCORE :', roc_auc_score(y_test, XG_pred))

########################################Predicting the Test Dataset############################################

##Predicting the Test Dataset
print('  \n     ')
print('Predicting the Test Dataset :   \n     ')
##Using only the Independent variables
##using only the independent Varibles with high correlation with Target varibles 
df_test_InformationVariables=df_test[finalColumn]
##Standardizing process
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(df_test_InformationVariables)
df_test_std=sc.fit_transform(df_test_InformationVariables)

##Using the Logistic Regression Model to predict Target of Test Data
Predict_Target=model.predict(df_test_std)
df_test['Target']=Predict_Target
print('Test Dataset Predicted variable\n ',df_test['Target'].value_counts())
df_test.to_csv('Logistic_Regression_Prediction_testDataset.csv', index=False)