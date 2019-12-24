#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:55:22 2019

@author: arunramji
"""
#Let's import required libraries
import pandas as pd
pd.options.display.max_rows=999
pd.options.display.max_columns =999
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('/Users/arunramji/Downloads/Sourcefiles/Kaggle_Housing_Price/train.csv')

fig , ax = plt.subplots(figsize=(12,6))

sns.distplot(df['SalePrice'])
plt.show()

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] =16.0
fig_size[1] = 4.0

x =df['SalePrice']
plt.hist(x, normed=True, bins=400)
plt.ylabel('SalePrice');

df_1 = df[df['SalePrice']<400000]

#Missing values
Null_Cols = pd.DataFrame(df.select_dtypes(include='object').isnull().sum(),columns=['Null_count'])
Null_Cols[Null_Cols.Null_count>0]
Null_Cols[(Null_Cols.Null_count/len(df_1))>0.8]

df_1 = df_1.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
df_1.head()

#Missing numeric Variable
dataset = df_1.select_dtypes(include='float')
pd.DataFrame(dataset.isnull().sum(),columns=['Null_count'])

#Imputing missing value for numeric variable
df_1['LotFrontage'].fillna(np.mean(df_1['LotFrontage']),inplace=True)
df_1['MasVnrArea'].fillna(np.mean(df_1['MasVnrArea']),inplace=True)
df_1['GarageYrBlt'].fillna(np.round(np.mean(df_1['GarageYrBlt'])),inplace=True)  #rounding of the value as it is year value
#np.round(np.mean(df_1['GarageYrBlt']))

df_1.sample(10)

#Categorical Variable
df_1.select_dtypes(include='object').count()

#dummy encoding for nominal variable
df_1 = pd.get_dummies(df_1,columns=['MSZoning','Street','Utilities','LotConfig','Neighborhood','Condition1'
                                    ,'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st'
                                   ,'Exterior2nd','MasVnrType','Foundation','Heating','CentralAir'
                                   ,'GarageType','SaleType','SaleCondition','MasVnrType','LandContour'],drop_first=True)

#drop_first tells to drop one of the encoded variable as it may cause "dummy variabel trap" .

data = df_1.select_dtypes(include=object).isna().sum()
data = pd.DataFrame(data,columns=['Count'])
data[data['Count']>0]

cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
    'BsmtFinType2','Electrical','FireplaceQu','FireplaceQu'
    ,'GarageFinish','GarageQual','GarageCond']

#replace missing value with new category
df_1[cols] = df_1[cols].replace({np.nan:'Unknown'}) #Replacing missing values with 'Unknown'

#encoding categorical variable using number 
dict_BsmtQual = {"BsmtQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_BsmtCond = {"BsmtCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_BsmtExposure = {"BsmtExposure":{"Gd":5,"Av":4,"Mn":3,"No":2,"Unknown":0}}
dict_BsmtFinType1 = {"BsmtFinType1":{"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"Unknown":0}}
dict_BsmtFinType2 = {"BsmtFinType2":{"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"Unknown":0}}
dict_Electrical = {"Electrical":{"SBrkr":6,"FuseA":5,"FuseF":4,"FuseP":3,"Mix":0,"Unknown":0}}
dict_FireplaceQu = {"FireplaceQu":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_GarageFinish = {"GarageFinish":{"Fin":6,"RFn":5,"Unf":4,"Unknown":0}}
dict_GarageQual = {"GarageQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_GarageCond = {"GarageCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_LotShape = {"LotShape":{"Reg":5,"IR1":4,"IR2":3,"IR3":2}}
dict_LandSlope = {"LandSlope":{"Gtl":5,"Mod":4,"Sev":3}}
dict_ExterQual = {"ExterQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_ExterCond = {"ExterCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_HeatingQC = {"HeatingQC":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_KitchenQual = {"KitchenQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}
dict_Functional = {"Functional":{"Typ":5,"Min1":4,"Min2":3,"Mod":2,"Maj1":1,"Maj2":0,
                                "Sev":0}}
dict_PavedDrive = {"PavedDrive":{"Y":3,"P":2,"N":1}}


for i in [dict_BsmtQual,dict_BsmtCond,dict_BsmtExposure,dict_BsmtFinType1,dict_BsmtFinType2,dict_Electrical
         ,dict_FireplaceQu,dict_GarageFinish,dict_GarageQual,dict_GarageCond,dict_LotShape,dict_LandSlope
         ,dict_ExterQual,dict_ExterCond,dict_HeatingQC,dict_KitchenQual,dict_Functional,dict_PavedDrive] :
    #print(type(i))
    df_1.replace(i,inplace=True)
    
df_1.shape

'''#Imputing categorical variable with most frequent values
df_1 = df_1.fillna(df_1['GarageFinish'].value_counts().index[0]) #fill NaNs with the most frequent value from that column.
df_1 = df_1.fillna(df_1['BsmtQual'].value_counts().index[0])
df_1 = df_1.fillna(df_1['GarageType'].value_counts().index[0])
df_1 = df_1.fillna(df_1['GarageQual'].value_counts().index[0])
df_1 = df_1.fillna(df_1['GarageCond'].value_counts().index[0])
df_1 = df_1.fillna(df_1['BsmtCond'].value_counts().index[0])
df_1 = df_1.fillna(df_1['BsmtExposure'].value_counts().index[0])
df_1 = df_1.fillna(df_1['BsmtFinType1'].value_counts().index[0])
df_1 = df_1.fillna(df_1['FireplaceQu'].value_counts().index[0])'''

'''#encoding categorical variable
df_1 = pd.get_dummies(df_1, columns=['FireplaceQu','MSZoning','Street','LotShape',
                                           'LandContour','Utilities','LotConfig',
                                           'LandSlope','Neighborhood','Condition1',
                                           'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
                                           'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])
 
#Dropping the following features as they are not available in test set
df_1 = df_1.drop(['Condition2_RRAe','Exterior2nd_Other','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Heating_Floor','Heating_OthW','Electrical_Mix','GarageQual_Ex', 'Exterior1st_Stone','Utilities_NoSeWa'], axis=1)
df_1.shape

#variable assignment
#data = df_1.drop(columns='SalePrice')
#Variable Assignment
#X = data.values
#y = df_1['SalePrice']'''

#preparing test set
dataset = pd.read_csv('/Users/arunramji/Downloads/Sourcefiles/Kaggle_Housing_Price/test.csv')
dataset.drop('Id',axis=1,inplace=True)

dataset.shape

dataset =  dataset.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)

#imputing missing value
dataset['LotFrontage'].fillna(np.mean(df_1['LotFrontage']),inplace=True)
dataset['MasVnrArea'].fillna(np.mean(df_1['MasVnrArea']),inplace=True)
dataset['GarageYrBlt'].fillna(np.round(np.mean(df_1['GarageYrBlt'])),inplace=True)  #rounding of the value as it is year value
#np.round(np.mean(df_1['GarageYrBlt']))

#dummy encoding
dataset = pd.get_dummies(dataset,columns=['MSZoning','Street','Utilities','LotConfig','Neighborhood','Condition1'
                                    ,'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st'
                                   ,'Exterior2nd','MasVnrType','Foundation','Heating','CentralAir'
                                   ,'GarageType','SaleType','SaleCondition','MasVnrType','LandContour'],drop_first=True)

#drop_first tells to drop one of the encoded variable as it may cause "dummy variabel trap" .

#replace missing value
cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
    'BsmtFinType2','Electrical','FireplaceQu','FireplaceQu'
    ,'GarageFinish','GarageQual','GarageCond']
dataset[cols] = dataset[cols].replace({np.nan:'Unknown'}) #Replacing missing values with 'Unknown'

for i in [dict_BsmtQual,dict_BsmtCond,dict_BsmtExposure,dict_BsmtFinType1,dict_BsmtFinType2,dict_Electrical
         ,dict_FireplaceQu,dict_GarageFinish,dict_GarageQual,dict_GarageCond,dict_LotShape,dict_LandSlope
         ,dict_ExterQual,dict_ExterCond,dict_HeatingQC,dict_KitchenQual,dict_Functional,dict_PavedDrive] :
    #print(type(i))
    dataset.replace(i,inplace=True)


    
'''#categorical missing value
dataset = dataset.fillna(dataset['GarageFinish'].value_counts().index[0]) #fill NaNs with the most frequent value from that column.
dataset = dataset.fillna(dataset['BsmtQual'].value_counts().index[0])
dataset = dataset.fillna(dataset['FireplaceQu'].value_counts().index[0])
dataset = dataset.fillna(dataset['GarageType'].value_counts().index[0])
dataset = dataset.fillna(dataset['GarageQual'].value_counts().index[0])
dataset = dataset.fillna(dataset['GarageCond'].value_counts().index[0])
dataset = dataset.fillna(dataset['GarageFinish'].value_counts().index[0])
dataset = dataset.fillna(dataset['BsmtCond'].value_counts().index[0])
dataset = dataset.fillna(dataset['BsmtExposure'].value_counts().index[0])
dataset = dataset.fillna(dataset['BsmtFinType1'].value_counts().index[0])
dataset = dataset.fillna(dataset['BsmtFinType2'].value_counts().index[0])
dataset = dataset.fillna(dataset['BsmtUnfSF'].value_counts().index[0])

#encoding categorical variable
dataset = pd.get_dummies(dataset, columns=['FireplaceQu','MSZoning','Street','LotShape',
                                           'BsmtUnfSF',
                                           'LandContour','Utilities','LotConfig',
                                           'LandSlope','Neighborhood','Condition1',
                                           'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
                                           'MasVnrType','ExterQual','ExterCond',
                                           'Foundation','BsmtQual','BsmtCond',
                                           'BsmtExposure','BsmtFinType1','BsmtFinType2',
                                           'BsmtFinSF1','BsmtFinSF2','TotalBsmtSF',
                                           'BsmtFullBath','BsmtHalfBath','GarageCars'
                                           ,'GarageArea',
                                           'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])
 '''
dataset.shape

final_train, final_test = df_1.align(dataset,join='inner', axis=1)
data = final_train

final_train.shape
final_test.shape

#Variable assignment
X_train = data.values
y_train = df_1['SalePrice']

#XGBOOST
from xgboost import XGBRegressor
xgb_clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_clf.fit(X_train, y_train)

#Accuracy after cross validation with X
X_test = final_test.values

from sklearn.model_selection import cross_val_score
xgb_clf_cv = cross_val_score(xgb_clf,X_train,y_train,cv=10)

print(xgb_clf_cv.mean())   #0.9049113535433122

xgb_prediction_test = xgb_clf.predict(X_test)

final_prediction = pd.DataFrame(xgb_prediction_test)


#final_train.columns == final_test.columns


d#ataset.select_dtypes(include='object').isnull().sum()


#df.select_dtypes(include='object').isnull().sum()

#df.columns

