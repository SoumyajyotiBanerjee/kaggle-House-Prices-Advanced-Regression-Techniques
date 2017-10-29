# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


os.chdir("E:\kaggle")

print "Hello"

df = pd.read_csv('train.csv')

#print df.columns.values

'''
Feature Engineering part
'''
#removing coulmns with high null values
df2=df.dropna(axis=1, thresh=1200)
ls = df2.columns.values
#print df['LotConfig'].value_counts()

count = 0
c=0

cat_var = []
#1 - categorical - 0 for continues variable
type_col = []
catgorical_features = []
for e in ls:
    if df2[e].value_counts().shape[0]>20:
        type_col.append(0)
    else:
        type_col.append(1)
        catgorical_features.append(e)
        

#print type_col



temp_dummies_LotShape = df2.LotShape.str.get_dummies()

#print temp_dummies_LotShape

# Fetch the coloumn names containing null

list_null = df2.isnull().any()

need_care = []
c=0
#Need special care -1 | Replace by Mean as continues - 0 | need no care - 2
p=[]
for e in list_null:
    if (e and type_col[c]==1):
        need_care.append((ls[c],1))
    if e and type_col[c]==0:
        need_care.append((ls[c],0))
        p.append(ls[c])
    if not e:
        need_care.append((ls[c],2))
    #print c,type_col[c],ls[c], e
    c+=1
   
Y = df2['SalePrice']

print p

print  sum([True for idx,row in df2.iterrows() if any(row.isnull())])
print df2.shape
#print need_care

'''
df3=df2[df2.BsmtQual.isnull()]
dfs_dictionary = {'DF1':df2,'DF2':df3}
df5 = dfs_dictionary.drop_duplicates(keep=False)

df5.shape

need_care2 = []
c=0
list_null = df5.isnull().any()

for e in list_null:
    if (e and type_col[c]==1):
        need_care2.append((ls[c],1))
    if e and type_col[c]==0:
        need_care2.append((ls[c],0))
    if not e:
        need_care2.append((ls[c],2))
    #print c,type_col[c],ls[c], e
    c+=1
    
temp = [x for x in need_care2 if x[1]==1]

print temp



#print Y.shape





#print temp_dummies_LotShape

'''
#df2.plot( y=["LotFrontage", "MasVnrArea", "GarageYrBlt"], kind="line")

#plt.show()

df2['LotFrontage'].fillna(df2['LotFrontage'].median(), inplace=True)
df2['MasVnrArea'].fillna(df2['MasVnrArea'].median(), inplace=True)
df2['GarageYrBlt'].fillna(df2['GarageYrBlt'].median(), inplace=True)
   

print  sum([True for idx,row in df2.iterrows() if any(row.isnull())])
print df2.shape
list_null = df2.isnull().any()
p=[]
c=0
for e in list_null:
    if (e and type_col[c]==1):
        need_care.append((ls[c],1))
    if e and type_col[c]==0:
        need_care.append((ls[c],0))
        p.append(ls[c])
    if not e:
        need_care.append((ls[c],2))
    #print c,type_col[c],ls[c], e
    c+=1

print p

df3 = df2.dropna()
print df3.shape
print catgorical_features

Y = df3['SalePrice']
train = df3.drop('SalePrice',1)
print train.shape
le = LabelEncoder()
for e in catgorical_features:
    try:
        train[e] = le.fit_transform(train[e])
    except:
        print('Error encoding '+e)
        
train = train.drop('Neighborhood',1)


#print train.columns.values

train = train.drop('Id',1)

print "----",train.shape
#print train

train = train.convert_objects(convert_numeric=True)
regr = linear_model.LinearRegression()
train_2=train
regr.fit(train, Y)



#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train, Y)

model = SelectFromModel(regr, prefit=True)
X_new = model.transform(train)



print X_new.shape

index = ['Row'+str(i) for i in range(1, len(X_new)+1)]
X_new_df = pd.DataFrame(X_new, index=index)
train = X_new_df.convert_objects(convert_numeric=True)
c=0
list_col_new = X_new_df.columns.values
list_col_old = train_2.columns.values
list_already_taken = []
p=0
for e in list_col_new:
    k = X_new_df[e].iloc[0]
    c=1
    print k
    for ki in list_col_old:
        if train_2[ki].iloc[0]==k and ki not in list_already_taken:
            list_already_taken.append(ki)
            if type_col[c]==1:
                print "Catagorical var %s selected \n "%ki
                p+=1
            break
        c+=1
                
print "attributes cat--",p
print list_already_taken



X_train, X_test, y_train, y_test = train_test_split(X_new_df, Y, test_size=0.3, random_state=0)

print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape


regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)

Y_res = model.predict(X_test)

print("Mean squared error: %.2f"% mean_squared_error(y_test, Y_res))
print('Variance score: %.2f' % r2_score(y_test, Y_res))


print "END"

