# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


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
#1 - categorical - 0 for continues variable
type_col = []

for e in ls:
    if df2[e].value_counts().shape[0]>20:
        type_col.append(0)
    else:
        type_col.append(1)
        

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