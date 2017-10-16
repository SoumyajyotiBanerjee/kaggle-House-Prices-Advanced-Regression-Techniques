# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os

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
        

print type_col



temp_dummies_LotShape = df2.LotShape.str.get_dummies()

#print temp_dummies_LotShape

# Fetch the coloumn names containing null

list_null = df2.isnull().any()

need_care = []
c=0
#Need special care -1 | Replace by Mean as continues - 0 | need no care - 2
for e in list_null:
    if (e and type_col[c]==1):
        print "HERE IN CAT"
        need_care.append((ls[c],1))
    if e and type_col[c]==0:
        print "HERE IN CONT"
        need_care.append((ls[c],0))
    if not e:
        print "HERE IN LAST"
        need_care.append((ls[c],2))
    print c,type_col[c],ls[c], e
    c+=1
   
Y = df2['SalePrice']
#print need_care
#print Y.shape





#print temp_dummies_LotShape


    

    