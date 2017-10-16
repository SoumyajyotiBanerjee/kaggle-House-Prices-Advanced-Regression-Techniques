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

for e in ls:
    print df2[e].value_counts()
    count+=1
    if count>20:
        break




temp_dummies_LotShape = df2.LotShape.str.get_dummies()

print temp_dummies_LotShape


#print temp_dummies_LotShape


    

    