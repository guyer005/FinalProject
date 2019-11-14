# -*- coding: utf-8 -*-
#https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt  
from sklearn import datasets
from sklearn import linear_model
from regressors import stats
import statsmodels.api as sm 

path = "C:/Users/rick.guyer/Desktop/FIN_PROJECT/original_data_train.csv"
df = pd.read_csv(path)
#df = pd.read_csv("original_data_train.csv")
# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
df.head()
for col in df.columns:
    print(col)

# Set features. This will also be used as your x values.
NBA_x = df[['FGM Per 100 Poss', 'FTM Per 100 Poss', 'OREB Per 100 Poss', 'DREB Per 100 Poss','OPP FGM Per 100 Poss',
'OPP FTM Per 100 Poss','OPP OREB Per 100 Poss','OPP DREB Per 100 Poss']]
NBA_Label = ['FGM Per 100 Poss', 'FTM Per 100 Poss', 'OREB Per 100 Poss', 'DREB Per 100 Poss','OPP FGM Per 100 Poss',
'OPP FTM Per 100 Poss','OPP OREB Per 100 Poss','OPP DREB Per 100 Poss']
#kepler_x = df[['koi_fpflag_nt']]
NBA_y = df[['WIN%']]*82
#part1 = ['2009-10','2010-11','2011-12','2012-13','2013-14']
part2 = ['2014-15','2015-16','2016-17','2017-18','2018-19']
trainx = []
trainy = []

for x in range(len(NBA_x)):
    if df.iloc[x,1] not in part2:
        trainx.append(x)
    else:
        trainy.append(x)
#We are not using random sampling here. Since we have multiple seasons for NBA Teams
#We are splitting the ten years of data to see how years 2009-2014 compare to 2014-19
#Then flip the script and see what happens to our fit. 
#Do any variables demonstrate temporal model fit differences?        
NBAX_train = NBA_x.iloc[trainx,:]
NBAX_test = NBA_x.iloc[trainy,:]
NBAY_train = NBA_y.iloc[trainx,:]
NBAY_test = NBA_y.iloc[trainy,:]

regressor = LinearRegression()
regressor.fit(NBAX_train, NBAY_train) #training the algorithm

ols = linear_model.LinearRegression()
ols.fit(NBAX_train, NBAY_train)
print("coef_pval:\n", stats.coef_pval(ols, NBAX_train, NBAY_train))
Data1 = [stats.coef_tval(ols, NBAX_train, NBAY_train),stats.coef_pval(ols, NBAX_train, NBAY_train)]
Data2 = [stats.coef_tval(ols, NBAX_test, NBAY_test),stats.coef_pval(ols, NBAX_test, NBAY_test)]
stats.adj_r2_score(ols, NBAX_train, NBAY_train)
stats.adj_r2_score(ols, NBAX_test, NBAY_test)

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(NBAX_test)
np.savetxt("predicted.csv",y_pred,delimiter=",")
predY = pd.read_csv("predicted.csv")
#pd.DataFrame({"NBAY_test":NBAY_test,"y_pred":y_pred})

df.iloc[trainy,:].to_csv('NBATest.csv')

plt.scatter(NBAX_test.iloc[:,0], NBAY_test,  color='gray')
plt.plot(NBAX_test.iloc[:,0], predY, color='red', linewidth=2)
plt.show()