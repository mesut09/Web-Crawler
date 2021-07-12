# -*- coding: utf-8 -*-

import requests as req
import json
import sys
import pandas as pd

BaseURL = "https://www.mackolik.com/perform/p0/ajax/components/competition/livescores/json?sports[]=Basketball&matchDate="

jsons= []

for Date in range(2018,2022,1):
    for Month in range(1,13,1):
        for Day in range(1,32,1):
            Day = str(Day)
            Month = str(Month)
            Date = str(Date)
            if len(Day) != 2:
                Day = "0"+Day
            if len(Month) != 2:
                Month = "0"+Month
            MyDate = Date+"-"+Month+"-"+Day
            Resp = req.get(BaseURL+MyDate)
            MyJson = json.loads(Resp.content)
            jsons.append(MyJson)
            if(Day == 15):
                print(str(Day)+". Gün Tamamlandı")
        print(str(Month)+". Ay tamamlandı.")
    print(str(Date)+". Yıl tamamlandı")
    
    
neededdata=[]
for i in jsons:
    try:
        mathces = i["data"]["matches"].values()
        neededdata.append(mathces)  
    except:
        mathces = i["data"]["matches"]
        neededdata.append(mathces)
    

    
teams1 = []
teams2 = []
macdurums=[]
sonuclar1 = []
sonuclar2 = []
bilgiler = []
for i in neededdata:
  i = pd.DataFrame(i)
  for sayac in range(0,len(i.index)):
      isimler=i.loc[sayac]["matchName"].replace("vs",",").split(",")
      team1 = isimler[0]
      team2 = isimler[1]
      sonuc1 = i.loc[sayac]["score"]["home"]
      sonuc2 = i.loc[sayac]["score"]["away"]
      
      macdurum = i.loc[sayac]["substate"]
      
      sonuclar1.append(sonuc1)
      sonuclar2.append(sonuc2)
      teams1.append(team1)
      teams2.append(team2)

      macdurums.append(macdurum)
      

X = pd.DataFrame(columns=["Takım1","Takım2","macdurum"])
y = pd.DataFrame(columns=["Home","Away"])



uzunluk = len(teams1)
i=0
while(i<uzunluk):
    takim1 = teams1[i]
    takim2 = teams2[i]
    macdurum = macdurums[i]
    if(sonuclar1[i] == ""):
        sonuc1 = 0
    else:
        sonuc1 = sonuclar1[i]
    if(sonuclar2[i] == ""):
        sonuc2 = 0
    else:
        sonuc2 = sonuclar2[i]
    

    X = X.append(pd.Series({"Takım1":takim1,"Takım2":takim2,"macdurum":macdurum}),ignore_index=True)
    y = y.append({"Home":sonuc1,"Away":sonuc2},ignore_index=True)
    
    
    i+=1


X.to_excel("MackolikX.xlsx",sheet_name="liste",index=False)
y.to_excel("Mackoliky.xlsx",sheet_name="liste",index=False)


from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR


X = X.fillna(0)
y = y.fillna(0)

le = LabelEncoder()

X["Takım1"] = le.fit_transform(X["Takım1"])

le = LabelEncoder()
X["Takım2"] = le.fit_transform(X["Takım2"])

le = LabelEncoder()
X["macdurum"] = le.fit_transform(X["macdurum"])



imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)



# X= X.to_numpy()
y= y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)







from sklearn.metrics import r2_score

# evaluate multioutput regression model with k-fold cross-validation
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = DecisionTreeRegressor()
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
accuracyKnn = mean(n_scores)



# linear regression for multioutput regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
# create datasets
X, y = make_regression(n_samples=500, n_features=3, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = LinearRegression()
# fit model
model.fit(X, y)

accuracyLr = -r2_score(y_test,model.predict(X_test))/100000



import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
algoritmalar = ["LR","KNN"]
dogruluklar = [accuracyKnn,accuracyLr]
ax.set_ylim([0,100])
ax.bar(algoritmalar,dogruluklar)
plt.show()












