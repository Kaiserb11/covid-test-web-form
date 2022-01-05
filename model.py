import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_excel('COVID-19.xlsx')
con = preprocessing.LabelEncoder()
less = 'Sno'
predict = 'Corona result'

X = np.array(data.drop([predict, less],1))
y = np.array(data[predict])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=100)


rand_class = RandomForestClassifier(n_estimators=400,random_state=0)
sv = rand_class.fit(X_train, y_train)


pickle.dump(sv, open('model.pkl', 'wb'))