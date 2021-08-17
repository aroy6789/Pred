#importing some required libraries
import pandas as pd
import numpy as np
import pickle

#importing required csv files
rent_df1=pd.read_csv('day.csv')

#Renaming the columns of rent_df1 for better Understanding of Variables
rent_df1.rename(columns={'instant':'Rec_id','dteday':'Date','yr':'Year','mnth':'Month','weathersit':'Weather','temp':'Temp',
                       'hum':'Humidity','cnt':'Total_Count'},inplace=True)

#Deleted unwanted columns
rent_df1 = rent_df1.drop(columns = ['Rec_id','atemp','Date','Year','holiday','weekday','workingday','Weather','casual','registered'])

X = rent_df1.drop(['Humidity','windspeed','Total_Count'], axis = "columns")
Y0 = rent_df1['Total_Count']
Y2 = rent_df1['Humidity']
Y3 = rent_df1['windspeed']

from sklearn.model_selection import train_test_split
X_train,X_test,Y0_train,Y0_test,Y2_train,Y2_test,Y3_train,Y3_test=train_test_split(X,Y0,Y2,Y3,test_size=.10,random_state=0)

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
RFModel_Total_Count= RandomForestRegressor(n_estimators=100).fit(X_train,Y0_train)
RFModel_Humidity=RandomForestRegressor(n_estimators=100).fit(X_train,Y2_train)
RFModel_windspeed=RandomForestRegressor(n_estimators=100).fit(X_train,Y3_train)

# Saving model to current directory
pickle.dump(RFModel_Total_Count,open('model_Total_Count.pkl','wb'))
pickle.dump(RFModel_Humidity,open('model_Humidity.pkl','wb'))
pickle.dump(RFModel_windspeed,open('model_windspeed.pkl','wb'))

#Loading model to compare the results
model_Total_Count = pickle.load(open('model_Total_Count.pkl','rb'))
model_Humidity = pickle.load(open('model_Humidity.pkl','rb'))
model_windspeed = pickle.load(open('model_windspeed.pkl','rb'))
print(RFModel_Total_Count.predict([[1, 1, 0.34]]))
print(RFModel_Humidity.predict([[1, 1, 0.34]]))
print(RFModel_windspeed.predict([[1, 1, 0.34]]))