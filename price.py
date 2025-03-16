import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

house = pd.read_csv('houseData.csv')

# Data Preprocessing
print(house.isnull().sum())
house.drop(columns=house.columns[0], axis=1, inplace=True)
standard = StandardScaler()


# x --> Features
# y --> price 
x = house.drop(columns='PRICE', axis=1)
y = house['PRICE']
standardData = standard.fit_transform(x)

# Splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model
model = XGBRegressor()
model.fit(x_train, y_train)

# Prediction
traningPredict = model.predict(x_train)
print('Training data prediction:', traningPredict)

# Rsquared error
score1 = metrics.r2_score(y_train, traningPredict)
print('R2 error(Training Data):', score1)

# Mean Absolute Error
score2 = metrics.mean_absolute_error(y_train, traningPredict)
print('Mean Absolute Error(Training Data):', score2)

# Testing data prediction
testingPredict = model.predict(x_test)
print('Testing data prediction:', testingPredict)

# Rsquared error
score3 = metrics.r2_score(y_test, testingPredict)
print('R2 error(Test Data):', score3)

# Mean Absolute Error
score4 = metrics.mean_absolute_error(y_test, testingPredict)
print('Mean Absolute Error(Test Data):', score4)

# Visualization(Traning Data)
plt.scatter(y_train, traningPredict)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price(Training Data)')
plt.show()

# Visualization(Testing Data)
plt.scatter(y_test, testingPredict)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price(Testing Data)')
plt.show()

# Prediction

inputData = (0.02729,0.0,7.07,0.0,0.469,7.185,61.1,4.9671,2.0,242.0,17.8,392.83,4.03)
inputDataArray = np.asarray(inputData).reshape(1, -1)
standardInputData = standard.transform(inputDataArray)
prediction = model.predict(standardInputData)
print('Predicted Price(in 1000$):', abs((prediction[0])))

