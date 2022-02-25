import pandas as pd
import matplotlib
import numpy as np
from sklearn import linear_model

train = pd.read_csv('train.csv')

data = train.select_dtypes(include=[np.number])
data1 = data.dropna(axis=1)

x_val1 = data1.drop(['SalePrice'], axis = 1)
y_val1 = data1['SalePrice']

model = linear_model.LinearRegression().fit(x_val1, y_val1)


test = pd.read_csv('test.csv')

test1 = test.select_dtypes(include=[np.number])
test2 = test1.dropna(axis=1)

# x_val2 = test2.drop(['SalePrice'], axis = 1)
# y_val2 = test2['SalePrice']

test_predict = model.predict(test2)
test_predict1 = np.round(test_predict, decimals=2)

test['SalePrice'] = test_predict1

outcsv = pd.DataFrame(data=test[['Id', 'SalePrice']])
outcsv.to_csv("predict.csv", index=False, header=True)
