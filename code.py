# --------------
## Load the data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

data=pd.read_csv(path)
#print(data.shape)
#print(data.head(5))

## Split the data and preprocess
train=data[data['source']=='train']
#print(train.shape)
test=data[data['source']=='test']
#print(test.shape)

## Baseline regression model
columns_to_use=['Item_Weight', 'Item_MRP', 'Item_Visibility']
target='Item_Outlet_Sales'
X_Train, X_Test, y_Train, y_Test = train_test_split(train.loc[:, columns_to_use],train[target], test_size=0.3, random_state=43)

linmodel=LinearRegression()
linmodel.fit(X_Train, y_Train)
pred1=linmodel.predict(X_Test)
print('The baseline mean squared error is', mean_squared_error(y_Test, pred1))
print('The baseline r squared is', r2_score(y_Test,pred1))

## Effect on R-square if you increase the number of predictors
columns_to_avoid=['Item_Outlet_Sales','Item_Identifier','Unnamed: 0','source']
columns_to_use1= [col for col in train.columns if col not in columns_to_avoid]

X_train2, X_test2, y_train2, y_test2= train_test_split(train.loc[:,columns_to_use1], train[target], test_size=0.3, random_state=43)
linmodel.fit(X_train2, y_train2)
pred2=linmodel.predict(X_test2)

print('The revised r square of increased predictors is', r2_score(y_test2, pred2))

## Effect of decreasing feature from the previous model
columns_to_avoid3=['Item_Outlet_Sales','Item_Identifier', 'Item_Visibility', 'Outlet_Years','Unnamed: 0','source']
columns_to_use3 = [col for col in train.columns if col not in columns_to_avoid3]

X_train3, X_test3, y_train3, y_test3= train_test_split(train.loc[:,columns_to_use3], train[target], test_size=0.3, random_state=43)
linmodel.fit(X_train3, y_train3)
pred3=linmodel.predict(X_test3)

print('The revised r square with decreased predictors is', r2_score(y_test3, pred3))
## Detecting hetroskedacity
plt.scatter(pred3,y_test3-pred3)

## Model coefficients
coeff=pd.DataFrame(linmodel.coef_, X_train3.columns, columns=['Coefficient'])
print(coeff)
## Ridge regression
ridgereg=Ridge()
ridgereg.fit(X_train2, y_train2)
pred4=ridgereg.predict(X_test2)

print('The ridge r square is', r2_score(y_test2, pred4))
## Lasso regression
Lassoreg=Lasso()
Lassoreg.fit(X_train2, y_train2)
pred5=Lassoreg.predict(X_test2)

print('The Lasso r square is', r2_score(y_test2, pred5))

## Cross vallidation
score=cross_val_score(linmodel, X_train2,y_train2, cv=10)
mean_score=np.mean(score)
print(mean_score)


