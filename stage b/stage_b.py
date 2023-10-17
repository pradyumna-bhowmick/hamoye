import pandas as pd
import numpy as np
df = pd.read_csv("energydata_complete.csv")
df.head()
print(df)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['T2'].values.reshape(-1, 1), df['T6'].values.reshape(-1, 1), test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train,y_train)
pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(round(rmse,3))
del df['date']
del df['lights']
from sklearn.preprocessing import MinMaxScaler
ndf = df.drop(columns=['Appliances'])
scaler = MinMaxScaler()
norm_df = pd.DataFrame(scaler.fit_transform(ndf), columns=ndf.columns)
target = df['Appliances']
print(norm_df.head())
print(norm_df.columns)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(norm_df,target, test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train,y_train)
pred = reg.predict(X_train)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_train, pred)
print(round(mae, 3))
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_train, pred))
print(round(rmse, 3))
pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(round(rmse, 3))
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(round(rmse, 3))
from sklearn.linear_model import Ridge
ridge_reg = Ridge()
ridge_reg.fit(X_train,y_train)
pred_ridge = ridge_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, pred_ridge))
print(round(rmse, 3))
from sklearn.linear_model import Lasso
lasso_reg = Lasso()
lasso_reg.fit(X_train,y_train)
pred_lasso = lasso_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, pred_lasso))
print(round(rmse, 3))