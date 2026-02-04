# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import required libraries for data handling, visualization, and machine learning.

2.  Load the dataset CarPrice_Assignment.csv using Pandas.

3.  Select the independent variables (enginesize, horsepower, citympg, highwaympg) and the dependent variable (price).

4.  Split the dataset into training (80%) and testing (20%) sets.

5.  Apply StandardScaler to normalize the training and testing data.

6.  Create a Linear Regression model using LinearRegression().

7.  Train the model using the scaled training data.

8.  Predict car prices using the trained model on test data.

9.  Evaluate the model using MSE, MAE, RMSE, and R-squared metrics.

10.  Analyze model assumptions using plots and statistical tests (Actual vs Predicted, residual plots, Durbin–Watson test, and Q-Q plot).

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('CarPrice_Assignment.csv')

x = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled =scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
print('Name:JANNATHUL SHABAN.A')
print('Reg.No:212225220043')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10}")
print(f"{'Intercept':>12}: {model.intercept_:>10}")

print("\nMODEL PERFORMANCE:")
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rse=np.sqrt(mse)
print(f"{'MSE':>12}: {mse:>10}")
print(f"{'MASE':>12}: {mae:>10}")
print(f"{'RMSE':>12}:{rse:>10}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(), y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y=residuals, lowess=True,line_kws={'color':'red'})
plt.title("Homescedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

fig, (ax1, ax2)=plt.subplots(1, 2,figsize=(12,5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45',fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()

```

## Output:
<img width="350" height="189" alt="Screenshot 2026-02-04 142947" src="https://github.com/user-attachments/assets/b7c708b5-b646-47e8-9095-19e4699b7cf5" />

<img width="363" height="147" alt="Screenshot 2026-02-04 143009 - Copy" src="https://github.com/user-attachments/assets/40500105-0296-4758-bd46-ffd11349f0f2" />

# Linearity Check: Actual vs Predicted Prices
<img width="1144" height="588" alt="Screenshot 2026-02-04 143026" src="https://github.com/user-attachments/assets/e5733b67-5e00-4138-b50b-3e02cf87cf32" />

# Homescedasticity Check: Residuals vs Predicted

<img width="1253" height="523" alt="Screenshot 2026-02-04 143132" src="https://github.com/user-attachments/assets/7712ffc6-13b2-452f-b40a-d0432172c9c9" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
