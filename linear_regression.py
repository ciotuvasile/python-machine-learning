# daca esti curios testeaza in jupyter notebook

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#get_ipython().run_line_magic('matplotlib', 'inline')
%matplotlib inline

# df e deja curatat de NaN si ?
df = pd.read_csv('FuelConsumption.csv')

#df.head()

#df.describe()

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('CYLINDERS')
plt.ylabel('Emissions')
plt.show()


# din DataFrame (df) => 80% pentru training | 20% pentru test

# selectez 80% din inregistrari
mask = np.random.rand(len(df)) < 0.8

train = cdf[mask]
test = cdf[~mask]   # ce nu e inclus in training merge in test ~mask

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()


from sklearn import linear_model

regr = linear_model.LinearRegression()

#train.head(5)

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x, train_y)

# The coefficients
print('Reger. coef (sau slope): ', regr.coef_)
print('Intercept: ', regr.intercept_)

# yh = Th0          +  Th1                   * x1
#      (intercept)    (slope sau regr_coef)  

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')

plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')

plt.xlabel('Engine size')
plt.ylabel('Emission')


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

#print(test_y_)

print("MAE: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("MSE: %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2 s: %.2f" % r2_score(test_y_ , test_y) )

# just for fun
res = pd.DataFrame({'Eng. Size': test_x[:, 0], 'CO2E Cunoscut': test_y[:, 0], 'CO2Prezis': test_y_[:, 0]})
res.head()