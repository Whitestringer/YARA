import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import datasets

df = pd.read_csv('C:\BostonHousing.csv')
print (df)
plt.clf()
corr = df.drop('medv', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
import statsmodels.formula.api as smf
lm = smf.ols("medv~rad+tax+age+lstat", df).fit()
print(lm.summary() )



df.shape
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)
df_train.shape, df_test.shape
from sklearn.metrics import r2_score, mean_squared_error
lm = smf.ols("medv~rad+tax+age+lstat", df_train).fit()
pr_data = lm.get_prediction(df_test)
r2_score( pr_data.predicted, df_test.medv),
mean_squared_error( pr_data.predicted, df_test.medv)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
x = df_train[['rad', 'tax', 'age','lstat']]
y= df_train['medv']
lm = LinearRegression()
scores = cross_val_score(lm, x, y, scoring='r2', cv=5)
np.mean(scores)

X = [df_train[['rad', 'tax', 'age','lstat']], 
     df_train[['rad', 'age', 'nox', 'crim']],
     df_train[['tax','rm','nox','dis']],
     df_train[['dis','zn','rm']],
     df_train[['ptratio','crim','rad','nox']]]
lm = LinearRegression()
for i, i_df in enumerate(X):
  scores = cross_val_score(lm, i_df, y, scoring='r2', cv=5)
  print(f'data index = {i}, mean = {np.mean(scores)}')
reg = lm.fit(X[2], y)
pr_data = reg.predict(df_test[['tax','rm','nox','dis']])
from sklearn.metrics import r2_score, mean_squared_error
r2_score( pr_data, df_test.medv),
mean_squared_error( pr_data, df_test.medv)

reg = lm.fit(X[0], y)
pr_data = reg.predict(df_test[['rad', 'tax', 'age','lstat']])
from sklearn.metrics import r2_score, mean_squared_error
r2_score( pr_data, df_test.medv),
mean_squared_error( pr_data, df_test.medv)

reg = lm.fit(X[1], y)
pr_data = reg.predict(df_test[['rad', 'age', 'nox', 'crim']])
from sklearn.metrics import r2_score, mean_squared_error
r2_score( pr_data, df_test.medv),
mean_squared_error( pr_data, df_test.medv)

reg = lm.fit(X[3], y)
pr_data = reg.predict(df_test[['dis','zn','rm']])
from sklearn.metrics import r2_score, mean_squared_error
r2_score( pr_data, df_test.medv),
mean_squared_error( pr_data, df_test.medv)

reg = lm.fit(X[4], y)
pr_data = reg.predict(df_test[['ptratio','crim','rad','nox']])
from sklearn.metrics import r2_score, mean_squared_error
r2_score( pr_data, df_test.medv),
mean_squared_error( pr_data, df_test.medv)