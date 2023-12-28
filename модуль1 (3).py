
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scp
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import datasets # scikit-learn

data = datasets.fetch_california_housing()

print(data.DESCR)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

df.describe() 
lm = smf.ols("MedHouseVal~MedInc", df).fit()
lm.summary()
plt.clf()
sns.regplot(data=df, x="MedInc", y="MedHouseVal", scatter = True)
plt.show()
plt.clf()
scp.probplot(lm.resid, dist="norm", plot=plt)
plt.show()
plt.clf()
# Compute the correlation matrix
corr = df.drop('MedHouseVal', axis=1).corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
import statsmodels.formula.api as smf

lm = smf.ols("MedHouseVal~MedInc+Population+AveOccup+Latitude", df).fit()

print(lm.summary() )
import pandas as pd
df = pd.read_csv('BostonHousing.csv')
plt.clf()
# Compute the correlation matrix
corr = df.drop('medv', axis=1).corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
import statsmodels.formula.api as smf

lm = smf.ols("medv~zn+rm+age+tax", df).fit()

print(lm.summary() )
df.shape
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

df_train.shape, df_test.shape
from sklearn.metrics import r2_score, mean_squared_error
lm = smf.ols("medv~zn+rm+age+tax", df_train).fit()
pr_data = lm.get_prediction(df_test)
r2_score( pr_data.predicted, df_test.medv),mean_squared_error( pr_data.predicted, df_test.medv)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
x = df_train[['zn', 'rm', 'age', 'tax']]
y= df_train['medv']
lm = LinearRegression()
scores = cross_val_score(lm, x, y, scoring='r2', cv=5)

np.mean(scores)
X = [df_train[['zn', 'rm', 'age', 'nox']], df_train[[ 'zn', 'rm', 'dis', 'nox']], df_train[[ 'zn', 'crim', 'dis', 'nox']], df_train[[ 'indus', 'crim', 'dis', 'nox']]]
lm = LinearRegression()
for i, i_df in enumerate(X):
  scores = cross_val_score(lm, i_df, y, scoring='r2', cv=5)
  print(f'data index = {i}, mean = {np.mean(scores)}')
y= df_train['medv']
reg = lm.fit(X[1], y)

pr_data = reg.predict(df_test[[  'zn', 'rm', 'dis', 'nox']])

from sklearn.metrics import r2_score, mean_squared_error

r2_score( pr_data, df_test.medv),mean_squared_error( pr_data, df_test.medv)
