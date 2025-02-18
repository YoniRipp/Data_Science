import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

data = pd.read_csv('mlr.csv')
print(data.describe())

y = data['GPA']
x1 = data[['SAT','Rand 1,2,3']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())