import numpy as np
import pandas as pd
import scipy 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 

data = pd.read_csv('slr.csv')

y = data['GPA']
x1 = data['SAT']

plt.scatter(x1, y)
plt.xlabel('SAT',fontsize = 20)
plt.ylabel('GPA',fontsize = 20)

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())
