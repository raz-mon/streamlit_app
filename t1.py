import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probfit import Chi2Regression
from iminuit import Minuit

#   def chi2Reg_func(model, file_name, sheet_name, x_colName, y_colName, dy_colName):

model = lambda x, a, b: a * x + b
file_name = 'test.xlsx'
sheet_name = 'test_sheet'
x_colName = 'x'
y_colName = 'y'
dy_colName = 'dy'


df = pd.read_excel(file_name, sheet_name=sheet_name)
x_data = df[x_colName].values   #[x_start:x_stop]
y_data = df[y_colName].values   #[y_start:y_stop]
dy = df[dy_colName].values      #[y_start:y_stop]

reg = Chi2Regression(model, x_data, y_data, dy)
opt = Minuit(reg)
opt.migrad()

# plot
plt.rc("font", size=16, family="Times New Roman")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlabel("X_data", fontdict={"size": 20, "weight": "bold"})
ax.set_ylabel("y_data", fontdict={"size": 20, "weight": "bold"})
#ax.errorbar(x=X, y=y, yerr=y_err, xerr=X_err, capsize=4, elinewidth=3, fmt='none', ecolor="blue")
ax.errorbar(x=x_data, y=y_data, yerr=dy, capsize=4, elinewidth=3, fmt='none', ecolor="blue")
ax.scatter(x_data, y_data, c='blue', s=30)
# reg.show(minuit=None,args=opt.values,errors=opt.errors)
reg.show()  # show with no arguments once again works





















