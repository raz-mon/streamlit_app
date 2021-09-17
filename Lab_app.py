import streamlit as st
import numpy as np
from util import chi2Reg_func, effVarChi2Reg_func

st.set_page_config(page_title='Physics Lab app!')
#st.set_page_config(icon = --) Need to pass an image object (using PIL).
# Can also write: st.title('Pyhsics Lab application!')

st.write("""
# Welcome to the Physics Lab App!
## Here we will try to make your life **easier** with a nice application, that will do the ***hard work*** for you.
""")

# Get the name of the excel file of the user:
file_name = st.text_input('please insert the name of your excel file: ')
# Note: Concatenate an '.xlsx' to the end of the file name inserted by the user, so he will have to insert only the name of the file itself.


# Get the name of the x column, y column, dy and dx if exists in the data file (excel)
# If the dx is left blank (None) --> Use Chi2Reg of probfit, else --> Use effVarChi2Reg of phys_lab.
sheet_name = st.text_input('sheet name: ')
x_colName = st.text_input('x data column name:')
y_colName = st.text_input('y data column name:')
dy_colName = st.text_input('dy data column name:')
dx_colName = st.text_input('dx data column name:')

# Note1: Can (should) ask here for some initial values and limits for the parameters.
# Note2: Can display the data here, or ask the user if he wishes to see the data to make sure it's good.

title = st.text_input('title:')
x_axis_name = st.text_input('x axis title:')
y_axis_name = st.text_input('y axis title:')

# Ask the user to choose a model from the printed options:
model_str = st.radio('please choose a model from the following:', ('None', 'Linear', 'Power', 'Exponential', 'Cosine', 'Poly_2', 'Poly_3', 'normalised Gaussian', 'Gaussian', 'Poisson', 'Constant'))
models_str = ['Linear', 'Power', 'Exponential', 'Cosine', 'Poly_2', 'Poly_3', 'normalised Gaussian',
          'Gaussian', 'Poisson', 'Constant']

lin_fun                = lambda x, a, b: (a * x) + b
power_fun              = lambda x,a,b: a * (x ** (b))
exp_fun                = lambda x, a, b: a * (np.exp(b * x))
cos_fun                = lambda x, a, b: a * (np.cos(b* x))
#cos2_fun               = lambda x,a,b: a * (np.cos(b*x))**2
poly2_fun              = lambda x, a, b, c: a * (x ** 2) + b * x + c
poly3_fun              = lambda x, a, b, c, d: a * (x ** 3) + b * (x ** 2) + c * x + d
normalised_gauss_fun   = lambda x,a,b: 1/b*(np.sqrt(2*np.pi))*np.exp(-0.5*((x-a)/(b))**2)
gauss_fun              = lambda x,a,b,C: C*np.exp(-0.5*((x-a)**2/b**2))
#normalised_poisson_fun = lambda x,a: (( a ** x ) * np.exp((-1)*a)) /(np.math.factorial((x)))
poisson_fun            = lambda x,a,b: b *(( a ** x ) * np.exp((-1)*a)) /(np.math.factorial((x)))
const_fun              = lambda x,b: x*0+b

models = [lin_fun, power_fun, exp_fun, cos_fun, poly2_fun,
          poly3_fun, normalised_gauss_fun, gauss_fun, poisson_fun, const_fun]

#while model_str == '':
#    time.sleep(2)

if not model_str == 'None':
    ind = models_str.index(model_str)

    model = models[ind]

    st.write('model chosen is: ', models_str[ind], 'model.')


    # Check weather the user inserted a dx column name, and use appropriate function.
    if dx_colName == '':
        st.write('## Chi2Regression algorithm in action...')
        chi2Reg_func(model, file_name, sheet_name, x_colName, y_colName, dy_colName, title, x_axis_name, y_axis_name)
    else:
        st.write('## EffVarChi2Reg algorithm in action...')
        effVarChi2Reg_func(model, file_name, sheet_name, x_colName, y_colName, dy_colName, dx_colName, title, x_axis_name, y_axis_name)















