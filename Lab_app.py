import streamlit as st


"""
This will be the flow of things (in general):\n
1. Get the path of the excel file from the user.\n
2. Get the names of the sheet and columns of the excel file.\n
3. User inserts his desires: Model, parameter limits, graph title, x_label, y_label.\n
4. Do the calculations 'behind the screen':\n
    - Generate the minimization function (func to minimize) using Chi2Regression (of probfit) or effVarChi2Reg (of BGU_physics_Lab).\n
    - Minimize using Minuit (Make a Minuit Object, and minimize using ob.migrad() method).\n
    - Plot the data and the minimized function, print the parameter-values, the chi^2 val.\n
5. Make it possible to save the figure in current work-environment (of the user).\n
6. Make an option of exporting the code generated 'behind the screen'.\n

Seems like a good start to me.
"""



#st.set_page_config(page_title='Physics Lab app!')
# Can also write: st.title('Pyhsics Lab application!')

st.write("""
# Pyhsics Lab application!
## Here we will try to make your life **easier** with a nice application, that will do the ***hard work*** for you.
""")


































