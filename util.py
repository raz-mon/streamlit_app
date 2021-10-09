import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression_with_app_changes import Chi2Reg, EffVarChi2Reg
from probfit import Chi2Regression
from iminuit import Minuit
from PIL import Image

def chi2Reg_func(model, file_name, sheet_name, x_colName, y_colName, dy_colName, title, x_axis_name, y_axis_name):
    df = pd.read_csv(file_name, sheet_name=sheet_name)
    x_data = df[x_colName].values   #[x_start:x_stop]
    y_data = df[y_colName].values   #[y_start:y_stop]
    dy = df[dy_colName].values      #[y_start:y_stop]

    reg = Chi2Reg(model, x_data, y_data, dy)
    opt = Minuit(reg)
    opt.migrad()
    reg.show(opt, title=title, x_title=x_axis_name, y_title=y_axis_name)

    # plot
    # Better to put all this in a separate function, easier to the eye..

    """
    plt.rc("font", size=16, family="Times New Roman")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel(x_axis_name, fontdict={"size": 20, "weight": "bold"})
    ax.set_ylabel(y_axis_name, fontdict={"size": 20, "weight": "bold"})
    #ax.errorbar(x=X, y=y, yerr=y_err, xerr=X_err, capsize=4, elinewidth=3, fmt='none', ecolor="blue")
    ax.errorbar(x=x_data, y=y_data, yerr=dy, capsize=4, elinewidth=3, fmt='none', ecolor="blue")
    ax.scatter(x_data, y_data, c='blue', s=30)
    # reg.show(minuit=None,args=opt.values,errors=opt.errors)
    reg.show()  # show with no arguments once again works
    plt.title(title)
    plt.savefig(title, dpi=300, bbox_inches='tight')
    #plt.show()
    im = Image.open(title + '.png')
    st.image(im)

    """

    # Now must make an option to DOWNLOAD the image (not hard.. see documentation or 'testing_shit' file).



def effVarChi2Reg_func(model, file_name, sheet_name, x_colName, y_colName, dy_colName, dx_colName, title, x_axis_name, y_axis_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    x_data = df[x_colName].values  # [x_start:x_stop]
    y_data = df[y_colName].values  # [y_start:y_stop]
    dy = df[dy_colName].values  # [y_start:y_stop]
    dx = df[dx_colName].values  # [x_start:x_stop]

    reg = EffVarChi2Reg(model, x_data, y_data, dx, dy)
    opt = Minuit(reg)
    opt.migrad()
    reg.show(opt, title=title, x_title=x_axis_name, y_title=y_axis_name)








































