'''
Weighted Regression + Graph: Original code by BGU Lab Staff, modified by Ze'ev Girsh
Last version update Dec2020

INSTRUCTIONS:

1) Import Data:
Under ## Importing Data:
fill in the path location of the excel data file under file_location.
fill in name of excel sheet under sheet_name.
fill in column names as they appear on the excel sheet for x,y and errors in x and y under x,y_column_name

2) Choosing which function to fit to the data:
Under ## Function types for fitting:
Choose a function from the listed functions fill it after func_type=

Note: If fitting a to a constant function, find efopt = Minuit(efvtest, a=a_0, b=b_0) in both locations, and delete a=a_0

OPTIONAL 1: It is possible to choose a range for the starting values of the minimzation parameters a,b. Look at line: parameter_limits.
OPTIONAL 2: It is possible to put a limit on the a,b parameters. look for line: efopt = Minuit(efvtest, a=a_0, b=b_0) and add what's greyed out.

3) Graph Design:
OPTIONAL: change the parameter names in the fit function instead of x,y,a,b to whatever you want, under x.y_parameter for x,y and parameter for abcd
Under ## Graph Design:
fill the graph title
fill in x,y axis labels
choose the on-screen location for the text box
fill in the file name for the exported image

And you're done!

* Currently having an issue fitting to Poisson function *

'''


## Package imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from iminuit import Minuit, describe
from iminuit.util import make_func_code
from matplotlib.offsetbox import AnchoredText
warnings.simplefilter("ignore")

## Importing Data

file_location = "file data" # Location of the excel with the data
sheet_name = ""

# Defining X,Y,dx,dy axes from excel:

x_column_name = ""      # x column name in Excel
y_column_name = ""      # y column name in Excel
x_err_column_name = ""       # x errors column name in Excel
y_err_column_name = ""       # y errors column name in Excel

# If there's a need to take only specific indices from the data column, change here:

x_start = 0
x_stop = 100
y_start = 0
y_stop = 100

# Reading the data using pandas

df = pd.read_excel(file_location, sheet_name=sheet_name)
x_data = df[x_column_name].values[x_start:x_stop]
y_data = df[y_column_name].values[y_start:y_stop]
x_err = df[x_err_column_name].values[x_start:x_stop]
y_err = df[y_err_column_name].values[y_start:y_stop]

# Sanity check to see that data is okay:

print("Data received is:")
print("x:", x_data)
print("y:", y_data)
print("x errors:", x_err)
print("y errors:", y_err)

## Function types for fitting

# Function types

lin_fun                = lambda x, a, b: (a * x) + b
power_fun              = lambda x,a,b: a * (x ** (b))
exp_fun                = lambda x, a, b: a * (np.exp(b * x))
cos_fun                = lambda x, a, b: a * (np.cos(b* x))
cos2_fun               = lambda x,a,b: a * (np.cos(b*x))**2
poly2_fun              = lambda x, a, b, c: a * (x ** 2) + b * x + c
poly3_fun              = lambda x, a, b, c, d: a * (x ** 3) + b * (x ** 2) + c * x + d
normalised_gauss_fun   = lambda x,a,b: 1/b*(np.sqrt(2*np.pi))*np.exp(-0.5*((x-a)/(b))**2)
gauss_fun              = lambda x,a,b,C: C*np.exp(-0.5*((x-a)**2/b**2))
normalised_poisson_fun = lambda x,a: (( a ** x ) * np.exp((-1)*a)) /(np.math.factorial((x)))
poisson_fun            = lambda x,a,b: b *(( a ** x ) * np.exp((-1)*a)) /(np.math.factorial((x)))
const_fun              = lambda x,b: x*0+b

func_type = lin_fun # Choose which function you want to the graph to fit to
                    # Choose between the functions above: lin_fun, power_fun, exp_fun, etc.

##  Graph Design

main_title = "Graph Title"
x_label_title = "X Data $[units]$" # use LaTex syntax inside $[]$
y_label_title = "Y Data $[units]$"
box_location = 2 # This is the on-screen location of the data box that is presented on top of the graph.
                 # Change number for corners: top right=1 ; top left=2 ;bottom left=3; bottom right=4

x_parameter = "x"              # IF NEEDED: change the name of the parameters. for example instead of showing y=ax+b it will show y=at+b
y_parameter = "y(%s)"%(x_parameter)  # IF NEEDED: change the name of the parameters. for example instead of showing y=ax+b it will show f(x)=ax+b
parameter =["a","b","c","d"]             # IF NEEDED: change the name of the parameters. for example instead of showing ax+b it will show mx+n
graph_photo_name = "Graph Image"     # Name of the exported graph photo file

##

# Function types in LaTex text for description at the graph box

lin_fun_text                = "%s\cdot %s +%s" % (parameter[0],x_parameter,parameter[1])
power_fun_text              = "%s \cdot %s^{%s}" % (parameter[0],x_parameter,parameter[1])
exp_fun_text                = "%s \cdot e^{%s\cdot %s}" % (parameter[0],x_parameter,parameter[1])
cos_fun_text                = "%s\cdot \cos(%s\cdot %s)" % (parameter[0],x_parameter,parameter[1])
cos2_fun_text               = "%s\cdot \cos(%s\cdot %s)^2" % (parameter[0],x_parameter,parameter[1])
poly2_fun_text              = "%s\cdot %s^2 + %s\cdot %s + %s" % (parameter[0],x_parameter,parameter[1],x_parameter,parameter[2])
poly3_fun_text              = "%s\cdot %s^3 + %s\cdot %s^2 + %s\cdot %s +%s" % (parameter[0],x_parameter,parameter[1],x_parameter,parameter[2],x_parameter,parameter[3])
normalised_gauss_fun_text   = "\dfrac{1}{%s\sqrt{2\pi}} e^{-\dfrac{(%s-%s)^2}{2%s^2}}" %(parameter[1],x_parameter,parameter[0],parameter[1])
gauss_fun_text              = "C\cdot e^{-\dfrac{(%s-%s)^2}{2%s^2}}"%(x_parameter,parameter[0],parameter[1])
normalised_poisson_fun_text = "\dfrac{\lambda^%s \cdot e^{-\lambda}}{%s!}"%(x_parameter,x_parameter)
poisson_fun_text            = "C\cdot \dfrac{\lambda^%s \cdot e^{-\lambda}}{%s!}"%(x_parameter,x_parameter)
const_fun_text              = "%s" %(x_parameter)

num_of_parameters = [2,2,2,2,2,3,4,2,3,1,2,1] # This will take the right number of parameters depending the chosen function and will allow proper text replacements in the text box.

# Setting the Latex Description of the chosen function in the graph text box:

func_type_text_list = [lin_fun_text, power_fun_text, exp_fun_text,cos_fun_text,cos2_fun_text,
                       poly2_fun_text,poly3_fun_text,
                       normalised_gauss_fun_text, gauss_fun_text, normalised_poisson_fun,poisson_fun_text,const_fun_text]

func_type_list = [lin_fun, power_fun, exp_fun, cos_fun, cos2_fun,
                  poly2_fun,poly3_fun,
                  normalised_gauss_fun, gauss_fun, normalised_poisson_fun, poisson_fun,const_fun]

func_type_text = ""
for i in range(len(func_type_text_list)):
    if func_type == func_type_list[i]:
        func_type_text = func_type_text_list[i]
        parameter = parameter[:num_of_parameters[i]]

## Defining a,b arrays to run the minimization against
parameter_limits = [-1,1,-1,1] # [a_initial,a_final,b_initial,b_final]
step_size = 1 #if runtime is too slow choose a bigger step size, or change a,b limits

a_0_array = np.arange(parameter_limits[0],parameter_limits[1],step_size)
b_0_array = np.arange(parameter_limits[2],parameter_limits[3],step_size)
chi2_ndof_matrix = np.zeros([len(a_0_array),len(b_0_array)])

## Loop going over all a,b parameters, building a chi2/ndof matrix from all initial parameters.
for i in range(len(a_0_array)):
    print("Minimization process", np.round(i / len(a_0_array) * 100), '%')
    for j in range(len(b_0_array)):
        a_0 = a_0_array[i]
        b_0 = b_0_array[j]

        # This part is the minimization:
        class EffVarChi2Reg:  # This class is like Chi2Regression but takes into account dx (errors on both axis, not only y).
            # This part defines the variables the class will use
            def __init__(self, model, x, y, dx, dy):
                self.model = model  # model predicts y value for given x value
                self.x = np.array(x)  # the x values
                self.y = np.array(y)  # the y values
                self.dx = np.array(dx)  # the x-axis uncertainties
                self.dy = np.array(dy)  # the y-axis uncertainties
                self.func_code = make_func_code(describe(self.model)[1:])
                self.h = (x[-1] - x[0]) / 10000  # this is the step size for the numerical calculation of the df/dx = last value in x (x[-1]) - first value in x (x[0])/10000

            # This part defines the calculations when the function is called
            def __call__(self, *par):  # par are a variable number of model parameters
                self.ym = self.model(self.x, *par)
                df = (self.model(self.x + self.h,
                                 *par) - self.ym) / self.h  # the derivative df/dx at point x is taken as [f(x+h)-f(x)]/h
                chi2 = sum(((self.y - self.ym) ** 2) / (self.dy ** 2 + ( df * self.dx ) ** 2))  # chi2 is now Sum of: (f(x)-y)^2/(uncert_y^2+(df/dx*uncert_x)^2)
                return chi2

            # this part defines a function called "show" which will make a nice plot when invoked
            def show(self, optimizer, x_title="X", y_title="Y", goodness_loc=2):
                self.par = optimizer.parameters
                self.fit_arg = optimizer.fitarg
                self.chi2 = optimizer.fval
                self.ndof = len(self.x) - len(self.par)
                self.chi_ndof = self.chi2 / self.ndof
                self.par_values = []
                self.par_error = []
                text = "$ Fitted \  to \ {}={} $\n".format(y_parameter,func_type_text)
                for _ in (self.par):
                    self.par_values.append(self.fit_arg[_])
                    self.par_error.append(self.fit_arg["error_" + _])
                    text += "$\ \ \ {} $ = %0.4f $\pm$ %0.4f \n".format(_) % (self.fit_arg[_], self.fit_arg["error_" + _])
                for i in range(len(parameter)): # This is replacing the strings from self.par in the chosen parameters
                    old = self.par[i]
                    new = parameter[i]
                    text = new.join(text.rsplit(old, 1))
                text = text + "$\dfrac{{\chi}^2}{N_{dof}} = %0.4f(%0.4f/%d)$\n" % (self.chi_ndof, self.chi2, self.ndof)
                self.func_x = np.linspace(self.x[0], self.x[-1], 10000)  # 10000 linearly spaced numbers
                self.y_fit = self.model(self.func_x, *self.par_values)
                plt.rc("font", size=16, family="Helvetica")
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.plot(self.func_x, self.y_fit)  # plot the function over 10k points covering the x axis
                ax.scatter(self.x, self.y, c="black")
                ax.errorbar(self.x, self.y, self.dy, self.dx, fmt='none', ecolor='red', capsize=3)
                ax.set_xlabel(x_title, fontdict={"size": 21})
                ax.set_ylabel(y_title, fontdict={"size": 21})
                anchored_text = AnchoredText(text, loc=box_location)
                ax.add_artist(anchored_text)
                plt.grid(True)


        efvtest = EffVarChi2Reg(func_type, x_data, y_data, x_err, y_err)
        efopt = Minuit(efvtest, a=a_0, b=b_0)
        efopt.migrad()
        number_param = len(list(efopt.args))
        ndof = len(x_data) - number_param
        chi2_ndof = efopt.fval / ndof
        chi2_ndof_matrix[i][j] = chi2_ndof # fills in the chi2/ndof matrix

## Cleaning the chi2/ndof matrix, picking it's minimum value and getting the corresponding a and b

chi2_matrix = np.nan_to_num(chi2_ndof_matrix,1000000000000) # This part cleans the chi2/ndof matrix from Nans and Infs by replacing them with an irrelevant value
#print("Chi 2 matrix using different a and b:", chi2_matrix)
# This part looks for the minimal chi2/ndof and calls out its' index in the matrix
#print("Minimal value of chi2/ndof in the matrix is:",chi2_matrix.min())
minimal_chi2_index = np.array(np.where(chi2_matrix == chi2_matrix.min())).flatten()
#print("index of minimal value is:", minimal_chi2_index)

# Defining the best initial a and b values, if more than 1 minimum, choosing the first pair of a and b.
a_0 = a_0_array[minimal_chi2_index[0]]
b_0 = b_0_array[minimal_chi2_index[int(len(minimal_chi2_index)/2)]]

# final minimization with the best a and b + graph output

efvtest = EffVarChi2Reg(func_type, x_data, y_data ,x_err, y_err)
efopt = Minuit(efvtest, a=a_0, b=b_0) #limit_a = (lower lim,upper lim), limit_b = (lower lim,upper lim))
efopt.migrad()
number_param = len(list(efopt.args))
ndof = len(x_data) - number_param
chi2_ndof = efopt.fval / ndof
print("best Chi2/ndof found: %0.4f(%0.4f/%d)" % (chi2_ndof, efopt.fval, ndof))
efvtest.show(efopt, goodness_loc=4, x_title=x_label_title, y_title=y_label_title)
plt.title(main_title)
plt.savefig(graph_photo_name, dpi=300, bbox_inches='tight')
plt.show()

## Defining parameters using the data generated from the graph

a = efopt.np_values()[0]
a_err = efopt.np_errors()[0]
b = efopt.np_values()[1]
b_err = efopt.np_errors()[1]

print("Parameters found:","a =",a,"±",a_err,"b =",b,"±",b_err)

## Room for further calculations here
