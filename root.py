# Root Finder 
import numpy as np   
from scipy.optimize import fsolve   

def func(x):
    return np.cosh(x - np.sqrt(1.19*(1.19-1))) * np.cosh(x - np.sqrt(1.19*(1.19-1))) - 1.19

root = fsolve(func, 0)

print(root)



