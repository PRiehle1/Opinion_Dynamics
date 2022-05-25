import model
import plot 
import sim
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import random
import pandas as pd

#######################################################################################################################################
# Calculate The LogLikelihood
#######################################################################################################################################

#simulation = sim.simulateModel(N = 50, T = 200, nu = 3 , alpha0 = 0.00, alpha1 = 1.2, deltax = 1/50, deltat = 0.01,seed= 15)
#d = simulation.eulermm(0)



training_data_x = pd.read_excel("zew.xlsx", header=None)
X_train= training_data_x[1].to_numpy()
X_train= X_train[~np.isnan(X_train)]
plt.plot(X_train)
plt.show()

from scipy.optimize import fmin


def logL(guess, deltax = 0.0025, deltat = 1/16, T = 3, N = 175):
    y = X_train
    nu , alpha0, alpha1,  = guess
    print("The actual guess is: " +str(guess))

    test = model.OpinionFormation(N, T, nu , alpha0, alpha1, deltax, deltat)
    logf = np.zeros(len(y))
    numRun = 0 

    dum = np.around(X_train, 2) 

    for elem in tqdm(range(len(np.around(X_train, 2))-1)):
        
        _,prob_end = test.CrankNicolson(x_0 = dum[elem], check_stability=False, calc_dens= False)

        #prob_end = 1/(1+np.exp(-prob_end))
        
        for i in range(len(test.x)):
            if test.x[i] == dum[elem+1]:
                logf[numRun] = np.log(prob_end[i])
        numRun +=1
    
    print(-np.sum(logf))
    return -np.sum(logf)     

def print_fun(x):
    print("Current value: {}".format(x))

bounds = [(0,5), (0, 0.1), (0, 2)]
res = fmin(logL, (1, 0.1, 0.99), disp = True, retall = True)  #,  method='Nelder-Mead',callback=print_fun,options={'return_all': True, 'adaptive': True})