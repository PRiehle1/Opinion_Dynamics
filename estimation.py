# Import Packages 
from tqdm import tqdm
import numpy as np
import model
from scipy.optimize import minimize, dual_annealing
import matplotlib.pyplot as plt

import multiprocessing as mp

# Define the class 

class Estimation(object):
    ''' Class for the Estimation of the Social Model'''
    def __init__(self, time_series: np.array, multiprocess : bool) -> None: 
        self.time_series = time_series 
        self.multiprocess = multiprocess

    def logL(self, guess) -> np.array:
        
        """
        The logL function takes a guess for the parameters and returns the log likelihood of that guess.
        The function takes as input:
            - time_series: The times series to be estimated. 
            - nu, alpha0, alpha0: Guesses for the parameters.
        Args:
            guess (_type_): Initialize the parameters of the model

        Returns:
            np.array: The sum of the logarithm of the density function at each point
        """
        # Times Series to be estimated
        time_series = self.time_series
    
        # Parameters to be estimated
        nu, alpha0, alpha1, N = guess

        print("The actual guess is: " + str(guess))

        # The Model
        mod = model.OpinionFormation(N = N, T = 3, nu = nu, alpha0= alpha0 , alpha1= alpha1, deltax= 0.01, deltat= 1/16)
        
        # Initialize the log(function(X, Theta))
        logf = np.zeros(len(time_series))

        if self.multiprocess == True:
            # Time Series to List
            time_series_list = self.time_series.tolist()
            # Multiprocessing 
            pool = mp.Pool(mp.cpu_count())

            # Calculate the PDF for all values in the Time Series
            pdf = list(tqdm(pool.imap(mod.CrankNicolson, time_series_list)))
            pool.close()  
            pdf = np.array(pdf)

            for elem in range(len(pdf)-1):
                for x in range(len(mod.x)):
                    if mod.x[x] == np.around(time_series[elem+1],3):
                        logf[elem] = np.log(pdf[elem,x])
            logL = (-1)* np.sum(logf)
            print("The Log Likelihood is: " + str(logL)) 
        
        else: 
        
            for elem in tqdm(range(len(time_series)-1)):

                # Solve the Fokker Plank Equation: 
                pdf = mod.CrankNicolson(x_0 = time_series[elem])

                # Search for the Value of the PDF at X_k+1
                for x in range(len(mod.x)):
                    if mod.x[x] == np.around(time_series[elem+1],2):
                        logf[elem] = np.log((pdf[x]))
        
            logL = (-1)* np.sum(logf)
            print("The Log Likelihood is: " + str(logL)) 

        return logL
    
    def solver_BFGS(self, initial_guess: list):
        
        # Unpack the inital guess
        nu, alpha0, alpha1, N = initial_guess

        # Minimite the negative Log Likelihood Function
        res = minimize(self.logL, (nu, alpha0 , alpha1, N), method='L-BFGS-B', bounds = [(0.0001, None), (-2, 2), ( 0, None), (2, None)],  callback=None, options={ 'maxiter': 100, 'disp': True})
        print(res)

if __name__ == '__main__':
    import pandas as pd
    import sim
    from sympy import *

    training_data_x = pd.read_excel("zew.xlsx", header=None)
    X_train= training_data_x[1].to_numpy()
    X_train= X_train[~np.isnan(X_train)]
    plt.plot(X_train)
    plt.show()

    simulation = sim.Simulation(N = 21, T = 200, nu = 0.15 , alpha0 = 0.09, alpha1 = 0.99, deltax = 0.02, deltat = 1/16, seed = 150)
    d = simulation.eulermm(-0.59)
    plt.plot(d)
    plt.show()

    est = Estimation(X_train, multiprocess= False)
    #est.solver_BFGS((0.15, 0.09, 0.99, 21))
    res = dual_annealing(est.logL, bounds = [(0.01, 1), (-0.09, 0.3), ( 0.1, 2), (10, 40)])





