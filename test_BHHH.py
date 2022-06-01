# Packages 
import numpy as np
import model
from tqdm import tqdm 
import multiprocessing as mp


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
        self.guess = guess
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
    
    def gradient(self, guess_initial, eps: float):
        
        # Convert the gues tuple to list
        guess_in = list(guess_initial)
        
        # Initialize the Gradient Column Vector
        g = np.zeros([4,1]) # The Gradient is a column vector

        # Log Likelihood of the guess
        logL = self.logL(guess_in)
        
        guess = guess_in.copy()
        
        for param in range(len(guess_in)):
            guess[param] = guess[param] + eps 
            g[param] = (self.logL(guess) - logL)/ eps
            guess = guess_in.copy()
        
        return g
    
    def cov_matrix(self, g):
        
        r_t = len(self.time_series)**2 * np.dot(g,g.T)
        
        return r_t
        
    def bhhh(self, initial_guess, tolerance_level, max_iter):
              
        # Calculate the Gradient
        g = self.gradient(initial_guess, eps = 0.00001)
        
        # Calculate the Variance Covariance Matrix
        r_t = self.cov_matrix(g)
        
        # Calculate the direction vector
        direc = np.dot(np.linalg.inv(r_t),g)
        
        # Check for convergence
        dum = np.zeros(len(direc))
        for elem in range(len(direc)):
            dum[elem] = np.abs(direc[elem])/ max((1, np.abs(initial_guess[elem])))
        
        if max(dum) < tolerance_level:
            print(" Final Estimate")
        
        lamb =  1
        # Calculate the Lambda
        nu = (self.logL((initial_guess + lamb * direc)) - self.logL(initial_guess))/lamb * direc.T * g 
        
        print("Hello Wolrd")
        
            
        

if __name__ == '__main__':
    import pandas as pd
    import sim
    from sympy import *

    training_data_x = pd.read_excel("zew.xlsx", header=None)
    X_train= training_data_x[1].to_numpy()
    X_train= X_train[~np.isnan(X_train)]
    
    est = Estimation(X_train, multiprocess= False)
    est.bhhh((1,0,1.2,20), tolerance_level= 0.00000001, max_iter = 10000)
        
        