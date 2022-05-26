# Import Packages 
from tqdm import tqdm
import numpy as np
import model
from scipy.optimize import fmin

from multiprocessing import Pool

# Define the class 

class Estimation(object):
    ''' Class for the Estimation of the Social Model'''
    def __init__(self, time_series: np.array) -> None: 
        self.time_series = time_series 
        self.model = model

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
        # Multiprocessing 
        my_pool = Pool(6)

        # Times Series to be estimated
        time_series = self.time_series

        # Parameters to be estimated
        nu, alpha0, alpha1 = guess

        print("The actual guess is: " + str(guess))

        # The Model
        mod = model.OpinionFormation(N = 175, T = 10, nu = nu, alpha0= alpha0 , alpha1= alpha1, deltax= 0.001, deltat= 1/16)

        # Initialize the log(function(X, Theta))
        logf = np.zeros(len(time_series))

        # The Loop
        for elem in tqdm(range(len(time_series)-1)):

            # Solve the Fokker Plank Equation: 
            _,pdf = mod.CrankNicolson(x_0 = time_series[elem], check_stability= False, calc_dens= False)

            # Search for the Value of the PDF at X_k+1
            for x in range(len(mod.x)):
                if mod.x[x] == np.around(time_series[elem+1],3):
                    logf[elem] = (-1)* np.log(pdf[x])
        
        logL = np.sum(logf)
        print("The Log Likelihodd is: " + str(logL)) 

        return logL
    
    def solver(self):
        pass






if __name__ == '__main__':
    import pandas as pd

    training_data_x = pd.read_excel("zew.xlsx", header=None)
    X_train= training_data_x[1].to_numpy()
    X_train= X_train[~np.isnan(X_train)]

    est = Estimation(X_train)
    res = fmin(est.logL, (1, 0.00, 1.19), disp = True, retall = True)
