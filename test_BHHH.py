# Packages 
from matplotlib.cbook import delete_masked_points
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
        #self.guess = guess
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
        
            for elem in (range(len(time_series)-1)):    #TQDM

                # Solve the Fokker Plank Equation: 
                pdf = mod.CrankNicolson(x_0 = time_series[elem])

                # Search for the Value of the PDF at X_k+1
                for x in range(len(mod.x)):
                    if mod.x[x] == np.around(time_series[elem+1],2):
                        logf[elem] = np.log((pdf[x]))
        
            logL = np.sum(logf)
            #print("The Log Likelihood is: " + str(logL)) 

        return logL
    
    def gradient(self, guess_initial, eps: float):

        print("Calculate the Gradient")
        
        # Convert the gues tuple to list
        guess_in = list(guess_initial)
        
        # Initialize the Gradient Column Vector
        g = np.zeros([4,1]) # The Gradient is a column vector

        # Log Likelihood of the guess
        logL = self.logL(guess_in)
        
        guess_r = guess_in.copy()
        #guess_l = guess_in.copy()
        
        for param in range(len(guess_in)):
            guess_r[param] = guess_r[param] + eps 
            #guess_l[param] = guess_l[param] - eps

            g[param] = (self.logL(guess_r) - logL)/(eps)
            
            guess_r = guess_in.copy()
            #guess_l = guess_in.copy()
        
        return g
    
    def cov_matrix(self, g):
        
        r_t = (len(self.time_series)**2) * np.dot(g,g.T)
        
        return r_t
        
    def bhhh(self, initial_guess, tolerance_level, max_iter):
        
        ##########################
        ### Initial Values 
        ###########################

        # Calculate the initial Gradient
        g_in = self.gradient(initial_guess, eps = 0.000001)
        
        # Calculate the initial Variance Covariance Matrix
        r_t_in = self.cov_matrix(g_in)
        
        # Check if the Variance Covariance Matrix is singular
        if (np.linalg.det(r_t_in)):
            print("Covariance Matrix is  not singular ")
            pass
        else: 
            print("Covariance Matrix is singular ")
            r_t_in = np.array([[10000, 0, 0, 0], [0, 10000, 0, 0], [0, 0, 10000, 0], [0, 0, 0, 1]])
        
        # Calculate the initial direction vector
        direc_in = np.dot(np.linalg.inv(r_t_in),g_in).reshape(4,)
        
        lamb =  1
        delta = 0.02

        # Initial Beta         
        beta_in= np.array(initial_guess)

        for it in range(max_iter):
            print("Number of Iterations:" + str(it))
            if it == 0:
                beta = beta_in + lamb*direc_in
                print("The actual Estimate is:   " +str(beta))
            else:
                
                # Calculate the Gradient
                g = self.gradient(tuple(beta), eps = 0.000001)
                
                # Calculate the initial Variance Covariance Matrix
                r_t = self.cov_matrix(g)
                
                # Check if the Variance Covariance Matrix is singular
                if (np.linalg.det(r_t)):
                    print("Covariance Matrix is  not singular ")
                    pass
                else: 
                    print("Covariance Matrix is singular ")
                    r_t = np.array([[10000, 0, 0, 0], [0, 100000, 0, 0], [0, 0, 10000, 0], [0, 0, 0, 1]])
                
                # Calculate the initial direction vector
                direc = np.dot(np.linalg.inv(r_t),g).reshape(4,)

                # Check for convergence
                dum = np.zeros(len(direc))
                for elem in range(len(direc)):
                    dum[elem] = np.abs(direc[elem])/ max((1, np.abs(beta[elem])))
        
                if max(dum) < tolerance_level:
                    print(" Final Estimate foud" + str(beta))
                    break

                # Calculate the Lambda
        
                def calcnu(lamb):
                    logL =  self.logL(list(beta))
                    nu = float((self.logL(list((beta + lamb * direc))) - logL)) / (lamb * float(np.dot(direc, g)))
                    return nu
                
                if calcnu(lamb) >= delta:
                    lamb = 1
                    print("Lambda is:   " + str(lamb))
                else:
                    while delta >= calcnu(lamb) or calcnu(lamb) >=1-delta:
                        lamb *=0.8
                        if lamb <= 0.001:
                            pass
                    print("Lambda is:   " + str(lamb))

                beta = beta + lamb*direc
                print("The actual Estimate is:   " +str(beta))
        
            
if __name__ == '__main__':
    import pandas as pd
    import sim
    from sympy import *

    training_data_x = pd.read_excel("zew.xlsx", header=None)
    X_train= training_data_x[1].to_numpy()
    X_train= X_train[~np.isnan(X_train)]
    
    est = Estimation(X_train, multiprocess= False)
    est.bhhh((0.7,0.3,0.7,35), tolerance_level= 0.000000001, max_iter = 10000)
        
        