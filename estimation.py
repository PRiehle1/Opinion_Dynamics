# Import Packages 
from tqdm import tqdm
import time
import numpy as np
from errors import UncompleteLikelihoodError
from model import OpinionFormation
from scipy.optimize import minimize

import multiprocessing as mp


# The class Estimation is a class that contains the functions that are used to estimate the parameters
# of the model.
class Estimation():
    
    def __init__(self, time_series: np.array,multiprocess : bool, model_type: int, y = 0, x_l = 0) -> None: 
        self.time_series = time_series 
        self.multiprocess = multiprocess
        self.model_type = model_type
        self.y = y
        self.x_l = x_l
        
    def logL(self, guess:tuple) -> np.array:
        """
        The logL function takes a guess for the parameters and returns the log likelihood of that guess.
        The function takes as input:
            - time_series: The times series to be estimated. 
            - nu, alpha0, alpha0: Guesses for the parameters.
        Args:
            guess (tuple): Initialize the parameters of the model

        Returns:
            np.array: The sum of the logarithm of the density function at each point
        """
        # Times Series to be estimated
        time_series = self.time_series
        y = self.y
        x_l = self.x_l
    
        # Parameters to be estimated
        if self.model_type == 0:
            nu_guess, alpha0_guess, alpha1_guess = guess
        elif self.model_type == 1: 
            nu, alpha0, alpha1, N_guess = guess
        elif self.model_type == 2: 
            nu, alpha0, alpha1, N, alpha2 = guess
        elif self.model_type == 3: 
            nu, alpha0, alpha1, N, alpha2, alpha3 = guess
        elif self.model_type == 4: 
            nu, alpha0, alpha1, N, alpha3 = guess
        elif self.model_type == 5: 
            nu, alpha0, alpha1, alpha3 = guess
        elif self.model_type == 6:
            nu, alpha0, alpha1, alpha2, alpha3 = guess

        # The Model
        if self.model_type == 0:
            mod = OpinionFormation(N = 175, T = 1 , nu = nu_guess, alpha0= alpha0_guess , alpha1= alpha1_guess, alpha2 = None, alpha3 = None, deltax = 0.01, deltat = 1/16, model_type= self.model_type)
        elif self.model_type == 1: 
            mod = OpinionFormation(N = N_guess, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 2: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 3: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 4: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 5:
            mod = OpinionFormation(N = 20, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)            
        elif self.model_type == 6: 
            mod = OpinionFormation(N = 20, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type) 
        
        # Initialize the log(function(X, Theta))
        logf = []
        start = time.time()
        for elem in range(len(time_series)-1):
            #Solve the Fokker Plank Equation: 
            if self.model_type == 0 or self.model_type == 1:
                pdf = mod.CrankNicolson(x_0 = time_series[elem])
            elif self.model_type == 2: 
                pdf = mod.CrankNicolson(x_0 = time_series[elem], y = y[elem])
            elif self.model_type == 3 or self.model_type == 6: 
                pdf = mod.CrankNicolson(x_0 = time_series[elem], y = y[elem], x_l = x_l[elem])
            elif self.model_type == 4 or self.model_type == 5: 
                pdf = mod.CrankNicolson(x_0 = time_series[elem], x_l = x_l[elem])
            
            # Search for the Value of the PDF at X_k+1
            for x in range(len(mod.x)):
                if np.around(mod.x[x], decimals= 2) == np.round(time_series[elem+1],2):
                    if pdf[x] <= 0: 
                        pdf[x] = 0.0000000001
                        logf.append(np.log((pdf[x])))
                    else:
                        if pdf[x] == 0:
                            print("PDF at x is zero")
                        logf.append(np.log((pdf[x])))
    
        logL = np.sum(logf)

        print("The Log Likelihood is: " + str(logL) + "and" + "The Minimization_Guess was: " + str(guess)) 
        end = time.time()
        dum = end - start
        print("Time past for one caclulaion of the likelihood:  " + str(dum)) 
        return logL
    
    def neglogL(self, guess:tuple) -> np.array:
        """
        The neglogL function returns the negative log likelihood of a given guess. 
        The function takes in a tuple of parameters and returns an array. The array is the negative log likelihood for each value in the parameter space.

        Args:
            guess (tuple): Pass the Initial Guess that are being estimated

        Returns:
            np.array: The negative log likelihood of the data given a guess for the parameter values
        """     

        nlogL = (-1) * self.logL(guess= guess)
        return nlogL 

    def outer_product_gradient(self, guess: tuple, eps: float) -> np.array:
        """
        The function calculates the outer product of the gradient of the log likelihood function for a
        given set of parameters. 
        
        The function takes as input the guess of the parameters and the epsilon value. 
        
        The function returns the outer product of the gradient of the log likelihood function for a
        given set of parameters. 
        
        :param guess: tuple of the parameters to be estimated
        :type guess: tuple
        :param eps: float
        :type eps: float
        :return: The outer product of the gradient.
        """

        # First Step: Caluclation of the log likelihood at xj for given theta 

        # Preface
        time_series = self.time_series
        y = self.y
        x_l = self.x_l
        product = []
        # Parameters to be estimated
        if self.model_type == 0:
            nu_guess, alpha0_guess, alpha1_guess = guess
        elif self.model_type == 1: 
            nu, alpha0, alpha1, N = guess
        elif self.model_type == 2: 
            nu, alpha0, alpha1, N, alpha2 = guess
        elif self.model_type == 3: 
            nu, alpha0, alpha1, N, alpha2, alpha3 = guess
        elif self.model_type == 4: 
            nu, alpha0, alpha1, N, alpha3 = guess
        elif self.model_type == 5: 
            nu, alpha0, alpha1, alpha3 = guess
        elif self.model_type == 6:
            nu, alpha0, alpha1, alpha2, alpha3 = guess


        # The Model
        if self.model_type == 0:
            mod = OpinionFormation(N = 175, T = 1 , nu = nu_guess, alpha0= alpha0_guess , alpha1= alpha1_guess, alpha2 = None, alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= self.model_type)
        elif self.model_type == 1: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= self.model_type)
        elif self.model_type == 2: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 3: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 4: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)        
        elif self.model_type == 5:
            mod = OpinionFormation(N = 20, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)        
        elif self.model_type == 6: 
            mod = OpinionFormation(N = 20, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type) 

        # Initialize the log(function(X, Theta))
        logf = []
        ######################################################################################################################################
        # Convert the gues tuple to list
        guess_in = list(guess)
        
        # Initialize the Gradient Column Vector
        g = np.zeros([len(guess_in),1]) # The Gradient is a column vector

        guess_r = guess_in.copy()
        
        for elem in range(len(time_series)-1):

            for param in range(len(guess_in)):
                guess_r[param] = guess_r[param] + eps 
                
                # Calculate logf
                #Solve the Fokker Plank Equation: 
                if self.model_type == 0 or self.model_type == 1:
                    pdf = mod.CrankNicolson(x_0 = time_series[elem])
                elif self.model_type == 2: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], y = y[elem])
                elif self.model_type == 3 or self.model_type ==6: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], y = y[elem], x_l = x_l[elem])
                elif self.model_type == 4 or self.model_type == 5: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], x_l = x_l[elem])

                # Search for the Value of the PDF at X_k+1
                for x in range(len(mod.x)):
                    if np.around(mod.x[x], decimals= 2) == np.around(time_series[elem+1],2):
                        if pdf[x] <= 0: 
                            pdf[x] = 0.0000000001
                            logf = np.log((pdf[x]))
                        else:
                            if pdf[x] == 0:
                                print("PDF at x is zero")
                            logf =np.log((pdf[x]))
                
                # Calculate logf_r 
                # Parameters to be estimated
                if self.model_type == 0:
                    nu_guess, alpha0_guess, alpha1_guess = guess_r
                elif self.model_type == 1: 
                    nu, alpha0, alpha1, N = guess_r
                elif self.model_type == 2: 
                    nu, alpha0, alpha1, N, alpha2 = guess_r
                elif self.model_type == 3: 
                    nu, alpha0, alpha1, N, alpha2, alpha3 = guess_r
                elif self.model_type == 4: 
                    nu, alpha0, alpha1, N, alpha3 = guess_r
                elif self.model_type == 5: 
                    nu, alpha0, alpha1, alpha3 = guess_r
                elif self.model_type == 6: 
                    nu, alpha0, alpha1, alpha2, alpha3 = guess_r

                # The Model
                if self.model_type == 0:
                    mod_r = OpinionFormation(N = 175, T = 1 , nu = nu_guess, alpha0= alpha0_guess , alpha1= alpha1_guess, alpha2 = None, alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= self.model_type)
                elif self.model_type == 1: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= self.model_type)
                elif self.model_type == 2: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
                elif self.model_type == 3: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
                elif self.model_type == 4: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)        
                elif self.model_type == 5: 
                    mod_r = OpinionFormation(N = 20, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)                        
                elif self.model_type == 6: 
                    mod_r = OpinionFormation(N = 20, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
                
                #Solve the Fokker Plank Equation: 
                if self.model_type == 0 or self.model_type == 1:
                    pdf_r = mod_r.CrankNicolson(x_0 = time_series[elem])
                elif self.model_type == 2: 
                    pdf_r = mod_r.CrankNicolson(x_0 = time_series[elem], y = y[elem])
                elif self.model_type == 3 or self.model_type ==6: 
                    pdf_r = mod_r.CrankNicolson(x_0 = time_series[elem], y = y[elem], x_l = x_l[elem])
                elif self.model_type == 4 or self.model_type ==5: 
                    pdf_r = mod_r.CrankNicolson(x_0 = time_series[elem], x_l = x_l[elem])

                # Search for the Value of the PDF at X_k+1
                for x_r in range(len(mod_r.x)):
                    if np.around(mod_r.x[x_r], decimals= 2) == np.around(time_series[elem+1],2):
                        if pdf_r[x_r] <= 0: 
                            pdf_r[x_r] = 0.0000000001
                            logf_r = np.log((pdf_r[x_r]))
                        else:
                            if pdf_r[x_r] == 0:
                                print("PDF at x is zero")
                            logf_r=np.log((pdf_r[x_r]))   
                g[param] = (logf_r - logf)/(eps)
                guess_r = guess_in.copy()
            product.append(g @ g.T) 
        for elem in range(len(product)):
            if elem == 0: 
                sum = product[elem]
            else:
                sum = sum + product[elem]    
        opg = np.linalg.inv(1/len(product) * sum)
        return opg    
        
    def solver_L_BFGS_B(self, initial_guess: list) -> tuple:
        """
        The solver_L_BFGS_B function takes in an initial guess for the parameters and returns the best
        fit parameters. The function uses a L-BFGS-B minimization algorithm to minimize the negative log
        likelihood function. 
        
        The function is called by the following function:
        
        :param initial_guess: list
        :type initial_guess: list
        :return: The optimized parameters, the log likelihood value and the number of iterations
        required to reach convergence
        """

        # Unpack the inital guess
        if self.model_type == 0:
            #T = initial_guess
            nu, alpha0, alpha1 = initial_guess
        elif self.model_type == 1: 
            nu, alpha0, alpha1, N = initial_guess
        elif self.model_type == 2: 
            nu, alpha0, alpha1, N, alpha2= initial_guess
        elif self.model_type == 3: 
            nu, alpha0, alpha1, N, alpha2, alpha3= initial_guess
        elif self.model_type == 4: 
            nu, alpha0, alpha1, N, alpha3= initial_guess
        elif self.model_type == 5: 
            nu, alpha0, alpha1, alpha3= initial_guess
        elif self.model_type == 6: 
            nu, alpha0, alpha1, alpha2, alpha3= initial_guess    
                
        print("The Initial guess" + str(initial_guess))
        
        print('Starting:', mp.current_process().name)
        start = time.time()

        # Minimite the negative Log Likelihood Function 
        if self.model_type == 0:
            #exogenous N
            res = minimize(self.neglogL, (nu, alpha0 , alpha1), method='L-BFGS-B', bounds = [(0.01, None), (None, None), ( 0.1, None)], callback=None)
        elif self.model_type == 1: 
            # endogenous N 
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( 0.1, None), (2, None)],  callback=None)
        elif self.model_type == 2: 
            # endogenous N plus Industrial Production
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N, alpha2), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( 0.1, None), (2, None), (None, None)],  callback=None)
        elif self.model_type == 3: 
            # endogenous N plus Industrial Production plus Lagged Time Series
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N, alpha2, alpha3), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( None, None), (2, None), (None, None), (None, None)],  callback=None)
        elif self.model_type == 4: 
            # endogenous N plus Lagged Feedback
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N, alpha3), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( None, None), (1, None), (None, None)],  callback=None)
        elif self.model_type == 5: 
            # endogenous N plus Lagged Feedback with fixed alpha1 = 0
            res = minimize(self.neglogL, (nu, alpha0, alpha1, alpha3), method='L-BFGS-B', bounds = [(0.001, 20), ( None, None), (None, None), (None, None)],  callback=None)
        elif self.model_type == 6: 
            # endogenous N plus Industrial Production plus Lagged Time Series
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, alpha2, alpha3), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( None, None), (None, None), (None, None)],  callback=None)
        print('Exiting :', mp.current_process().name)

        print("Final Estimates found:  " + str(res.x) + "With Maximized Log Likelihood of:  " + str(res.fun))
        end = time.time()
        dum = (end - start)/60
        print("Time past for one estimation of the parameters:  " + str(dum) + " minutes") 

        return res
    




