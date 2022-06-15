# Import Packages 
from tqdm import tqdm
import time
import numpy as np
import model
from scipy.optimize import minimize, dual_annealing
import multiprocessing as mp
from optimparallel import minimize_parallel


# Define the class 

class Estimation(object):
    
    ''' Class for the Estimation of the Social Model'''
    def __init__(self, time_series: np.array, multiprocess : bool) -> None: 
        self.time_series = time_series 
        self.multiprocess = multiprocess
        

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
    
        # Parameters to be estimated
        nu, alpha0, alpha1= guess

        print("The Minimization_Guess is: " + str(guess))

        # The Model
        mod = model.OpinionFormation(N = 175, T = 3, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None,alpha3 = None, y = None, deltax= 0.01, deltat= 1/16)
        
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
                    if mod.x[x] == np.around(time_series[elem+1],2):
                        logf[elem] = np.log(np.abs(pdf[elem,x]))
            logL = (-1)* np.sum(logf)
            print("The Log Likelihood is: " + str(logL)) 
        
        else:   
            start = time.time()

            for elem in range(len(time_series)-1):
                # Solve the Fokker Plank Equation: 
                pdf = mod.CrankNicolson(x_0 = time_series[elem])
                # Search for the Value of the PDF at X_k+1
                for x in range(len(mod.x)):
                    if mod.x[x] == np.around(time_series[elem+1],2):
                        logf[elem] = np.log((np.abs(pdf[x])))
        
            logL = np.sum(logf)
            
            print("The Log Likelihood is: " + str(logL)) 
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

    def gradient(self, guess_initial: tuple, eps: float) -> np.array:
        """
        The gradient function calculates the gradient of the log likelihood function at a given point. 
        The gradient is a vector with four components, one for each parameter in our model. 
        It is calculated by taking the partial derivative of each component of the log likelihood function with respect to that parameter.

        Args:
            guess_initial (tuple): the actual guess of the parameters
            eps (float): epsilion for the gradient calculation 

        Returns:
            np.array: array of the gradient values
             
        """

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
    
    def cov_matrix(self, gradient: np.array) -> np.array:
        """
        The cov_matrix function takes in the gradient and returns the outer product which is the variance-covariance matrix 

        Args:
            gradient (np.array): Pass the gradient that is used to calculate the covariance matrix

        Returns:
            np.array: The covariance matrix of the gardient
            
        """
        
        r_t = (len(self.time_series)) * np.dot(gradient,gradient.T)
        
        return r_t
        
#########################################################################################################################################################################################
#                                               BFGS Minimisation
#########################################################################################################################################################################################
    def solver_BFGS(self, initial_guess: list) -> tuple:
        """
        The solver_BFGS function takes in an initial guess for the parameters and returns the 
        best fit parameters. The function uses a L-BFGS-B minimization algorithm to minimize 
        the negative log likelihood function. 

        Args:
            initial_guess (list): Initialize the minimization process

        Returns:
            tuple: he optimized parameters, the log likelihood value and the number of iterations required to reach convergence
        """
        
        # Unpack the inital guess
        nu, alpha0, alpha1, N = initial_guess
        print("The Initial guess" + str(initial_guess))
        
        print('Starting:', mp.current_process().name)
        start = time.time()
        
        # Minimite the negative Log Likelihood Function endogenous N
        #res = minimize(self.neglogL, (nu, alpha0 , alpha1, N), method='L-BFGS-B', bounds = [(0.0001, None), (-2, 2), ( 0, None), (2, None)],  callback=None, options={ 'maxiter': 100, 'iprint': -1})
        
        # Minimite the negative Log Likelihood Function exogenous N
        res = minimize(self.neglogL, (nu, alpha0 , alpha1), method='L-BFGS-B', bounds = [(0.0001, None), (-2, 2), ( 0, None)],  callback=None, options={ 'maxiter': 100, 'iprint': -1})


        #res = minimize_parallel(self.neglogL, x0 =(nu, alpha0 , alpha1, N) )
        
        print('Exiting :', mp.current_process().name)

        print("Final Estimates found:  " + str(res.x) + "With Maximized Log Likelihood of:  " + str(res.fun))
        end = time.time()
        dum = (end - start)/60
        print("Time past for one estimation of the parameters:  " + str(dum) + " minutes") 

        return res
    
#########################################################################################################################################################################################
#                                               BHHHH Maximisation
#########################################################################################################################################################################################


    def bhhh(self, initial_guess: tuple, tolerance_level: float, max_iter:int) -> np.array:
        """
        The bhhh function takes in the initial guess, tolerance level and maximum number of iterations as input. 
        It returns the final estimate after performing bhhh method for a given number of iterations.

        Args:
            initial_guess (tuple): Set the initial value of beta
            tolerance_level (float): Determine the convergence of the algorithm
            max_iter (int): Set the maximum number of iterations

        Returns:
            np.array: The final estimated
           
        """
        ##########################
        ### Initial Values 
        ###########################

        # Calculate the initial Gradient
        g_in = self.gradient(initial_guess, eps = 0.01)
        
        # Calculate the initial Variance Covariance Matrix
        r_t_in = self.cov_matrix(g_in)
        
        # Check if the Variance Covariance Matrix is singular
        if (np.linalg.det(r_t_in)):
            print("Covariance Matrix is  not singular ")
            pass
        else: 
            print("Covariance Matrix is singular ")
            r_t_in = np.array([[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]])
        
        # Calculate the initial direction vector
        direc_in = np.dot(np.linalg.inv(r_t_in),g_in).reshape(4,)
        
        lamb =  1
        delta = 0.25

        # Initial Beta         
        beta_in= np.array(initial_guess)

        for it in range(max_iter):
            print("Number of Iterations:" + str(it))
            if it == 0:
                # Calculate the Lambda
                # Helper Function
                def calcnu(lamb):
                    logL =  self.logL(list(beta_in))
                    nu = float((self.logL(list((beta_in + lamb * direc_in))) - logL)) / (lamb * float(np.dot(direc_in, g_in)))
                    return nu
        
                
                if calcnu(lamb) >= delta:
                    lamb = 1
                    print("Lambda is:   " + str(lamb))
                else:
                    while delta >= calcnu(lamb) or calcnu(lamb) >=1-delta:
                        lamb *=0.8
                        if lamb <= 0.09:
                            pass
                    print("Lambda is:   " + str(lamb))
                
                beta = beta_in + lamb*direc_in
                print("The actual Estimate is:   " +str(beta))
            else:
                
                # Calculate the Gradient
                g = self.gradient(tuple(beta), eps = 0.01)
                
                # Calculate the initial Variance Covariance Matrix
                r_t = self.cov_matrix(g)
                
                # Check if the Variance Covariance Matrix is singular
                if (np.linalg.det(r_t)):
                    print("Covariance Matrix is  not singular ")
                    pass
                else: 
                    print("Covariance Matrix is singular ")
                    r_t = np.array([[10000, 0, 0, 0], [0, 100000, 0, 0], [0, 0, 10000, 0], [0, 0, 0, 1000]])
                
                # Calculate the initial direction vector
                direc = np.dot(np.linalg.inv(r_t),g).reshape(4,)

                # Check for convergence
                dum = np.zeros(len(direc))
                for elem in range(len(direc)):
                    dum[elem] = np.abs(direc[elem])/ max((1, np.abs(beta[elem])))
        
                if max(dum) < tolerance_level:
                    print(" Final Estimate foud" + str(beta))
                    return beta 

                # Calculate the Lambda
                # Helper Function
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
                        if lamb <= 0.09:
                            pass
                    print("Lambda is:   " + str(lamb))

                beta = beta + lamb*direc
                print("The actual Estimate is:   " +str(beta))
                





