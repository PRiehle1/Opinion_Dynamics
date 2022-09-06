# Import Packages 
from tqdm import tqdm
import time
import numpy as np
from errors import UncompleteLikelihoodError
from model import OpinionFormation
from scipy.integrate import simps
from scipy import interpolate
from scipy.optimize import minimize, dual_annealing, differential_evolution

import multiprocessing as mp



# Define the class 

class Estimation():
    
    ''' Class for the Estimation of the Social Model'''
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
            nu, alpha0, alpha1, N = guess
        elif self.model_type == 2: 
            nu, alpha0, alpha1, N, alpha2 = guess
        elif self.model_type == 3: 
            nu, alpha0, alpha1, N, alpha2, alpha3 = guess
        elif self.model_type == 4: 
            nu, alpha0, alpha1, N, alpha3 = guess

        # The Model
        if self.model_type == 0:
            mod = OpinionFormation(N = 50, T = 1 , nu = nu_guess, alpha0= alpha0_guess , alpha1= alpha1_guess, alpha2 = None, alpha3 = None, deltax= 1/50, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 1: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 2: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 3: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 4: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
    
        # Initialize the log(function(X, Theta))
        logf = []
        ######################################################################################################################################
        if self.multiprocess == True:
            pass
        #     start = time.time()
        #     # Time Series to List
        #     time_series_list = list(time_series)
        #     # Multiprocessing 
        #     pool = mp.Pool(2)         
        #     # Calculate the PDF for all values in the Time Series
        #     if self.model_type == 0 or self.model_type == 1:
        #         for _ in range(10):
        #             pdf = list(pool.starmap(mod.CrankNicolson, zip(time_series_list)))
        #             # Check if the area under the PDF equals one if not adapt the grid size in time and space 
        #             pdf = np.array(pdf)
        #             dummy_1 = mod.dt
        #             for elem in range(len(pdf)-1):
        #                 area = simps(pdf[elem,:], x = mod.x)
        #                 if area > 1 + 0.03 or area < 1- 0.03:
        #                     dt_new = dummy_1/2
        #                     print("The grid size is expanded to dt = " + str(dt_new))
        #                     mod = OpinionFormation(N = 175, T = 2, nu = nu_guess, alpha0= alpha0_guess , alpha1= alpha1_guess, alpha2 = None,alpha3 = None, deltax= mod.dx, deltat= dt_new, model_type= self.model_type)
        #                     pdf = []
        #                     break
        #             if mod.dt == dummy_1:
        #                 break
        #     elif self.model_type == 2: 
        #         # y to List
        #         y_list = list(y)
        #         for _ in range(10):
        #             pdf = list(pool.starmap(mod.CrankNicolson, zip(tuple(time_series_list), tuple(y_list))))
        #             # Check if the area under the PDF equals one if not adapt the grid size in time 
        #             pdf = np.array(pdf)
        #             dummy_1 = mod.dt
        #             for elem in range(len(pdf)-1):
        #                 area = simps(pdf[elem,:], x = mod.x)
        #                 if area > 1 + 0.03 or area < 1- 0.03:
        #                     dt_new = dummy_1/2
        #                     print("The grid size is expanded to dt = " + str(dt_new))
        #                     mod = OpinionFormation(N = N, T = 1.6, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2,alpha3 = None, deltax= mod.dx, deltat= dt_new, model_type= self.model_type)
        #                     pdf = []
        #                     break
        #             if mod.dt == dummy_1:
        #                 break         
        #     elif self.model_type == 3: 
        #         # y to List
        #         y_list = list(y)
        #         # Lagged x to list 
        #         x_l_list = list(x_l)
        #         pdf = list(tqdm(pool.imap(mod.CrankNicolson, time_series_list, y_list, x_l_list)))
        #     pool.close()  

        #     # Search for the Value of the PDF at X_k+1
        #     for elem in range(len(pdf)-1):
        #         # Interpolate the PDF
        #         pdf_new = interpolate.interp1d(mod.x,pdf[elem])
        #         # Store the Likelihood Value
        #         logf.append(np.log(pdf_new(time_series[elem+1])))


        ################################################################################################################################################
        else:   
            import matplotlib.pyplot as plt 
            start = time.time()
            for elem in range(len(time_series)-1):
                #Solve the Fokker Plank Equation: 
                if self.model_type == 0 or self.model_type == 1:
                    pdf = mod.CrankNicolson(x_0 = time_series[elem])
                elif self.model_type == 2: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], y = y[elem])
                elif self.model_type == 3: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], y = y[elem], x_l = x_l[elem])
                elif self.model_type == 4: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], x_l = x_l[elem])
                
                # Search for the Value of the PDF at X_k+1
                for x in range(len(mod.x)):
                    if np.around(mod.x[x], decimals= 2) == np.around(time_series[elem+1],2):
                        if pdf[x] <= 0: 
                            pdf[x] = 0.0000000001
                            logf.append(np.log((pdf[x])))
                        else:
                            if pdf[x] == 0:
                                print("PDF at x is zero")
                            logf.append(np.log((pdf[x])))


#########################################################################################################################################################           
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


        # The Model
        if self.model_type == 0:
            mod = OpinionFormation(N = 175, T = 1 , nu = nu_guess, alpha0= alpha0_guess , alpha1= alpha1_guess, alpha2 = None, alpha3 = None, deltax= 1/100, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 1: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 2: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 3: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
        elif self.model_type == 4: 
            mod = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)        
        
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
                elif self.model_type == 3: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], y = y[elem], x_l = x_l[elem])
                elif self.model_type == 4: 
                    pdf = mod.CrankNicolson(x_0 = time_series[elem], x_l = x_l[elem])

                # Search for the Value of the PDF at X_k+1
                for x in range(len(mod.x)):
                    if np.around(mod.x[x], decimals= 2) == np.around(time_series[elem+1],2):
                        if pdf[x] == 0: 
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
                    nu, alpha0, alpha1, N, alpha3 = guess

                # The Model
                if self.model_type == 0:
                    mod_r = OpinionFormation(N = 175, T = 1 , nu = nu_guess, alpha0= alpha0_guess , alpha1= alpha1_guess, alpha2 = None, alpha3 = None, deltax= 1/100, deltat= 1/16, model_type= self.model_type)
                elif self.model_type == 1: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = None, deltax= 1/100, deltat= 1/16, model_type= self.model_type)
                elif self.model_type == 2: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = None, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
                elif self.model_type == 3: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = alpha2, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)
                elif self.model_type == 4: 
                    mod_r = OpinionFormation(N = N, T = 1, nu = nu, alpha0= alpha0 , alpha1= alpha1, alpha2 = None, alpha3 = alpha3, deltax= 0.01, deltat= 1/16, model_type= self.model_type)        
                
                #Solve the Fokker Plank Equation: 
                if self.model_type == 0 or self.model_type == 1:
                    pdf_r = mod_r.CrankNicolson(x_0 = time_series[elem])
                elif self.model_type == 2: 
                    pdf_r = mod_r.CrankNicolson(x_0 = time_series[elem], y = y[elem])
                elif self.model_type == 3: 
                    pdf_r = mod_r.CrankNicolson(x_0 = time_series[elem], y = y[elem], x_l = x_l[elem])
                elif self.model_type == 4: 
                    pdf_r = mod.CrankNicolson(x_0 = time_series[elem], x_l = x_l[elem])

                # Search for the Value of the PDF at X_k+1
                for x_r in range(len(mod_r.x)):
                    if np.around(mod_r.x[x_r], decimals= 2) == np.around(time_series[elem+1],2):
                        if pdf_r[x_r] == 0: 
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
        
#########################################################################################################################################################################################
#                                               Nelder Mead Optimization
#########################################################################################################################################################################################
    def solver_Nelder_Mead(self, initial_guess: list) -> tuple:
        """
        The solver_Nelder_Mead function takes in an initial guess for the parameters and returns the 
        best fit parameters. The function uses a Nelder Mead minimization algorithm to minimize 
        the negative log likelihood function. 

        Args:
            initial_guess (list): Initialize the minimization process

        Returns:
            tuple: he optimized parameters, the log likelihood value and the number of iterations required to reach convergence
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
        
        print("The Initial guess" + str(initial_guess))
        
        print('Starting:', mp.current_process().name)
        start = time.time()

        # Minimite the negative Log Likelihood Function 
        if self.model_type == 0:
            #exogenous N
            res = minimize(self.neglogL, (nu, alpha0 , alpha1), method='L-BFGS-B', bounds = [(0.01, None), (None, None), ( 0.1, None)], callback=None, options={'gtol': 1e-03, 'eps': 1.4901161193847656e-04})
        elif self.model_type == 1: 
            # endogenous N 
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( 0.1, None), (2, None)],  callback=None,options={'gtol': 1e-03, 'eps': 1.4901161193847656e-04})
        elif self.model_type == 2: 
            # endogenous N plus Industrial Production
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N, alpha2), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( 0.1, None), (2, None), (None, None)],  callback=None,options={'gtol': 1e-03, 'eps': 1.4901161193847656e-04})
        elif self.model_type == 3: 
            # endogenous N plus Industrial Production plus Lagged Time Series
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N, alpha2, alpha3), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( 0.1, None), (2, None), (None, None), (None, None)],  callback=None,options={'gtol': 1e-03, 'eps': 1.4901161193847656e-04})
        elif self.model_type == 4: 
            # endogenous N plus Lagged Feedback
            res = minimize(self.neglogL, (nu, alpha0 , alpha1, N, alpha3), method='L-BFGS-B', bounds = [(0.001, 20), (None, None), ( 0.1, None), (2, None), (None, None)],  callback=None,options={'gtol': 1e-03, 'eps': 1.4901161193847656e-04})

        print('Exiting :', mp.current_process().name)

        print("Final Estimates found:  " + str(res.x) + "With Maximized Log Likelihood of:  " + str(res.fun))
        end = time.time()
        dum = (end - start)/60
        print("Time past for one estimation of the parameters:  " + str(dum) + " minutes") 

        return res
    
#########################################################################################################################################################################################
#                                               BHHHH Maximisation
#########################################################################################################################################################################################


    def solver_bhhh(self, initial_guess: tuple, tolerance_level: float, max_iter:int) -> np.array:
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
        g_in = self.gradient(np.concatenate(initial_guess).ravel(), eps = 0.01)
        
        # Calculate the initial Variance Covariance Matrix
        r_t_in = self.cov_matrix(g_in)
        
        # Check if the Variance Covariance Matrix is singular
        if (np.linalg.det(r_t_in)):
            print("Covariance Matrix is  not singular ")
            pass
        else: 
            print("Covariance Matrix is singular ")
            r_t_in = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1000]]) # Reshape according to number of
        
        # Calculate the initial direction vector
        direc_in = np.dot(np.linalg.inv(r_t_in),g_in).reshape(3,) # Change according to the number of parameters 
        
        lamb =  1
        delta = 0.25

        # Initial Beta         
        beta_in= np.concatenate(initial_guess).ravel()

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
                    r_t = np.array([[10000, 0, 0], [0, 10000, 0], [0, 0, 1000]])
                
                # Calculate the initial direction vector
                direc = np.dot(np.linalg.inv(r_t),g).reshape(3,)

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
                





