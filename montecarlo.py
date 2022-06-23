# Class for Monte Carlo Estimation of the Time Series: 
import estimation
import numpy as np
import multiprocessing
import pandas as pd
import os


class MonteCarlo(object):

    def __init__(self, numSim: int,  estimation: object, parallel: bool, real_data: bool) -> None:
        self.numSim = numSim
        self.estimation = estimation
        self.parallel = parallel
        self.real_data = real_data
  

    def estim(self, init_guess) -> np.array:
        #### Solver BFGS 
        res = self.estimation.solver_BFGS(init_guess)

        #res = self.estimation.solver_bhhh(init_guess, tolerance_level= 0.0001, max_iter= 40)
        estim_array = res.x
        logL_array = res.fun
        if self.real_data == True:
            if self.estimation.model_type == 0:

                if os.path.exists("Estimation/Model_0/estimates_model_0.csv") == False:
                    np.savetxt("Estimation/Model_0/estimates_model_0.csv", [0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/Model_0/estimates_model_0.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/Model_0/estimates_model_0.csv", estim_array, delimiter=",")

                if os.path.exists("Estimation/Model_0/logL_array_model_0.csv") == False:
                    np.savetxt("Estimation/Model_0/logL_array_model_0.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/Model_0/logL_array_model_0.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/Model_0/logL_array_model_0.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_0/initial_estim_model_0.csv") == False:
                    np.savetxt("Estimation/Model_0/initial_estim_model_0.csv", [0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/Model_0/initial_estim_model_0.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/Model_0/initial_estim_model_0.csv", init_guess, delimiter=",")

            elif self.estimation.model_type == 1: 
                estim_old =np.genfromtxt("Estimation/estimates_model_1.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/estimates_model_1.csv", estim_array, delimiter=",")

                logL =np.genfromtxt("Estimation/logL_array_model_1.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/logL_array_model_1.csv", logL_array, delimiter=",")
                
                in_est =np.genfromtxt("Estimation/initial_estim_model_1.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/initial_estim_model_1.csv", init_guess, delimiter=",")

            elif self.estimation.model_type == 2: 
                estim_old =np.genfromtxt("Estimation/estimates_model_2.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/estimates_model_2.csv", estim_array, delimiter=",")

                logL =np.genfromtxt("Estimation/logL_array_model_2.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/logL_array_model_2.csv", logL_array, delimiter=",")
                
                in_est =np.genfromtxt("Estimation/initial_estim_model_2.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/initial_estim_model_2.csv", init_guess, delimiter=",")

            elif self.estimation.model_type == 3: 
                estim_old =np.genfromtxt("Estimation/estimates_model_3.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/estimates_model_3.csv", estim_array, delimiter=",")

                logL =np.genfromtxt("Estimation/logL_array_model_3.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/logL_array_model_3.csv", logL_array, delimiter=",")
                
                in_est =np.genfromtxt("Estimation/initial_estim_model_3.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/initial_estim_model_3.csv", init_guess, delimiter=",")
        
        elif self.real_data == False: 

            if self.estimation.model_type == 0:

                if os.path.exists("Estimation/sim_Data/exoN/estimates_model_0.csv") == False:
                    np.savetxt("Estimation/sim_Data/exoN/estimates_model_0.csv", [0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/sim_Data/exoN/estimates_model_0.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/sim_Data/exoN/estimates_model_0.csv", estim_array, delimiter=",")

                if os.path.exists("Estimation/sim_Data/exoN/logL_array_model_0.csv") == False:
                    np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/sim_Data/exoN/logL_array_model_0.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/sim_Data/exoN/initial_estim_model_0.csv") == False:
                    np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0.csv", [0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/sim_Data/exoN/initial_estim_model_0.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0.csv", init_guess, delimiter=",")

    def run(self) -> np.array:

        if self.parallel == False:
            for i in range(self.numSim):
                
                if self.estimation.model_type == 0:
                    init_guess = (1 + np.random.normal(0, 0.03, 1), 0  + np.random.normal(0, 0.01, 1), 1.2 + np.random.normal(0, 0.02, 1))
                elif self.estimation.model_type == 1: 
                    init_guess = (0.15 + np.random.normal(0, 0.1, 1), 0.09  + np.random.normal(0, 0.01, 1), 0.9 + np.random.normal(0, 0.1, 1), 21.21 + np.random.normal(0, 5, 1))
                elif self.estimation.model_type == 2: 
                    pass
                elif self.estimation.model_type == 3: 
                    pass
                
                self.estim(tuple(init_guess))
        else: 
            
            for i in range(int(self.numSim/5)):
                jobs = []
                for d in range(5):
                    if self.estimation.model_type == 0:
                        init_guess = (1 + np.random.normal(0, 0.1, 1), 0  + np.random.normal(0, 0.01, 1), 0.8 + np.random.normal(0, 0.1, 1))
                    elif self.estimation.model_type == 1: 
                        init_guess = (1 + np.random.normal(0, 0.1, 1), 0  + np.random.normal(0, 0.01, 1), 0.8 + np.random.normal(0, 0.1, 1), 50 + np.random.normal(0, 5, 1))
                    elif self.estimation.model_type == 2: 
                        pass
                    elif self.estimation.model_type == 3: 
                        pass
                    p = multiprocessing.Process(target=self.estim, args= (tuple(init_guess),))
                    jobs.append(p)
                    p.start()#

                for proc in jobs:
                    proc.join()