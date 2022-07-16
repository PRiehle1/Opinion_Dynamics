# Class for Monte Carlo Estimation of the Time Series: 
from mimetypes import init
import estimation
import numpy as np
import multiprocessing
import pandas as pd
import os
from model import OpinionFormation


class MonteCarlo():

    def __init__(self, numSim: int, model: object, estimation: object, multiprocess: bool, real_data: bool) -> None:
        self.numSim = numSim
        self.estimation = estimation
        self.model = model
        self.multiprocess = multiprocess
        self.real_data = real_data
  

    def estim(self, init_guess) -> np.array:
        #### Solver BFGS 
        init_guess = list(init_guess)
        # for elem in range(len(init_guess)):
        #     if elem == 1:
        #         init_guess[elem] = np.around(init_guess[elem] + np.random.normal(0,0.001),4)
        #     else:
        #         init_guess[elem] = np.around(init_guess[elem] + np.random.normal(0,0.005),4)

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

    def run(self, init_guess) -> np.array:

        if self.multiprocess == False:

            for i in range(self.numSim):

                self.estim(tuple(init_guess))
        
        else: 
            ## Wenn Anzahl der Estimations nicht durch 4 dividierbar sind FEHLERMELDUNG AUSGEBEN und um Änderung bitten/ das Programm das ändern lassen 
            for i in range(int(self.numSim/6)):
                jobs = []
                for d in range(4):
                    p = multiprocessing.Process(target=self.estim, args= (tuple(init_guess),))
                    jobs.append(p)
                    p.start()#

                for proc in jobs:
                    proc.join()

if __name__ == '__main__':

    # Load the Training Data 
    import pandas as pd
    from sympy import *
    import multiprocessing
    import sim 
    import matplotlib.pyplot as plt

    # # First Set of Data 

    # #Simulated data
    # sim_1 = sim.Simulation(N = 50, T = 30, nu = 3 , alpha0 = 0, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/300, model_type =0, seed = 3)  
    # test_data_1 = sim_1.simulation(0, sim_length = 200)
    # #plt.plot(test_data_1)
    # #plt.show()

    # # Set up the Monte Carlo Estimation
    # mC_1 = MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_1,multiprocess= True,model_type=0), multiprocess= False, real_data= False)
    # mC_1.run(init_guess=(3,0,0.8))

    # # #Second Set of Data 

    # # Simulated data
    # sim_2 = sim.Simulation(N = 50, T = 30, nu = 3 , alpha0 = 0.2, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/300, model_type =0, seed = 150)  
    # test_data_2 = sim_2.simulation(0, sim_length = 200)
    # #plt.plot(test_data_2)
    # #plt.show()

    # # Set up the Monte Carlo Estimation
    # mC_2 = MonteCarlo(numSim= 5 , model = OpinionFormation ,estimation= estimation.Estimation(test_data_2, multiprocess= True, model_type= 0), multiprocess= False, real_data = False)
    # mC_2.run(init_guess= (3,0.2,0.8))

    # #Third Set of Data 

    # # Simulated data
    # sim_3= sim.Simulation(N = 50, T = 30, nu = 3 , alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/16, model_type =0, seed = 150)  
    # test_data_3 = sim_3.simulation(0, sim_length = 200)
    # plt.plot(test_data_3)
    # plt.show()

    # # # # Set up the Monte Carlo Estimation
    # mC_3 = MonteCarlo(numSim= 5 , model = OpinionFormation ,estimation= estimation.Estimation(test_data_3, multiprocess= True, model_type= 0), multiprocess= False, real_data = False)
    # mC_3.run(init_guess= (3,0,1.2))

    # # # Fourth Set of Data 

    # # Simulated data
    # sim_4 = sim.Simulation(N = 50, T = 10, nu = 3 , alpha0 = 0.2, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/300, model_type =0, seed = 150)  
    # test_data_4 = sim_4.simulation(0, sim_length = 200)
    # plt.plot(test_data_4)
    # plt.show()

    # # Set up the Monte Carlo Estimation
    # mC_4 = MonteCarlo(numSim= 5 , model = OpinionFormation ,estimation= estimation.Estimation(test_data_4, multiprocess= True, model_type= 0), multiprocess= False, real_data = False)
    # mC_4.run(init_guess= (3,0.2,1.2))

    #Real Data 
    from data_reader import data_reader

    data = data_reader(time_period= 175)
    zew = data.zew()/100
    ip = data.industrial_production()

    # Model with exogenous N
    mC = MonteCarlo(numSim= 30 , model = OpinionFormation ,estimation= estimation.Estimation(zew, y= ip, multiprocess= True, model_type= 0), multiprocess= False, real_data = True)
    mC.run(init_guess= (5,0.01,1.19))

    # # Model with endogenous N 
    # mC = MonteCarlo(numSim= 1 , model = OpinionFormation ,estimation= estimation.Estimation(zew, y= ip, multiprocess= True, model_type= 1), multiprocess= False, real_data = True)
    # mC.run(init_guess= (0.15,0.1,0.99, 21))

    # # # Model with industrial production
    # # mC = MonteCarlo(numSim= 1 , model = OpinionFormation ,estimation= estimation.Estimation(zew, y= ip, multiprocess= True, model_type= 2), multiprocess= False, real_data = True)
    # # mC.run(init_guess= (0.2,0.1,1, 20, (-4.5)))

    

