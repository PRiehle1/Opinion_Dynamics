# Class for Monte Carlo Estimation of the Time Series: 
import estimation
import numpy as np
import os
from model import OpinionFormation


class MonteCarlo():

    def __init__(self, numSim: int, model: object, estimation: object, real_data: bool) -> None:
        self.numSim = numSim
        self.estimation = estimation
        self.model = model
        self.real_data = real_data
  

    def estim(self, init_guess) -> np.array:

        #### Solver Nelder Mead
        init_guess = list(init_guess)

        for elem in range(len(init_guess)):
             if elem == 1:
                 init_guess[elem] = np.around(init_guess[elem] + np.random.normal(0,0.3),4)
             else:
                 init_guess[elem] = np.around(init_guess[elem] + np.random.normal(0,0.1),4)

        res = self.estimation.solver_Nelder_Mead(init_guess)

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
                if os.path.exists("Estimation/Model_1/estimates_model_1.csv") == False:
                    np.savetxt("Estimation/Model_1/estimates_model_1.csv", [0,0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/Model_1/estimates_model_1.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/Model_1/estimates_model_1.csv", estim_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_1/logL_array_model_1.csv") == False:
                    np.savetxt("Estimation/Model_1/logL_array_model_1.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/Model_1/logL_array_model_1.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/Model_1/logL_array_model_1.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_1/initial_estim_model_1.csv") == False:
                    np.savetxt("Estimation/Model_1/initial_estim_model_1.csv", [0,0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/Model_1/initial_estim_model_1.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/Model_1/initial_estim_model_1.csv", init_guess, delimiter=",")

            elif self.estimation.model_type == 2: 
                if os.path.exists("Estimation/Model_2/estimates_model_2.csv") == False:
                    np.savetxt("Estimation/Model_2/estimates_model_2.csv", [0,0,0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/Model_2/estimates_model_2.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/Model_2/estimates_model_2.csv", estim_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_2/logL_array_model_2.csv") == False:
                    np.savetxt("Estimation/Model_2/logL_array_model_2.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/Model_2/logL_array_model_2.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/Model_2/logL_array_model_2.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_2/initial_estim_model_2.csv") == False:
                    np.savetxt("Estimation/Model_2/initial_estim_model_2.csv", [0,0,0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/Model_2/initial_estim_model_2.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/Model_2/initial_estim_model_2.csv", init_guess, delimiter=",")

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


        self.estim(tuple(init_guess))
        


if __name__ == '__main__':

    # Load the Training Data 
    import pandas as pd
    from sympy import *
    import sim 
    import matplotlib.pyplot as plt
    import multiprocessing as mp

################################################################################################################################################
#                                                    Simulated Data
################################################################################################################################################
# # #   First Set of Data 
#     numSim = 200
#     sim_1 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/100, model_type =0, seed = 3)  

#     init_guess = (3,0,0.8)
#     for i in range(int(numSim/20)):
#         jobs = []
#         test_data_1 = []
#         mC_1 = []
#         for proc in range(20):
#             # Simulate the time series:      
#             test_data_1.append(sim_1.simulation(0, sim_length = 200))
#             mC_1.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_1[proc],multiprocess= False,model_type=0), real_data= False))
#             p = mp.Process(target=mC_1[proc].run, args= (tuple(init_guess),))
#             jobs.append(p)
#             p.start()

#         for proc in jobs:
#             proc.join()
# # #############################################################    
# # Second Set of Data 

#     numSim = 200
#     sim_2 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0.1, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/100, model_type =0, seed = 3)  

#     init_guess = (3,0.1,0.8)
#     for i in range(int(numSim/20)):
#         jobs = []
#         test_data_2 = []
#         mC_2 = []
#         for proc in range(20):
#             # Simulate the time series:      
#             test_data_2.append(sim_2.simulation(-0.59, sim_length = 200))
#             mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_2[proc],multiprocess= False,model_type=0), real_data= False))
#             p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
#             jobs.append(p)
#             p.start()

#         for proc in jobs:
#             proc.join()

#     # # #Third Set of Data 

#     numSim = 40
#     sim_3 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/100, model_type =0, seed = 3)  

#     init_guess = (3,0,1.2)
#     for i in range(int(numSim/20)):
#         jobs = []
#         test_data_3 = []
#         mC_3 = []
#         for proc in range(20):
#             # Simulate the time series:      
#             test_data_3.append(sim_3.simulation(-0.6, sim_length = 200))
#             mC_3.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_3[proc],multiprocess= False,model_type=0), real_data= False))
#             p = mp.Process(target=mC_3[proc].run, args= (tuple(init_guess),))
#             jobs.append(p)
#             p.start()

#         for proc in jobs:
#             proc.join()

#     # # # Fourth Set of Data 

#     numSim = 200
#     sim_4 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0.1, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/100, model_type =0, seed = 3)  

#     init_guess = (3,0.1,1.2)
#     for i in range(int(numSim/20)):
#         jobs = []
#         test_data_4 = []
#         mC_4 = []
#         for proc in range(20):
#             # Simulate the time series:      
#             test_data_4.append(sim_4.simulation(-0.6, sim_length = 200))
#             mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_4[proc],multiprocess= False,model_type=0), real_data= False))
#             p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
#             jobs.append(p)
#             p.start()

#         for proc in jobs:
#             proc.join()
################################################################################################################################################
#                                                   Real Data 
################################################################################################################################################
    #  
    from data_reader import data_reader

    data = data_reader(time_period= 365)
    zew = data.zew()/100
    ip = data.industrial_production()
    numSim = 20
   
    # init_guess = (0.8,0,1.18)
    # for i in range(int(numSim/5)):
    #     jobs = []
    #     data = []
    #     mC_1 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         data.append(zew)
    #         mC_1.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(data[proc],multiprocess= False,model_type=0), real_data= True))
    #         p = mp.Process(target=mC_1[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

    #Model with endogenous N

    # init_guess = (0.12,0.09,0.99, 22)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     data = []
    #     mC_2 = []
    #     for proc in range(20):
    #         # Simulate the time series:      
    #         data.append(zew)
    #         mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(data[proc], y = ip, multiprocess= False,model_type=1), real_data= True))
    #         p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

    # # Model with industrial production
    init_guess = (0.12,0.09,0.99, 22, -4.5)
    for i in range(int(numSim/20)):
        jobs = []
        data = []
        mC_2 = []
        for proc in range(1):
            # Simulate the time series:      
            data.append(zew)
            mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(data[proc], y = ip, multiprocess= False,model_type=2), real_data= True))
            p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

    

