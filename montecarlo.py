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
                 init_guess[elem] = np.around(init_guess[elem],4)
             else:
                 init_guess[elem] = np.around(init_guess[elem],4)

        res = self.estimation.solver_Nelder_Mead(init_guess)
        opg_res = self.estimation.outer_product_gradient(res.x, eps = 0.0001)

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

                if os.path.exists("Estimation/Model_0/var_model_0.csv") == False:
                    np.savetxt("Estimation/Model_0/var_model_0.csv", [0,0,0], delimiter=",")
                opg_old =np.genfromtxt("Estimation/Model_0/var_model_0.csv", delimiter=',')
                opg_array = np.vstack([opg_old, opg_res.diagonal()])
                np.savetxt("Estimation/Model_0/var_model_0.csv", opg_array, delimiter=",")

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

                if os.path.exists("Estimation/Model_1/var_model_1.csv") == False:
                    np.savetxt("Estimation/Model_1/var_model_1.csv", [0,0,0,0], delimiter=",")
                opg_old =np.genfromtxt("Estimation/Model_1/var_model_1.csv", delimiter=',')
                opg_array = np.vstack([opg_old, opg_res.diagonal()])
                np.savetxt("Estimation/Model_1/var_model_1.csv", opg_array, delimiter=",")

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

                if os.path.exists("Estimation/Model_2/var_model_2.csv") == False:
                    np.savetxt("Estimation/Model_2/var_model_2.csv", [0,0,0,0,0], delimiter=",")
                opg_old =np.genfromtxt("Estimation/Model_2/var_model_2.csv", delimiter=',')
                opg_array = np.vstack([opg_old, opg_res.diagonal()])
                np.savetxt("Estimation/Model_2/var_model_2.csv", opg_array, delimiter=",")

            elif self.estimation.model_type == 3: 
                if os.path.exists("Estimation/Model_3/estimates_model_3.csv") == False:
                    np.savetxt("Estimation/Model_3/estimates_model_3.csv", [0,0,0,0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/Model_3/estimates_model_3.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/Model_3/estimates_model_3.csv", estim_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_3/logL_array_model_3.csv") == False:
                    np.savetxt("Estimation/Model_3/logL_array_model_3.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/Model_3/logL_array_model_3.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/Model_3/logL_array_model_3.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_3/initial_estim_model_3.csv") == False:
                    np.savetxt("Estimation/Model_3/initial_estim_model_3.csv", [0,0,0,0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/Model_3/initial_estim_model_3.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/Model_3/initial_estim_model_3.csv", init_guess, delimiter=",")

                if os.path.exists("Estimation/Model_3/var_model_3.csv") == False:
                    np.savetxt("Estimation/Model_3/var_model_3.csv", [0,0,0,0,0,0], delimiter=",")
                opg_old =np.genfromtxt("Estimation/Model_3/var_model_3.csv", delimiter=',')
                opg_array = np.vstack([opg_old, opg_res.diagonal()])
                np.savetxt("Estimation/Model_3/var_model_3.csv", opg_array, delimiter=",")


            elif self.estimation.model_type == 4: 
                if os.path.exists("Estimation/Model_4/estimates_model_4.csv") == False:
                    np.savetxt("Estimation/Model_4/estimates_model_4.csv", [0,0,0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/Model_4/estimates_model_4.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/Model_4/estimates_model_4.csv", estim_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_4/logL_array_model_4.csv") == False:
                    np.savetxt("Estimation/Model_4/logL_array_model_4.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/Model_4/logL_array_model_4.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/Model_4/logL_array_model_4.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_4/initial_estim_model_4.csv") == False:
                    np.savetxt("Estimation/Model_4/initial_estim_model_4.csv", [0,0,0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/Model_4/initial_estim_model_4.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/Model_4/initial_estim_model_4.csv", init_guess, delimiter=",")

                if os.path.exists("Estimation/Model_4/var_model_4.csv") == False:
                    np.savetxt("Estimation/Model_4/var_model_4.csv", [0,0,0,0,0], delimiter=",")
                opg_old =np.genfromtxt("Estimation/Model_4/var_model_4.csv", delimiter=',')
                opg_array = np.vstack([opg_old, opg_res.diagonal()])
                np.savetxt("Estimation/Model_4/var_model_4.csv", opg_array, delimiter=",")
        
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
# #   #First Set of Data 
# #     numSim = 200
# #     sim_1 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/100, model_type =0, seed = 3)  

# #     init_guess = (3,0,0.8)
# #     for i in range(int(numSim/8)):
# #         jobs = []
# #         test_data_1 = []
# #         mC_1 = []
# #         for proc in range(8):
# #             # Simulate the time series:      
# #             test_data_1.append(sim_1.simulation(0, sim_length = 60))
# #             mC_1.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_1[proc],multiprocess= False,model_type=0), real_data= False))
# #             p = mp.Process(target=mC_1[proc].run, args= (tuple(init_guess),))
# #             jobs.append(p)
# #             p.start()

# #         for proc in jobs:
# #             proc.join()
# # # #############################################################    
# # # Second Set of Data 

#     # numSim = 200
#     # sim_2 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0.2, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/16, model_type =0, seed = np.random.randint(0,300))  

#     # init_guess = (3,0.2,0.8)
#     # for i in range(int(numSim/5)):
#     #     jobs = []
#     #     test_data_2 = []
#     #     mC_2 = []
#     #     for proc in range(5):
#     #         # Simulate the time series:  
            
#     #         test_data_2.append(sim_2.simulation(0, sim_length = 200))  
#     #         mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_2[proc],multiprocess= False,model_type=0), real_data= False))
#     #         p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
#     #         jobs.append(p)
#     #         p.start()

#     #     for proc in jobs:
#     #          proc.join()

# #     # # #Third Set of Data 

#     # numSim = 200
#     # sim_3 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/200, model_type =0, seed = 3)  

#     # init_guess = (3,0,1.2)
#     # for i in range(int(numSim/6)):
#     #     jobs = []
#     #     test_data_3 = []
#     #     mC_3 = []
#     #     for proc in range(6):
#     #         # Simulate the time series:      
#     #         test_data_3.append(sim_3.simulation(-0.6, sim_length = 200))
#     #         mC_3.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_3[proc],multiprocess= False,model_type=0), real_data= False))
#     #         p = mp.Process(target=mC_3[proc].run, args= (tuple(init_guess),))
#     #         jobs.append(p)
#     #         p.start()

#     #     for proc in jobs:
#     #         proc.join()

#     # # # Fourth Set of Data 

#     numSim = 200
#     sim_4 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0.2, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/200, model_type =0, seed = np.random.randint(0,300))  

#     init_guess = (3,0.2,1.2)
#     for i in range(int(numSim/5)):
#         jobs = []
#         test_data_4 = []
#         mC_4 = []
#         for proc in range(4):
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

####################################################################
#       First 175 Time Periods (identical to Lux)
####################################################################
    from data_reader import data_reader

    data = data_reader(time_start= 0, time_end= 175)
    zew = data.zew()/100
    zew_fw = zew[1:]
    ip = data.industrial_production()
    numSim = 20
    
    
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(ip)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
   
###########################################
# Model with exogenous N
###########################################

    # init_guess = (0.8,0,1.18)
    # for i in range(int(numSim/20)):
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

# #################################
# #   Model with endogenous N
# #################################

    # init_guess = (0.12,0.09,0.99, 22)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     data = []
    #     mC_2 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         data.append(zew)
    #         mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(data[proc], y = ip, multiprocess= False,model_type=1), real_data= True))
    #         p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

#################################
# Model with industrial production
#################################

    # init_guess = (0.12,0.09,0.99, 22, -4.5)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     data = []
    #     mC_2 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         data.append(zew)
    #         mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(data[proc], y = ip, multiprocess= False,model_type=2), real_data= True))
    #         p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

#########################################################
# Model with industrial production and laged time series
########################################################

    # init_guess = (0.12,0.09,0.99, 22, -4.5, 3)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_4 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=3), real_data= True))
    #         p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

    
########################################################
# Model with laged time series
#########################################################

    init_guess = (0.12,0.09,0.99, 22, 3)
    for i in range(int(numSim/20)):
        jobs = []
        mC_4 = []
        for proc in range(1):
            # Simulate the time series:      
            mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=4), real_data= True))
            p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

####################################################################
#      Time Periods (176:END) 
####################################################################
    from data_reader import data_reader

    data = data_reader(time_start= 176, time_end= -1)
    zew = data.zew()/100
    zew_fw = zew[1:]
    ip = data.industrial_production()
    numSim = 20

    
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(ip)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
   
###########################################
# Model with exogenous N
###########################################

    # init_guess = (0.8,0,1.18)
    # for i in range(int(numSim/20)):
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

# #################################
# #   Model with endogenous N
# #################################

    # init_guess = (0.12,0.09,0.99, 22)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     data = []
    #     mC_2 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         data.append(zew)
    #         mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(data[proc], y = ip, multiprocess= False,model_type=1), real_data= True))
    #         p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

#################################
# Model with industrial production
#################################

    # init_guess = (0.12,0.09,0.99, 22, -4.5)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     data = []
    #     mC_2 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         data.append(zew)
    #         mC_2.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(data[proc], y = ip, multiprocess= False,model_type=2), real_data= True))
    #         p = mp.Process(target=mC_2[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

#########################################################
# Model with industrial production and laged time series
########################################################

    # init_guess = (0.12,0.09,0.99, 22, -4.5, 3)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_4 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=3), real_data= True))
    #         p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

    
########################################################
# Model with laged time series
#########################################################

    init_guess = (0.09,0.13,0.91, 33, 2.1)
    for i in range(int(numSim/20)):
        jobs = []
        mC_4 = []
        for proc in range(1):
            # Simulate the time series:      
            mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=4), real_data= True))
            p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()