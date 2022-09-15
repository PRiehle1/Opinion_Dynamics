# Class for Monte Carlo Estimation of the Time Series: 
from ipaddress import ip_interface
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
        res = self.estimation.solver_L_BFGS_B(init_guess)
        
        if self.real_data == True:
            opg_res = self.estimation.outer_product_gradient(res.x, eps = 0.000001)
        
        estim_array = res.x
        logL_array = res.fun
        hes = res.hess_inv.todense() 

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

                if os.path.exists("Estimation/Model_0/var_model_0_hes.csv") == False:
                    np.savetxt("Estimation/Model_0/var_model_0_hes.csv", [0,0,0], delimiter=",")
                hes_old =np.genfromtxt("Estimation/Model_0/var_model_0_hes.csv", delimiter=',')
                hes_array = np.vstack([hes_old, hes.diagonal()])
                np.savetxt("Estimation/Model_0/var_model_0_hes.csv", hes_array, delimiter=",")

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

                if os.path.exists("Estimation/Model_1/var_model_1_hes.csv") == False:
                    np.savetxt("Estimation/Model_1/var_model_1_hes.csv", [0,0,0,0], delimiter=",")
                hes_old =np.genfromtxt("Estimation/Model_1/var_model_1_hes.csv", delimiter=',')
                hes_array = np.vstack([hes_old, hes.diagonal()])
                np.savetxt("Estimation/Model_1/var_model_1_hes.csv", hes_array, delimiter=",")

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

                if os.path.exists("Estimation/Model_2/var_model_2_hes.csv") == False:
                    np.savetxt("Estimation/Model_2/var_model_2_hes.csv", [0,0,0,0,0], delimiter=",")
                hes_old =np.genfromtxt("Estimation/Model_2/var_model_2_hes.csv", delimiter=',')
                hes_array = np.vstack([hes_old, hes.diagonal()])
                np.savetxt("Estimation/Model_2/var_model_2_hes.csv", hes_array, delimiter=",")

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

                if os.path.exists("Estimation/Model_3/var_model_3_hes.csv") == False:
                    np.savetxt("Estimation/Model_3/var_model_3_hes.csv", [0,0,0,0,0,0], delimiter=",")
                hes_old =np.genfromtxt("Estimation/Model_3/var_model_3_hes.csv", delimiter=',')
                hes_array = np.vstack([hes_old, hes.diagonal()])
                np.savetxt("Estimation/Model_3/var_model_3_hes.csv", hes_array, delimiter=",")


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

                if os.path.exists("Estimation/Model_4/var_model_4_hes.csv") == False:
                    np.savetxt("Estimation/Model_4/var_model_4_hes.csv", [0,0,0,0,0], delimiter=",")
                hes_old =np.genfromtxt("Estimation/Model_4/var_model_4_hes.csv", delimiter=',')
                hes_array = np.vstack([hes_old, hes.diagonal()])
                np.savetxt("Estimation/Model_4/var_model_4_hes.csv", hes_array, delimiter=",")
            
            
            elif self.estimation.model_type == 5: 
                if os.path.exists("Estimation/Model_5/estimates_model_5.csv") == False:
                    np.savetxt("Estimation/Model_5/estimates_model_5.csv", [0,0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/Model_5/estimates_model_5.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/Model_5/estimates_model_5.csv", estim_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_5/logL_array_model_5.csv") == False:
                    np.savetxt("Estimation/Model_5/logL_array_model_5.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/Model_5/logL_array_model_5.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/Model_5/logL_array_model_5.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_5/initial_estim_model_5.csv") == False:
                    np.savetxt("Estimation/Model_5/initial_estim_model_5.csv", [0,0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/Model_5/initial_estim_model_5.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/Model_5/initial_estim_model_5.csv", init_guess, delimiter=",")

                if os.path.exists("Estimation/Model_5/var_model_5.csv") == False:
                    np.savetxt("Estimation/Model_5/var_model_5.csv", [0,0,0,0], delimiter=",")
                opg_old =np.genfromtxt("Estimation/Model_5/var_model_5.csv", delimiter=',')
                opg_array = np.vstack([opg_old, opg_res.diagonal()])
                np.savetxt("Estimation/Model_5/var_model_5.csv", opg_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_5/var_model_5_hes.csv") == False:
                    np.savetxt("Estimation/Model_5/var_model_5_hes.csv", [0,0,0,0], delimiter=",")
                hes_old =np.genfromtxt("Estimation/Model_5/var_model_5_hes.csv", delimiter=',')
                hes_array = np.vstack([hes_old, hes.diagonal()])
                np.savetxt("Estimation/Model_5/var_model_5_hes.csv", hes_array, delimiter=",")

            elif self.estimation.model_type == 6: 
                if os.path.exists("Estimation/Model_6/estimates_model_6.csv") == False:
                    np.savetxt("Estimation/Model_6/estimates_model_6.csv", [0,0,0,0,0], delimiter=",")
                estim_old =np.genfromtxt("Estimation/Model_6/estimates_model_6.csv", delimiter=',')    
                estim_array = np.vstack([estim_old, estim_array])
                np.savetxt("Estimation/Model_6/estimates_model_6.csv", estim_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_6/logL_array_model_6.csv") == False:
                    np.savetxt("Estimation/Model_6/logL_array_model_6.csv", [0], delimiter=",")
                logL =np.genfromtxt("Estimation/Model_6/logL_array_model_6.csv", delimiter=',')
                logL_array = np.append(logL, logL_array)
                np.savetxt("Estimation/Model_6/logL_array_model_6.csv", logL_array, delimiter=",")
                
                if os.path.exists("Estimation/Model_6/initial_estim_model_6.csv") == False:
                    np.savetxt("Estimation/Model_6/initial_estim_model_6.csv", [0,0,0,0,0], delimiter=",")
                in_est =np.genfromtxt("Estimation/Model_6/initial_estim_model_6.csv", delimiter=',')
                init_guess = np.vstack([in_est, np.block(list(init_guess))])
                np.savetxt("Estimation/Model_6/initial_estim_model_6.csv", init_guess, delimiter=",")

                if os.path.exists("Estimation/Model_6/var_model_6.csv") == False:
                    np.savetxt("Estimation/Model_6/var_model_6.csv", [0,0,0,0,0], delimiter=",")
                opg_old =np.genfromtxt("Estimation/Model_6/var_model_6.csv", delimiter=',')
                opg_array = np.vstack([opg_old, opg_res.diagonal()])
                np.savetxt("Estimation/Model_6/var_model_6.csv", opg_array, delimiter=",")

                if os.path.exists("Estimation/Model_6/var_model_6_hes.csv") == False:
                    np.savetxt("Estimation/Model_6/var_model_6_hes.csv", [0,0,0,0,0], delimiter=",")
                hes_old =np.genfromtxt("Estimation/Model_6/var_model_6_hes.csv", delimiter=',')
                hes_array = np.vstack([hes_old, hes.diagonal()])
                np.savetxt("Estimation/Model_6/var_model_6_hes.csv", hes_array, delimiter=",")
        
        elif self.real_data == False: 

            if self.estimation.model_type == 0:
                if init_guess == [3,0,0.8]:
                    if os.path.exists("Estimation/sim_Data/exoN/estimates_model_0_set1.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set1.csv", [0,0,0], delimiter=",")
                    estim_old =np.genfromtxt("Estimation/sim_Data/exoN/estimates_model_0_set1.csv", delimiter=',')    
                    estim_array = np.vstack([estim_old, estim_array])
                    np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set1.csv", estim_array, delimiter=",")

                    if os.path.exists("Estimation/sim_Data/exoN/logL_array_model_0_set1.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set1.csv", [0], delimiter=",")
                    logL =np.genfromtxt("Estimation/sim_Data/exoN/logL_array_model_0_set1.csv", delimiter=',')
                    logL_array = np.append(logL, logL_array)
                    np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set1.csv", logL_array, delimiter=",")
                    
                    if os.path.exists("Estimation/sim_Data/exoN/initial_estim_model_0_set1.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set1.csv", [0,0,0], delimiter=",")
                    in_est =np.genfromtxt("Estimation/sim_Data/exoN/initial_estim_model_0_set1.csv", delimiter=',')
                    init_guess = np.vstack([in_est, np.block(list(init_guess))])
                    np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set1.csv", init_guess, delimiter=",")
                elif init_guess == [3,0.08,0.8]:
                    if os.path.exists("Estimation/sim_Data/exoN/estimates_model_0_set2.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set2.csv", [0,0,0], delimiter=",")
                    estim_old =np.genfromtxt("Estimation/sim_Data/exoN/estimates_model_0_set2.csv", delimiter=',')    
                    estim_array = np.vstack([estim_old, estim_array])
                    np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set2.csv", estim_array, delimiter=",")

                    if os.path.exists("Estimation/sim_Data/exoN/logL_array_model_0_set2.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set2.csv", [0], delimiter=",")
                    logL =np.genfromtxt("Estimation/sim_Data/exoN/logL_array_model_0_set2.csv", delimiter=',')
                    logL_array = np.append(logL, logL_array)
                    np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set2.csv", logL_array, delimiter=",")
                    
                    if os.path.exists("Estimation/sim_Data/exoN/initial_estim_model_0_set2.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set2.csv", [0,0,0], delimiter=",")
                    in_est =np.genfromtxt("Estimation/sim_Data/exoN/initial_estim_model_0_set2.csv", delimiter=',')
                    init_guess = np.vstack([in_est, np.block(list(init_guess))])
                    np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set2.csv", init_guess, delimiter=",")
                elif init_guess == [3,0,1.2]:
                    if os.path.exists("Estimation/sim_Data/exoN/estimates_model_0_set3.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set3.csv", [0,0,0], delimiter=",")
                    estim_old =np.genfromtxt("Estimation/sim_Data/exoN/estimates_model_0_set3.csv", delimiter=',')    
                    estim_array = np.vstack([estim_old, estim_array])
                    np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set3.csv", estim_array, delimiter=",")

                    if os.path.exists("Estimation/sim_Data/exoN/logL_array_model_0_set3.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set3.csv", [0], delimiter=",")
                    logL =np.genfromtxt("Estimation/sim_Data/exoN/logL_array_model_0_set3.csv", delimiter=',')
                    logL_array = np.append(logL, logL_array)
                    np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set3.csv", logL_array, delimiter=",")
                    
                    if os.path.exists("Estimation/sim_Data/exoN/initial_estim_model_0_set3.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set3.csv", [0,0,0], delimiter=",")
                    in_est =np.genfromtxt("Estimation/sim_Data/exoN/initial_estim_model_0_set3.csv", delimiter=',')
                    init_guess = np.vstack([in_est, np.block(list(init_guess))])
                    np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set3.csv", init_guess, delimiter=",")
                elif init_guess == [3,0.08,1.2]:
                    if os.path.exists("Estimation/sim_Data/exoN/estimates_model_0_set4.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set4.csv", [0,0,0], delimiter=",")
                    estim_old =np.genfromtxt("Estimation/sim_Data/exoN/estimates_model_0_set4.csv", delimiter=',')    
                    estim_array = np.vstack([estim_old, estim_array])
                    np.savetxt("Estimation/sim_Data/exoN/estimates_model_0_set4.csv", estim_array, delimiter=",")

                    if os.path.exists("Estimation/sim_Data/exoN/logL_array_model_0_set4.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set4.csv", [0], delimiter=",")
                    logL =np.genfromtxt("Estimation/sim_Data/exoN/logL_array_model_0_set4.csv", delimiter=',')
                    logL_array = np.append(logL, logL_array)
                    np.savetxt("Estimation/sim_Data/exoN/logL_array_model_0_set4.csv", logL_array, delimiter=",")
                    
                    if os.path.exists("Estimation/sim_Data/exoN/initial_estim_model_0_set4.csv") == False:
                        np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set4.csv", [0,0,0], delimiter=",")
                    in_est =np.genfromtxt("Estimation/sim_Data/exoN/initial_estim_model_0_set4.csv", delimiter=',')
                    init_guess = np.vstack([in_est, np.block(list(init_guess))])
                    np.savetxt("Estimation/sim_Data/exoN/initial_estim_model_0_set4.csv", init_guess, delimiter=",")

                else:
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
#                                                    Simulated Data #Please change Number of agents in estimation.py line 59 to 50
################################################################################################################################################
#   #First Set of Data 
    # numSim = 200
    # sim_1 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/100, model_type =0, seed = 3)  

    # init_guess = (3,0,0.8)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     test_data_1 = []
    #     mC_1 = []
    #     for proc in range(20):
    #         # Simulate the time series:      
    #         test_data_1.append(sim_1.simulation(-0.59, sim_length = 400))
    #         mC_1.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_1[proc],multiprocess= False,model_type=0), real_data= False))
    #         p = mp.Process(target=mC_1[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()
###########################################################################################
    # #Second Set of Data 
    # numSim = 200
    # sim_1 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0.08, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/100, model_type =0, seed = 3)  

    # init_guess = (3,0.08,0.8)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     test_data_1 = []
    #     mC_1 = []
    #     for proc in range(20):
    #         # Simulate the time series:      
    #         test_data_1.append(sim_1.simulation(-0.59, sim_length = 400))
    #         mC_1.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_1[proc],multiprocess= False,model_type=0), real_data= False))
    #         p = mp.Process(target=mC_1[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

# # # # #############################################################    
#     # # #Third Set of Data 

#     numSim = 200
#     sim_3 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/100, model_type =0, seed = 3)  

#     init_guess = (3,0,1.2)
#     for i in range(int(numSim/20)):
#         jobs = []
#         test_data_3 = []
#         mC_3 = []
#         for proc in range(20):
#             # Simulate the time series:      
#             test_data_3.append(sim_3.simulation(-0.59, sim_length = 400))
#             mC_3.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_3[proc],multiprocess= False,model_type=0), real_data= False))
#             p = mp.Process(target=mC_3[proc].run, args= (tuple(init_guess),))
#             jobs.append(p)
#             p.start()

#         for proc in jobs:
#             proc.join()
# #################################################################################################################################################
# #     # # # Fourth Set of Data 

#     numSim = 200
#     sim_4 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0.08, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/100, model_type =0, seed = np.random.randint(0,300))  

#     init_guess = (3,0.08,1.2)
#     for i in range(int(numSim/20)):
#         jobs = []
#         test_data_4 = []
#         mC_4 = []
#         for proc in range(20):
#             # Simulate the time series:      
#             test_data_4.append(sim_4.simulation(-0.59, sim_length = 400))
#             mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(test_data_4[proc],multiprocess= False,model_type=0), real_data= False))
#             p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
#             jobs.append(p)
#             p.start()

#         for proc in jobs:
#             proc.join()



# ################################################################################################################################################
# #                                                   Real Data 
# ################################################################################################################################################

# ####################################################################
# #       First 175 Time Periods (identical to Lux)
# ####################################################################
    # from data_reader import data_reader

    # data = data_reader(time_start= 0, time_end= 175)
    # zew = data.zew()/100
    # zew_fw = zew[1:]
    # ip = data.industrial_production()
    # numSim = 20
    
    
    # from statsmodels.tsa.stattools import adfuller

    # result = adfuller(ip)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
   
###########################################
# Model with exogenous N
###########################################
    #Please change Number of agents in estimation.py line 59 to 175

    # init_guess = (1,0.2,1.4)
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

#################################
#   Model with endogenous N
#################################

    # init_guess = (0.10,0.1,0.40, 40)
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

# #################################
# # Model with industrial production
# #################################

    # init_guess = (0.20,0.00,0.80, 35, -2.5)
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

# #########################################################
# # Model with industrial production and laged time series
# ########################################################

    # init_guess = (0.3,0.00,0.4, 35, -2.5, 1)
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

    
# # ########################################################
# # # Model with laged time series
# # #########################################################

    # init_guess = (0.19728611, 0.03526088,  0.41597744, 53.59484022,  4.18270495)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_4 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=4), real_data= True))
    #         p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

# # ########################################################
# # # Model with laged time series and N = 22
# # #########################################################

    # init_guess = (0.25489732, 0.14130141, 0.49925405, 0.81895132)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_5 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_5.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = None, x_l= zew, multiprocess= False,model_type=5), real_data= True))
    #         p = mp.Process(target=mC_5[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

# #########################################################
# # Model with industrial production and laged time series and N =22
# ########################################################

    # init_guess = (0.25,0.00,0.69, -1.5, 1)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_6 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_6.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=6), real_data= True))
    #         p = mp.Process(target=mC_6[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

# ####################################################################
# #      Time Periods (176:END) 
# ####################################################################
    # from data_reader import data_reader

    # data = data_reader(time_start= 176, time_end= -1)
    # zew = data.zew()/100
    # ip = data.industrial_production()
    # # Account for smaller time Series
    # zew = zew[0:len(ip)]
    # zew_fw = zew[1:]
    # numSim = 20

    # from statsmodels.tsa.stattools import adfuller

    # result = adfuller(ip)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
   
##########################################
#Model with exogenous N
##########################################
   #Please change Number of agents in estimation.py line 59 to 175

    # init_guess = (1,0.2,1.4)
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

# # #################################
# # #   Model with endogenous N
# # #################################

    # init_guess = (0.10,0.05,0.80, 40)
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

# #################################
# # Model with industrial production
# #################################

    # init_guess = (0.40,0.00,0.80, 35, -2.5)
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

# # #########################################################
# # # Model with industrial production and laged time series
# # ########################################################

    # init_guess = (0.10, 0.22, 0.8, 15, -2.5, 1.13)
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

    
# # ########################################################
# # # Model with laged time series
# # #########################################################

    # init_guess = (0.2268104, 0.079795, 0.70,  15.64176628,  1.94470055)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_4 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = None, x_l= zew, multiprocess= False,model_type=4), real_data= True))
    #         p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()
# # ########################################################
# # # Model with laged time series and N = 22
# # #########################################################

    # init_guess = (0.2489732, 0.1130141, 0.9925405, 0.61895132)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_5 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_5.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = None, x_l= zew, multiprocess= False,model_type=5), real_data= True))
    #         p = mp.Process(target=mC_5[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

# #########################################################
# # Model with industrial production and laged time series and N =22
# ########################################################

    # init_guess = (0.172,0.0,0.8, -2.5, 1)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_6 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_6.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=6), real_data= True))
    #         p = mp.Process(target=mC_6[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()


# ####################################################################
# #      Time Periods (START:END) 
# ####################################################################
    # from data_reader import data_reader

    # data = data_reader(time_start= 0, time_end= -1)
    # zew = data.zew()/100
    # ip = data.industrial_production()
    # # Account for smaller time Series
    # zew = zew[0:len(ip)]
    # zew_fw = zew[1:]
    # numSim = 20

    # from statsmodels.tsa.stattools import adfuller

    # result = adfuller(ip)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
   
##########################################
#Model with exogenous N
##########################################
 ##!!!!!!!!!!!!! Please change Number of agents in estimation.py line 59 to 175 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # init_guess = (1,0.2,1.4)
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
#################################
#   Model with endogenous N
#################################
    # init_guess = (0.10,0.05,0.80, 40)
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
# #################################
# # Model with industrial production
# #################################
    # init_guess = (0.20,0.00,0.80, 35, -2.5)
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
# #########################################################
# # Model with industrial production and laged time series
# ########################################################
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
# # ########################################################
# # # Model with laged time series
# # #########################################################
    # init_guess = (0.09,0.13,0.91, 33, 2.1)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_4 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_4.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=4), real_data= True))
    #         p = mp.Process(target=mC_4[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

# # ########################################################
# # # Model with laged time series and N = 22
# # #########################################################

    # init_guess = (0.19489732, 0.10141, 0.9925405, 0.5132)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_5 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_5.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = None, x_l= zew, multiprocess= False,model_type=5), real_data= True))
    #         p = mp.Process(target=mC_5[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()

# ########################################################
# #Model with industrial production and laged time series and N =22
# #######################################################

    # init_guess = (0.2,0.0,0.79, -2.5, 5)
    # for i in range(int(numSim/20)):
    #     jobs = []
    #     mC_6 = []
    #     for proc in range(1):
    #         # Simulate the time series:      
    #         mC_6.append(MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=6), real_data= True))
    #         p = mp.Process(target=mC_6[proc].run, args= (tuple(init_guess),))
    #         jobs.append(p)
    #         p.start()

    #     for proc in jobs:
    #         proc.join()


############################################################################################################################################
#                                              Rolling Window Estimation                                                                                             #                             
############################################################################################################################################
    
# #################################################################
#                   Model 3
# #################################################################    
    
    from data_reader import data_reader
    from scipy.stats import skew, kurtosis, norm
    zew_data = []
    zew_fw_data = []
    ip_data = []

    real_statistics = []

    for i in range(0,20):
        data = data_reader(time_start= 0 + (12*i), time_end= 120 + (12*i))
        zew = data.zew()/100
        ip = data.industrial_production()
        # # Account for smaller time Series
        zew = zew[0:len(ip)]
        zew_fw = zew[1:]
        mean = zew_fw.mean()
        std = zew_fw.std()
        ske = skew(np.array(zew_fw), axis = 0, bias = True)
        kurt = kurtosis(zew_fw,axis = 0, bias = True)  
        rel_dev = (mean**2)/zew_fw.var()     
        zew_data.append(zew)
        zew_fw_data.append(zew_fw)
        ip_data.append(ip)   
        real_statistics.append((mean, std,ske, kurt, rel_dev))
        #mC_3 = MonteCarlo(numSim= 5, model = OpinionFormation , estimation= estimation.Estimation(time_series= zew_fw, y = ip, x_l= zew, multiprocess= False,model_type=3), real_data= True)
        #mC_3.run((9.047996351698969764e-02,1.495954758529130790e-01,8.745193519211922339e-01,3.187844863450407118e+01,-5.184832000945729824e+00,2.125734982220825575e+00))

np.savetxt("Estimation/real_statistics_rolling.csv", real_statistics, delimiter=",",fmt ='% s')
