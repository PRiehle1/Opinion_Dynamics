# Class for Monte Carlo Estimation of the Time Series: 
import model
import estimation
import numpy as np
import multiprocessing
import pandas as pd


class MonteCarlo(object):

    def __init__(self, numSim: int, model: object, estimation: object, parallel: bool) -> None:
        self.numSim = numSim
        self.model = model
        self.estimation = estimation
        self.parallel = parallel
  

    def estim(self, init_guess) -> np.array:

        res = self.estimation.solver_BFGS(init_guess)
        estim_array = res.x
        logL_array = res.fun

        estim =np.genfromtxt(r"C:\Users\phill\OneDrive\Desktop\Master_Thesis\Opinion_Dynamics\Opinion_Dynamics\Estimation\estimates.csv", delimiter=',')
        estim_array = np.np.vstack([estim, estim_array])
        np.savetxt(r"C:\Users\phill\OneDrive\Desktop\Master_Thesis\Opinion_Dynamics\Opinion_Dynamics\Estimation\estimates.csv", estim_array, delimiter=",")

        logL =np.genfromtxt(r"C:\Users\phill\OneDrive\Desktop\Master_Thesis\Opinion_Dynamics\Opinion_Dynamics\Estimation\logL_array.csv", delimiter=',')
        logL_array = np.np.vstack([logL, logL_array] )
        np.savetxt(r"C:\Users\phill\OneDrive\Desktop\Master_Thesis\Opinion_Dynamics\Opinion_Dynamics\Estimation\logL_array.csv", logL_array, delimiter=",")
       
        in_est =np.genfromtxt(r"C:\Users\phill\OneDrive\Desktop\Master_Thesis\Opinion_Dynamics\Opinion_Dynamics\Estimation\initial_estim.csv", delimiter=',')
        init_guess = np.np.vstack([in_est, init_guess] )
        np.savetxt(r"C:\Users\phill\OneDrive\Desktop\Master_Thesis\Opinion_Dynamics\Opinion_Dynamics\Estimation\initial_estim.csv", init_guess, delimiter=",")



    def run(self) -> np.array:

        if self.parallel == False:
            for i in range(self.numSim):
                init_guess = (0.15 + np.random.normal(0, 0.1, 1), 0.09  + np.random.normal(0, 0.05, 1), 0.99 + np.random.normal(0, 0.2, 1), 21 + np.random.normal(0, 8, 1))
                self.estim(tuple(init_guess))
        else: 

             for i in range(int(self.numSim/2)):
                 jobs = []
                 for d in range(2):
                     init_guess = (0.15 + np.random.normal(0, 0.1, 1), 0.09  + np.random.normal(0, 0.05, 1), 0.99 + np.random.normal(0, 0.2, 1), 21 + np.random.normal(0, 8, 1))
                     p = multiprocessing.Process(target=self.estim, args= (tuple(init_guess),))
                     jobs.append(p)
                     p.start()

                 for proc in jobs:
                     proc.join()

if __name__ == '__main__':
    # Load the Training Data 
    import pandas as pd
    from sympy import *
    import multiprocessing
    import time
	

    training_data_x = pd.read_excel("zew.xlsx", header=None)
    X_train= training_data_x[1].to_numpy()
    X_train= X_train[~np.isnan(X_train)]

    # Set up the Monte Carlo Simulation
    start = time.time()
    mC = MonteCarlo(numSim= 20, model = model.OpinionFormation , estimation= estimation.Estimation(X_train, multiprocess= False), parallel= False)

    mC.run()
    end = time.time()
    print("The Estimation took:")
    print(end - start)

    # simulation = sim.Simulation(N = 21, T = 200, nu = 0.15 , alpha0 = 0.09, alpha1 = 0.99, deltax = 0.02, deltat = 1/16, seed = 150)
    # d = simulation.eulermm(-0.59)

    # est = Estimation(d, multiprocess= False)
    # #est.solver_BFGS((0.15, 0.09, 0.99, 21))
    # bet = est.bhhh((0.2, 0.3, 0.45, 40), tolerance_level= 0.00001, max_iter = 10000)
    #[ 0.04988776  0.09207874  0.99798172 21.0035398 ]
    #-654.5019808289726
