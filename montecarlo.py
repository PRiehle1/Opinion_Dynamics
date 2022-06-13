# Class for Monte Carlo Estimation of the Time Series: 
import model
import estimation
import multiprocessing as mp 
from tqdm import tqdm 
import numpy as np
import time 

class MonteCarlo(object):

    def __init__(self, numSim: int, model: object, estimation: object) -> None:
        self.numSim = numSim
        self.model = model
        self.estimation = estimation
  
    def run(self, multiprocess: bool) -> np.array:
        # Measure the Time
        start = time.time()
        # Initialize the Matrix of estimates

        estim_array = np.zeros([self.numSim, 4]) 
        initial_estim = np.zeros([self.numSim, 4]) 
        logL_array = np.zeros(self.numSim)

        if multiprocess == True: 
            # array of initial guesses based on the estimates of the Lux 2009 Paper
             
            init_guess = np.zeros([self.numSim, 4])
            for i in range(self.numSim):
                init_guess[i,0] = 0.15 + np.random.normal(0, 0.1, 1)
                init_guess[i,1] = 0.09  + np.random.normal(0, 0.05, 1)
                init_guess[i,2] = 0.99 + np.random.normal(0, 0.2, 1)
                init_guess[i,3] = 21 + np.random.normal(0, 8, 1)
            
            # Transform the array to a list 
            init_guess_list = init_guess.tolist()
            
            # Multiprocessing 
            
            pool = mp.Pool(mp.cpu_count())

            # Estimate the Parameters for the list of initial guesses:
            res = list(tqdm(pool.map(self.estimation.solver_BFGS, init_guess_list)))

            # Close the Pool of Workers
            pool.close() 

            end = time.time()
            print(end-start)

            return res

        else:

            for sim in tqdm(range(self.numSim)):
                init_guess = (0.15 + np.random.normal(0, 0.1, 1), 0.09  + np.random.normal(0, 0.05, 1), 0.99 + np.random.normal(0, 0.2, 1), 21 + np.random.normal(0, 8, 1))
                res = self.estimation.solver_BFGS(init_guess)
                initial_estim[sim,: ] = init_guess
                estim_array[sim, :] = res.x
                logL_array[sim] = res.fun
            end = time.time()
            print(end-start)
            return estim_array, logL_array, initial_estim

if __name__ == '__main__':
    import pandas as pd
    import sim
    
    from sympy import *

    training_data_x = pd.read_excel("zew.xlsx", header=None)
    X_train= training_data_x[1].to_numpy()
    X_train= X_train[~np.isnan(X_train)]

    mC = MonteCarlo(numSim= 10, model = model.OpinionFormation , estimation= estimation.Estimation(X_train, multiprocess= False))
    estim_array, logL_array, initial_estim =  mC.run(multiprocess= False)
    #res = mC.run(multiprocess= True)
    np.save('estimates.npy', estim_array)
    np.save('logL_array.npy', logL_array)
    np.save('initial_estim.npy', initial_estim)


    # simulation = sim.Simulation(N = 21, T = 200, nu = 0.15 , alpha0 = 0.09, alpha1 = 0.99, deltax = 0.02, deltat = 1/16, seed = 150)
    # d = simulation.eulermm(-0.59)

    # est = Estimation(d, multiprocess= False)
    # #est.solver_BFGS((0.15, 0.09, 0.99, 21))
    # bet = est.bhhh((0.2, 0.3, 0.45, 40), tolerance_level= 0.00001, max_iter = 10000)
#[ 0.04988776  0.09207874  0.99798172 21.0035398 ]
#-654.5019808289726