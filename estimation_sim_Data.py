# Import Packages
import pandas as pd
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import montecarlo
import estimation
import sim 

# First Set of Data 

# Simulated data
sim = sim.Simulation(N = 50, T = 20, nu = 3 , alpha0 = 0, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 0.01, model_type =0, seed = 3)  
test_data = sim.simulation(0, sim_length = 200)
plt.plot(test_data)
plt.show()

# Set up the Monte Carlo Estimation
mC = montecarlo.MonteCarlo(numSim= 40 , estimation= estimation.Estimation(test_data, multiprocess= False, model_type= 0), parallel= False, real_data = False)
mC.run()

# Second Set of Data 

# Simulated data
sim = sim.Simulation(N = 50, T = 20, nu = 3 , alpha0 = 0.2, alpha1 = 0.8,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 0.01, model_type =0, seed = 150)  
test_data = sim.simulation(0, sim_length = 200)
plt.plot(test_data)
plt.show()

# Set up the Monte Carlo Estimation
mC = montecarlo.MonteCarlo(numSim= 40 , estimation= estimation.Estimation(test_data, multiprocess= False, model_type= 0), parallel= False, real_data = False)
mC.run()

# Third Set of Data 

# Simulated data
sim = sim.Simulation(N = 50, T = 20, nu = 3 , alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 0.01, model_type =0, seed = 150)  
test_data = sim.simulation(0, sim_length = 20)
plt.plot(test_data)
plt.show()

# Set up the Monte Carlo Estimation
mC = montecarlo.MonteCarlo(numSim= 40 , estimation= estimation.Estimation(test_data, multiprocess= False, model_type= 0), parallel= False, real_data = False)
mC.run()

# Fourth Set of Data 

# Simulated data
sim = sim.Simulation(N = 50, T = 20, nu = 3 , alpha0 = 0.2, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 0.01, model_type =0, seed = 150)  
test_data = sim.simulation(0, sim_length = 200)
plt.plot(test_data)
plt.show()

# Set up the Monte Carlo Estimation
mC = montecarlo.MonteCarlo(numSim= 40 , estimation= estimation.Estimation(test_data, multiprocess= False, model_type= 0), parallel= False, real_data = False)
mC.run()

