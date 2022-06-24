import model
import plot 
import sim
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import random
import montecarlo
import estimation
import pandas as pd 

#######################################################################################################################################
#  Plot the transitional Density 
#######################################################################################################################################


test = model.OpinionFormation(N = 175, T = 80, nu = 0.83 , alpha0 = 0.0, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/16, model_type =0) #

test_1 = model.OpinionFormation(N = 175, T = 800, nu = 0.83 , alpha0 = 0.0, alpha1 = 1.2,alpha2 = None,alpha3 = None, y = None, deltax = 0.01, deltat = 1/16, model_type =0) 
#test_1 = model.OpinionFormation(N = 175, T = 4, nu = 0.8 , alpha0 = 0.01, alpha1 = 1.19,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/16, model_type =0)
#test_2 = model.OpinionFormation(N = 175, T = 4, nu = 0.8 , alpha0 = 0.01, alpha1 = 1.19,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 1/16, model_type =0)

prob_end = test.CrankNicolson(x_0 = 0.9,check_stability = True, calc_dens = False, converged =  False, fast_comp = True)
prob_end_1 = test_1.CrankNicolson(x_0 = -0.9,check_stability = False, calc_dens = False, converged =  False, fast_comp = True)
#prob_end_2 = test_2.CrankNicolson(x_0 = 0,check_stability = False, calc_dens = False, converged =  False, fast_comp = True)


# plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
# plot_0.surface_plot()

# plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# plot_1.surface_plot()

# plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
# plot_2.surface_plot()

plt.plot(prob_end)
plt.plot(prob_end_1)
#plt.plot(prob_end_2)
plt.show()




#######################################################################################################################################
#  BHHH Estimation Test
#######################################################################################################################################

#Simulated data
sim_0 = sim.Simulation(N = 175, T = 10, nu = 0.78 , alpha0 = 0.01, alpha1 = 1.19,alpha2 = None,alpha3 = None, y = None, deltax = 0.02, deltat = 0.01, model_type =0, seed = 3)  
test_data_0 = sim_0.simulation(-0.59, sim_length = 200)
plt.plot(test_data_0)
plt.show()

# mC = montecarlo.MonteCarlo(numSim= 1, estimation= estimation.Estimation(test_data_0, multiprocess= False, model_type= 0), parallel= False, real_data = False)
# mC.run()



# training_data_x = pd.read_excel("zew.xlsx", header=None)
# X_train= training_data_x[1].to_numpy()
# X_train= X_train[~np.isnan(X_train)]
# plt.plot(X_train)
# plt.show()

# mC = montecarlo.MonteCarlo(numSim= 1, estimation= estimation.Estimation(X_train, parallel= True, model_type= 0), parallel= False, real_data = True)
# mC.run()

