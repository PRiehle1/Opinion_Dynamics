import model
import plot 
import sim
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import random

#######################################################################################################################################
#  Plot the transitional Density 
#######################################################################################################################################


test = model.OpinionFormation(N = 158.44701634, T = 10, nu = 0.84119542, alpha0 = 0.16001432, alpha1 = 0.77629832, deltax = 0.01, deltat = 1/16) #

test_1 = model.OpinionFormation(N = 158.44701634, T = 20, nu = 0.84119542, alpha0 = 0.16001432, alpha1 = 0.77629832, deltax = 0.01, deltat = 1/16)
test_2 = model.OpinionFormation(N = 158.44701634, T = 20, nu = 0.84119542, alpha0 = 0.16001432, alpha1 = 0.77629832, deltax = 0.01, deltat = 1/16)

prob,prob_end = test.CrankNicolson(x_0 = 0,calc_dens= False, converged= False)
prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = -0.4, converged= False)
prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = 0, converged= False)


plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
plot_0.surface_plot()

plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
plot_1.surface_plot()

plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
plot_2.surface_plot()

#plt.plot(prob_end)
#plt.plot(prob_end_1)
#plt.plot(prob_end_2)
#plt.show()
#######################################################################################################################################
# Generate Pseudo Time Series
#######################################################################################################################################
  

simulation = sim.Simulation(N = 21, T = 20, nu = 0.13039116 , alpha0 = 0.00195546, alpha1 = 1.13044364, deltax = 0.0025, deltat = 1/128, seed = 150)
d = simulation.eulermm(-0.59)

set = d.tolist()

# plot_1 = plot.Plotting2D(np.arange(0, simulation.T, 1), d)
# plot_1.sim_plot()




#######################################################################################################################################
# Test Multiprocessing
#######################################################################################################################################

import multiprocessing as mp
test = model.OpinionFormation(N = 175, T = 10, nu = 3, alpha0 = 0.0, alpha1 = 1.2, deltax = 0.002, deltat = 1/16)
    
# if __name__ == '__main__':   


#     simulation = sim.Simulation(N = 21, T = 20, nu = 0.13039116 , alpha0 = 0.00195546, alpha1 = 1.13044364, deltax = 0.0025, deltat = 1/128, seed = 150)
#     d = simulation.eulermm(-0.59)

#     logf = np.zeros(len(d))

#     set = d.tolist()
    
#     pool = mp.Pool(mp.cpu_count())
#     pdf = pool.map(test.CrankNicolson, set)
#     pool.close()  

#     pdf = np.array(pdf)
#     print(np.array(pdf))

#     for elem in range(len(pdf)-1):
#         for x in range(len(test.x)):
#             if test.x[x] == np.around(d[elem+1],3):
#                 logf[elem] = (-1)* np.log(pdf[elem,x])
#     logL = np.sum(logf)
#     print("The Log Likelihodd is: " + str(logL)) 
