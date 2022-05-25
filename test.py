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


#test = model.OpinionFormation(N = 50, T = 30, nu = 1, alpha0 = 0.00, alpha1 = 1.2, deltax = 0.001, deltat = 0.01) #
#test_1 = model.OpinionFormation(N = 50, T = 30, nu = 2, alpha0 = 0.00, alpha1 = 1.2, deltax = 0.001, deltat = 0.01)
test_2 = model.OpinionFormation(N = 175, T = 3, nu = 1, alpha0 = 0.1, alpha1 = 0.99, deltax = 0.002, deltat = 1/16)

#_,prob_end = test.CrankNicolson(x_0 = 0, check_stability=False, calc_dens= False)
#_,prob_end_1 = test_1.CrankNicolson(x_0 = 0, check_stability=False, calc_dens= False)
area, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = -0.59, check_stability=True, calc_dens= True)
print(prob_2.min())
plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
plot_2.surface_plot()

#plt.plot(prob_end)
#plt.plot(prob_end_1)
#plt.plot(prob_end_2)
#plt.show()
#######################################################################################################################################
# Generate Pseudo Time Series
#######################################################################################################################################
  

simulation = sim.simulateModel(N = 21, T = 360, nu = 0.13039116 , alpha0 = 0.00195546, alpha1 = 1.13044364, deltax = 0.0025, deltat = 1/128, seed = 150)
d = simulation.eulermm(-0.59)

plot_1 = plot.Plotting2D(np.arange(0, simulation.T, 1), d)
plot_1.sim_plot()


################## Test Diffusion ##########

dif = np.zeros(len(test_2.x))
numRun = 0
for elem in test_2.x:
    dif[numRun] = test_2.diffusion(elem)
    numRun += 1

plot_3 = plot.Plotting2D(test_2.x, dif)
plot_3.sim_plot()



