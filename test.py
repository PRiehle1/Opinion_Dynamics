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


test = model.OpinionFormation(N = 175, T = 10, nu = 3, alpha0 = 0.0, alpha1 = 1.2, deltax = 0.002, deltat = 1/16) #
test_1 = model.OpinionFormation(N = 175, T = 10, nu = 3, alpha0 = 0.0, alpha1 = 1.2, deltax = 0.002, deltat = 1/16)
test_2 = model.OpinionFormation(N = 175, T = 10, nu = 3, alpha0 = 0.0, alpha1 = 1.2, deltax = 0.002, deltat = 1/16)

prob,prob_end = test.CrankNicolson(x_0 = -0.1, check_stability=False, calc_dens= False)
prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = 0.1, check_stability=False, calc_dens= False)
area, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = 0, check_stability=True, calc_dens= True)


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



