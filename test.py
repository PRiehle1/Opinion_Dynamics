from model import OpinionFormation
import plot 
import sim
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import random
import montecarlo
import estimation
import pandas as pd 
## CDF 




# #######################################################################################################################################
# #  Plot the transitional Density 
# #######################################################################################################################################


test = OpinionFormation(N = 175, T = 3, nu = 0.8, alpha0= 0.01, alpha1= 1.1901, alpha2 = None,alpha3 = None,deltax= 0.01, deltat= 1/300, model_type= 0)    #
test_1 =OpinionFormation(N = 175, T = 3, nu = 0.08, alpha0= 0.01, alpha1= 1.19, alpha2 = None,alpha3 = None, deltax= 0.01, deltat= 1/300, model_type= 0) 


area,prob,prob_end = test.CrankNicolson(x_0 = (0), y = 1, check_stability = False, calc_dens = True, converged =  False, fast_comp = False)
plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
plot_0.surface_plot()

area_1,prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = (0), y= 1, check_stability = False, calc_dens = True, converged =  False, fast_comp = False)
# # plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# # plot_1.surface_plot()

# # plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
# # plot_2.surface_plot()
plt.figure(figsize=(7, 6))
plt.plot(test.t,area, color='blue',
            label='Area Test 1')
plt.plot(test_1.t,area_1, color='red',
            label='Area Test 2')
plt.legend(loc='lower right')
plt.title("Area under the PDF")
plt.xlabel("Time")
plt.ylabel("Area")
plt.show()


plt.figure(figsize=(7, 6))
plt.plot(test.x,prob_end, color='blue',
            label='PDF Test 1')
plt.plot(test_1.x,prob_end_1, color='red',
            label='PDF Test 2')
plt.legend(loc='lower right')
plt.title("Final PDF after T")
plt.xlabel("Time")
plt.ylabel("Density")
plt.show()


# #######################################################################################################################################
# #  Model Type 3 Test
# #######################################################################################################################################
# # from data_reader import data_reader

# # data = data_reader(time_period= 175)

# # zew = data.zew()/100
# # ip = data.industrial_production()

# # model_1 = model.OpinionFormation(N = 50, T = 3, nu = 3 , alpha0 = 0., alpha1 = 1.2 ,alpha2 = None,alpha3 = None, y = None, deltax = 0.0025, deltat = 1/16, model_type =0)
# # prob_end = model_1.CrankNicolson(x_0 = 0.3,check_stability = True, calc_dens = False, converged =  False, fast_comp = True)
# # plt.plot(prob_end)
# # plt.show()


# # # Simulated data
# # #Real Data 
# from data_reader import data_reader

# data = data_reader(time_period= 360)
# zew = data.zew()/100
# ip = data.industrial_production()
# sim_3= sim.Simulation(N = 19.23, T = 30, nu = 0.13  , alpha0 = 0.09, alpha1 = 0.93 ,alpha2 = (-4.55) ,alpha3 = None, y =  ip, deltax = 0.01, deltat = 1/300, model_type =2, seed = 150)  
# test_data_3 = sim_3.simulation(-0.59, sim_length = 360)
# # plt.plot(test_data_3)
# # plt.plot(zew)
# # plt.show()

# plt.figure(figsize=(7, 6))
# plt.plot(test_data_3, color='blue',
#             label='Simulated Data')
# plt.plot(zew, color='red',
#             label='ZEW')
# plt.legend(loc='lower right')
# plt.title("Value")
# plt.xlabel("Time")
# plt.ylabel("Density")
# plt.show()
