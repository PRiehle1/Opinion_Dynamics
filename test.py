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
from scipy import interpolate, integrate
from data_reader import data_reader
from data_reader import data_reader

# data = data_reader(time_period=175)
# zew = data.zew()/100
# ip = data.industrial_production()


# data = []
# for _ in range(1):
#     sim_2= sim.Simulation(N = 19.23, T =1 , nu = 0.13  , alpha0 = 0.09, alpha1 = 0.93 ,alpha2 = -4.55 ,alpha3 = None, y =  ip, deltax = 0.02, deltat = 1/16, model_type =2, seed = 150)  
#     sim_3= sim.Simulation(N = 15.62, T =1 , nu = 0.09644127859916351575  , alpha0 = 0.2237817, alpha1 = 0.900 ,alpha2 = -7.2 ,alpha3 = None, y =  ip, deltax = 0.02, deltat = 1/16, model_type =2, seed = 150)  
#     test_data_1 = sim_3.simulation(-0.59, sim_length = 175)

#     test_data_2 = sim_2.simulation(-0.59, sim_length = 175)
#     data.append(test_data_1)
#     data.append(test_data_2)
#     plt.plot(zew, color = 'red')
#     plt.plot(test_data_1, color = "black")
#     plt.plot(test_data_2, color = "green")
#     plt.show()

# #######################################################################################################################################
# #  Plot the transitional Density 
# ############################################################
# # ###########################################################################


test = OpinionFormation(N = 50, T =1 , nu = 3, alpha0 = 0.1, alpha1 = 1.2,alpha2 = None,alpha3 = None,deltax= 0.02, deltat= 1/16, model_type= 0)    #
test_1 =OpinionFormation(N = 50, T =1, nu = 6, alpha0= 0.1, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.02, deltat= 1/16, model_type= 0) 


area,prob,prob_end = test.CrankNicolson(x_0 = (-0.56), y = 1, calc_dens = True, converged= False,fast_comp = False)
plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
plot_0.surface_plot()

area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = (-0.56), y= 1,calc_dens = True, converged= False, fast_comp = False)
plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
plot_1.surface_plot()

# plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
# plot_2.surface_plot()
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




# # # #######################################################################################################################################
# # # #  Model Type 3 Test
# # # #######################################################################################################################################
# from data_reader import data_reader

# data = data_reader(time_period=175)
# zew = data.zew()/100
# ip = data.industrial_production()
# sim_3= sim.Simulation(N = 15.62, T =1 , nu = 0.09644127859916351575  , alpha0 = 0.2237817, alpha1 = 0.900 ,alpha2 = -7.2 ,alpha3 = None, y =  ip, deltax = 0.02, deltat = 1/100, model_type =2, seed = 150)  
# test_data_3 = sim_3.simulation(-0.59, sim_length = 175)


# plt.plot(test_data_3)
# plt.plot(zew)
# plt.show()

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

# # from statsmodels.tsa.stattools import adfuller

# # result = adfuller(zew)
# # print('ADF Statistic: %f' % result[0])
# # print('p-value: %f' % result[1])
# # print('Critical Values:')
# # for key, value in result[4].items():
# # 	print('\t%s: %.3f' % (key, value))