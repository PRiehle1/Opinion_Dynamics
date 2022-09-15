from ipaddress import ip_network
from time import time
from model import OpinionFormation
import plot 
import sim
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import random
import estimation
import pandas as pd 
from scipy import interpolate, integrate
from data_reader import data_reader
from data_reader import data_reader
# # # #######################################################################################################################################
# from data_reader import data_reader

# data = data_reader(time_start= 0, time_end= -1)
# zew = data.zew()/100
# ip = data.industrial_production()
# # Account for smaller time Series
# zew = zew[0:len(ip)]
# zew_fw = zew[1:]
# numSim = 20

# data = []
# param = [3.603835661125873863e-02,1.465722907414069298e-01,3.363059051870710969e-01,8.038854823426559548e+00,-5.176101086238097615e+00,3.297306859294678372e+00]

# for _ in range(1):
#     sim_1 = sim.Simulation(N = param[3], T = 1, nu = param[0] , alpha0 = param[1], alpha1 = param[2],alpha2 = param[4] ,alpha3 = param[5], y = ip, deltax = 0.01, deltat = 1/100, model_type =3, seed = 300)
#     #sim_2 = sim.Simulation(N = param[3], T = 1, nu = param[0] , alpha0 = param[1], alpha1 = param[2],alpha2 = param[4] ,alpha3 = param[5], y = ip, deltax = 0.01, deltat = 1/100, model_type =3, seed = 300)
#     test_data_1 = sim_1.simulation(-0.59, sim_length = len(zew))

#     #test_data_2 = sim_2.simulation(-0.59, sim_length = len(zew))
#     data.append(test_data_1)
#     #data.append(test_data_2)
#     plt.plot(test_data_1, color = "black")
#     #plt.plot(test_data_2, color = "green")
# plt.plot(zew)   
# plt.show()
# # #######################################################################################################################################
# # #  Plot the transitional Density 
# # ############################################################
# # # ###########################################################################
# x_0 = 0

# test = OpinionFormation(N = 50, T =3 , nu = 3 , alpha0 = 0.01 , alpha1 = -1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.01, deltat = 1/16, model_type =0)    #
# test_1 =OpinionFormation(N = 50, T =3 , nu = 3 , alpha0 = 0.01, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.01, deltat = 1/16, model_type =0) 


# area,prob,prob_end = test.CrankNicolson(x_0 = zew_fw[0],  y = ip[0], x_l = zew[0],calc_dens = True, converged= False,fast_comp = False)
# plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
# plot_0.surface_plot()

# area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = zew_fw[0],  y = ip[0], x_l = zew[0],calc_dens = True, converged= False, fast_comp = False)
# plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# plot_1.surface_plot()

# plot_2 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# plot_2.surface_plot()
# plt.figure(figsize=(7, 6))
# plt.plot(test.t,area, color='blue',
#             label='Area Test 1')
# plt.plot(test_1.t,area_1, color='red',
#             label='Area Test 2')
# plt.legend(loc='lower right')
# plt.title("Area under the PDF")
# plt.xlabel("Time")
# plt.ylabel("Area")
# plt.show()


# plt.figure(figsize=(7, 6))
# plt.plot(test.x,prob_end, color='blue',
#             label='PDF Test 1')
# plt.plot(test_1.x,prob_end_1, color='red',
#             label='PDF Test 2')
# plt.legend(loc='lower right')
# plt.title("Final PDF after T")
# plt.xlabel("Time")
# plt.ylabel("Density")
# plt.show()


# # for x_0 in np.arange(-1,1.1,0.1):
# #     print(x_0)
# #     test = OpinionFormation(N = 50, T =1 , nu = 3, alpha0 = 0.06, alpha1 = 1.2,alpha2 = None,alpha3 = None,deltax= 0.02, deltat= 1/8, model_type= 0)    #
# #     test_1 =OpinionFormation(N = 50, T =1, nu = 10, alpha0= 0.06, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.02, deltat= 1/8, model_type= 0) 


# #     area,prob,prob_end = test.CrankNicolson(x_0,  y = 1, calc_dens = True, converged= False,fast_comp = False)
# #     #plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
# #     #plot_0.surface_plot()

# #     area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0, y= 1,calc_dens = True, converged= False, fast_comp = False)
# #     #plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# #     # plot_1.surface_plot()

# #     # plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
# #     # plot_2.surface_plot()
# #     plt.figure(figsize=(7, 6))
# #     plt.plot(test.t,area, color='blue',
# #                 label='Area Test 1')
# #     plt.plot(test_1.t,area_1, color='red',
# #                 label='Area Test 2')
# #     plt.legend(loc='lower right')
# #     plt.title("Area under the PDF")
# #     plt.xlabel("Time")
# #     plt.ylabel("Area")
# #     plt.show()


# #     plt.figure(figsize=(7, 6))
# #     plt.plot(test.x,prob_end, color='blue',
# #                 label='PDF Test 1')
# #     plt.plot(test_1.x,prob_end_1, color='red',
# #                 label='PDF Test 2')
# #     plt.legend(loc='lower right')
# #     plt.title("Final PDF after T")
# #     plt.xlabel("Time")
# #     plt.ylabel("Density")
# #     plt.show()


# # # # #######################################################################################################################################
# # # # #  Model Type 3 Test

                    
  
# sim_3= sim.Simulation(N = 3.53506447, T =1 , nu = 0.01783543 , alpha0 = -0.1206028, alpha1 = 0.1  ,alpha2 = -1.28791007, alpha3 =  4.63878819, y =  ip, deltax = 0.01, deltat = 1/100, model_type =3, seed = 150) 
# test_data_3 = sim_3.simulation(zew[0], sim_length = 188)


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

# # # from statsmodels.tsa.stattools import adfuller

# # # result = adfuller(zew)
# # # print('ADF Statistic: %f' % result[0])
# # # print('p-value: %f' % result[1])
# # # print('Critical Values:')
# # # for key, value in result[4].items():
# # # 	print('\t%s: %.3f' % (key, value))
# 
# 
data_set_1 = pd.read_csv(r"Estimation\Model_3\estimates_rolling_window.csv")
nu = data_set_1.iloc[:,0]
alpha_0 = data_set_1.iloc[:,1]
alpha_1 = data_set_1.iloc[:,2]
N = data_set_1.iloc[:,3]
alpha_2 = data_set_1.iloc[:,4]
alpha_3 = data_set_1.iloc[:,5]

data_set_2 = pd.read_csv(r"Estimation\real_statistics_rolling.csv")
mean = data_set_2.iloc[:,0]
std = data_set_2.iloc[:,1]

print("test")