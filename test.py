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
from data_reader import data_reader

data = data_reader(time_start= 0, time_end= 176)
zew = data.zew()/100
ip = data.industrial_production()
# Account for smaller time Series
zew = zew[0:len(ip)]
zew_fw = zew[1:]
numSim = 20















a = [2.513983560573791395e-03,7.151851888075135000e-03,1.051185290497904101e-01,1.004240291471391933e+00]

print(np.round(np.sqrt(a)/np.sqrt(176),2))

b = [5.575329665999526757e-02,1.885904505476352928e-03,1.098573718186092590e-01,1.001160867186168701e+00]

print(np.round(np.sqrt(b)/np.sqrt(188),2))


c = [1.719364542543115215e+00,9.584456873919964051e-03,2.058505042698002274e+00,2.218666376769498311e+04]

print(np.round(np.sqrt(c)/np.sqrt(364),2))

a = [6.117488170166698880e-02,1.782351976629232837e-01,6.812652391311114775e-01,-8.149912846434546054e+00,2.956119945965324902e+00,
1.636743398294992213e-01,4.285325524538987491e-02,6.366677043208186504e-01,-2.179584353366638361e+00,6.785611793848753681e-01,
9.982654770098728370e-02,7.725878108398381849e-02,7.430349735743482231e-01,-2.323357026504988543e+00,1.680065639009542089e+00]


print(np.round(a,2))










# from data_reader import data_reader

# data = data_reader(time_start= 176, time_end= 364)
# zew = data.zew()/100
# ip = data.industrial_production()
# # Account for smaller time Series
# zew = zew[0:len(ip)]
# zew_fw = zew[1:]
# numSim = 20

# # data = []
# # param = [1.670649116368989190e-02,-1.046296954418928365e-01,-7.048724836343813749e-01,3.372362793503125733e+00,-1.000910344445289901e+01,4.464152674741839633e+00]

# # for _ in range(1):
# #     sim_1 = sim.Simulation(N = 50, T = 1, nu = 3 , alpha0 = 0, alpha1 = 0,alpha2 = param[4] ,alpha3 = param[5], y = ip, deltax = 0.01, deltat = 1/100, model_type =0, seed = 300)
# #     sim_2 = sim.Simulation(N = param[3], T = 1, nu = param[0] , alpha0 = param[1], alpha1 = 0,alpha2 = param[4] ,alpha3 = param[5], y = ip, deltax = 0.01, deltat = 1/100, model_type =3, seed = 300)
# #     test_data_1 = sim_1.simulation(0, sim_length = 188)

# #     test_data_2 = sim_2.simulation(0, sim_length = 188)
# #     data.append(test_data_1)
# #     data.append(test_data_2)
# #     plt.plot(test_data_1, color = "black")
# #     plt.plot(test_data_2, color = "green")
# # plt.plot(zew)   
# # plt.show()
# # # #######################################################################################################################################
# # # #  Plot the transitional Density 
# # # ############################################################
# # # # ###########################################################################
x_0 = 0.8

test_1 = OpinionFormation(N = 50, T =1, nu = 3 , alpha0 = 0.0 , alpha1 = 0.8  ,alpha2 = None ,alpha3 =  None, deltax = 0.01, deltat = 1/100, model_type =0)    #
test_2 =OpinionFormation(N = 50, T =1, nu = 3 , alpha0 = 0.0, alpha1 =0 ,alpha2 = None ,alpha3 =  None, deltax = 0.01, deltat = 1/100, model_type =0) 
test_3 = OpinionFormation(N = 50, T = 1 , nu = 3 , alpha0 = 0.0 , alpha1 = -0.8 ,alpha2 = None ,alpha3 =  None, deltax = 0.01, deltat = 1/100, model_type =0)    #


area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = 0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
area_1, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = 0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
area_1, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = 0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

#print(np.round(np.divide(np.subtract(prob_end_3,prob_end_2),np.subtract(prob_end_2,prob_end_1)),2))

#plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
#plot_1.surface_plot()

# plot_2 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# plot_2.surface_plot()
plt.figure(figsize=(7, 6))
plt.plot(test_1.t,area_1, color='blue',
            label='Area Test 1')
plt.plot(test_1.t,area_1, color='red',
            label='Area Test 2')
plt.legend(loc='lower right')
plt.title("Area under the PDF")
plt.xlabel("Time")
plt.ylabel("Area")
plt.show()


plt.figure(figsize=(7, 6))
plt.plot(test_1.x,prob_end_1, color='blue',
            label='PDF Test 1')
plt.plot(test_2.x,prob_end_3, color='red',
            label='PDF Test 2')
plt.legend(loc='lower right')
plt.title("Final PDF after T")
plt.xlabel("Time")
plt.ylabel("Density")
plt.show()


# # # for x_0 in np.arange(-1,1.1,0.1):
# # #     print(x_0)
# # #     test = OpinionFormation(N = 50, T =1 , nu = 3, alpha0 = 0.06, alpha1 = 1.2,alpha2 = None,alpha3 = None,deltax= 0.02, deltat= 1/8, model_type= 0)    #
# # #     test_1 =OpinionFormation(N = 50, T =1, nu = 10, alpha0= 0.06, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.02, deltat= 1/8, model_type= 0) 


# # #     area,prob,prob_end = test.CrankNicolson(x_0,  y = 1, calc_dens = True, converged= False,fast_comp = False)
# # #     #plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
# # #     #plot_0.surface_plot()

# # #     area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0, y= 1,calc_dens = True, converged= False, fast_comp = False)
# # #     #plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# # #     # plot_1.surface_plot()

# # #     # plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
# # #     # plot_2.surface_plot()
# # #     plt.figure(figsize=(7, 6))
# # #     plt.plot(test.t,area, color='blue',
# # #                 label='Area Test 1')
# # #     plt.plot(test_1.t,area_1, color='red',
# # #                 label='Area Test 2')
# # #     plt.legend(loc='lower right')
# # #     plt.title("Area under the PDF")
# # #     plt.xlabel("Time")
# # #     plt.ylabel("Area")
# # #     plt.show()


# # #     plt.figure(figsize=(7, 6))
# # #     plt.plot(test.x,prob_end, color='blue',
# # #                 label='PDF Test 1')
# # #     plt.plot(test_1.x,prob_end_1, color='red',
# # #                 label='PDF Test 2')
# # #     plt.legend(loc='lower right')
# # #     plt.title("Final PDF after T")
# # #     plt.xlabel("Time")
# # #     plt.ylabel("Density")
# # #     plt.show()


# # # # # #######################################################################################################################################
# # # # # #  Model Type 3 Test

                    
  
# # sim_3= sim.Simulation(N = 3.53506447, T =1 , nu = 0.01783543 , alpha0 = -0.1206028, alpha1 = 0.1  ,alpha2 = -1.28791007, alpha3 =  4.63878819, y =  ip, deltax = 0.01, deltat = 1/100, model_type =3, seed = 150) 
# # test_data_3 = sim_3.simulation(zew[0], sim_length = 188)


# # plt.plot(test_data_3)
# # plt.plot(zew)
# # plt.show()

# # plt.figure(figsize=(7, 6))
# # plt.plot(test_data_3, color='blue',
# #             label='Simulated Data')
# # plt.plot(zew, color='red',
# #             label='ZEW')
# # plt.legend(loc='lower right')
# # plt.title("Value")
# # plt.xlabel("Time")
# # plt.ylabel("Density")
# # plt.show()

# # # # from statsmodels.tsa.stattools import adfuller

# # # # result = adfuller(zew)
# # # # print('ADF Statistic: %f' % result[0])
# # # # print('p-value: %f' % result[1])
# # # # print('Critical Values:')
# # # # for key, value in result[4].items():
# # # # 	print('\t%s: %.3f' % (key, value))
# # 
# # 
# # data_set_1 = pd.read_csv(r"Estimation\Model_3\estimates_rolling_window.csv")
# # nu = data_set_1.iloc[:,0]
# # alpha_0 = data_set_1.iloc[:,1]
# # alpha_1 = data_set_1.iloc[:,2]
# # N = data_set_1.iloc[:,3]
# # alpha_2 = data_set_1.iloc[:,4]
# # alpha_3 = data_set_1.iloc[:,5]

# # data_set_2 = pd.read_csv(r"Estimation\real_statistics_rolling.csv")
# # mean = data_set_2.iloc[:,0]
# # std = data_set_2.iloc[:,1]

# # print("test")