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


test = OpinionFormation(N = 50, T = 10, nu = 3, alpha0= 0.02, alpha1= 1.2, alpha2 = None,alpha3 = None, y = None, deltax= 0.01, deltat= 1/16, model_type= 0)    #
test_1 =OpinionFormation(N = 50, T = 10, nu = 3, alpha0= 0.02, alpha1= 0.8, alpha2 = None,alpha3 = None, y = None, deltax= 0.02, deltat= 1/16, model_type= 0) 


prob,prob_end = test.CrankNicolson(x_0 = (-0.8), y = 1, check_stability = False, calc_dens = False, converged =  False, fast_comp = False)
plot_0 = plot.Plotting3D(param = prob, x = test.x, t = test.t)
plot_0.surface_plot()



prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = (-0.8), y= 1, check_stability = False, calc_dens = False, converged =  False, fast_comp = False)
# plot_1 = plot.Plotting3D(param = prob_1, x = test_1.x, t = test_1.t)
# plot_1.surface_plot()

# plot_2 = plot.Plotting3D(param = prob_2, x = test_2.x, t = test_2.t)
# plot_2.surface_plot()



plt.plot(test_1.x,prob_end_1)
plt.plot(test.x,prob_end)
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