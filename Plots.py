from model import OpinionFormation
import plot 
import sim
import matplotlib.pyplot as plt 
from matplotlib import rc
import numpy as np
from tqdm import tqdm
import random
import montecarlo
import estimation
import pandas as pd 
from scipy import interpolate, integrate
from data_reader import data_reader
from data_reader import data_reader


##########################################################################################################
#                                          Figure 1                                                      #
##########################################################################################################

x_0 = 0 

mod_1_1 = OpinionFormation(N = 50, T =3 , nu = 3, alpha0 = 0, alpha1 = 0.8,alpha2 = None,alpha3 = None,deltax= 0.0025, deltat= 1/100, model_type= 0)   
mod_1_2 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= 0.02, alpha1= 0.8, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 
mod_1_3 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= -0.02, alpha1= 0.8, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 

mod_2_1 = OpinionFormation(N = 50, T =3 , nu = 3, alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None,deltax= 0.0025, deltat= 1/100, model_type= 0)   
mod_2_2 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= 0.02, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 
mod_2_3 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= -0.02, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 

mod_3_1 = OpinionFormation(N = 50, T =3 , nu = 3, alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None,deltax= 0.0025, deltat= 1/100, model_type= 0)   
mod_3_2 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= 0.06, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 
mod_3_3 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= -0.06, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 

prob_end_1_1 = mod_1_1.CrankNicolson(x_0) * mod_1_1.N 
prob_end_1_2 = mod_1_2.CrankNicolson(x_0) * mod_1_1.N 
prob_end_1_3 = mod_1_3.CrankNicolson(x_0) * mod_1_1.N 
 
prob_end_2_1 = mod_2_1.CrankNicolson(x_0)
prob_end_2_2 = mod_2_2.CrankNicolson(x_0)
prob_end_2_3 = mod_2_3.CrankNicolson(x_0)

prob_end_3_1 = mod_3_1.CrankNicolson(x_0)
prob_end_3_2 = mod_3_2.CrankNicolson(x_0)
prob_end_3_3 = mod_3_3.CrankNicolson(x_0)

fig, axs = plt.subplots(3,1)
#####################################################################
line1_1, = axs[0].plot(mod_1_1.x,prob_end_1_1, color='black',
            label=r'$\alpha_0 = 0.00$')

line1_2, = axs[0].plot(mod_1_2.x,prob_end_1_2, color='black',
            label=r'$\alpha_0 = 0.02$')
line1_2.set_dashes([10, 2, 10, 2])

line1_3, = axs[0].plot(mod_1_3.x,prob_end_1_3, color='black',
            label=r'$\alpha_0 = -0.02$')
line1_3.set_dashes([2, 2, 2, 2])

axs[0].legend(loc='best', fontsize = 6)
########################################################################
line2_1, = axs[1].plot(mod_2_1.x,prob_end_2_1, color='black',
            label=r'$\alpha_0 = 0.00$')

line2_2, = axs[1].plot(mod_2_2.x,prob_end_2_2, color='black',
            label=r'$\alpha_0 = 0.02$')
line2_2.set_dashes([10, 2, 10, 2])

line2_3, = axs[1].plot(mod_2_3.x,prob_end_2_3, color='black',
            label=r'$\alpha_0 = -0.02$')
line2_3.set_dashes([2, 2, 2, 2])

axs[1].legend(loc='best', fontsize = 6)
axs[1].set_ylabel("Density")
axs[1].set_xlabel("x")
########################################################################
line3_1, = axs[2].plot(mod_3_1.x,prob_end_3_1, color='black',
            label=r'$\alpha_0 = 0.00$')

line3_2, = axs[2].plot(mod_3_2.x,prob_end_3_2, color='black',
            label=r'$\alpha_0 = 0.06$')
line3_2.set_dashes([10, 2, 10, 2])

line3_3, = axs[2].plot(mod_3_3.x,prob_end_3_3, color='black',
            label=r'$\alpha_0 = -0.06$')
line3_3.set_dashes([2, 2, 2, 2])

axs[2].legend(loc='best', fontsize = 6)

plt.show()


##########################################################################################################
#                                          Figure 2                                                      #
##########################################################################################################

