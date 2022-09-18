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

# x_0 = 0 

# mod_1_1 = OpinionFormation(N = 50, T =3, nu = 3, alpha0 = 0, alpha1 = 0.8,alpha2 = None,alpha3 = None,deltax= 0.0025, deltat= 1/100, model_type= 0)   
# mod_1_2 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= 0.05, alpha1= 0.8, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 
# mod_1_3 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= -0.05, alpha1= 0.8, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 

# mod_2_1 = OpinionFormation(N = 50, T =3 , nu = 3, alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None,deltax= 0.0025, deltat= 1/100, model_type= 0)   
# mod_2_2 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= 0.02, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 
# mod_2_3 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= -0.02, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 

# mod_3_1 = OpinionFormation(N = 50, T =3 , nu = 3, alpha0 = 0, alpha1 = 1.2,alpha2 = None,alpha3 = None,deltax= 0.0025, deltat= 1/100, model_type= 0)   
# mod_3_2 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= 0.06, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 
# mod_3_3 = OpinionFormation(N = 50, T =3, nu = 3, alpha0= -0.06, alpha1= 1.2, alpha2 = None,alpha3 = None, deltax= 0.0025, deltat= 1/100, model_type= 0) 

# prob_end_1_1 = mod_1_1.CrankNicolson(x_0) 
# prob_end_1_2 = mod_1_2.CrankNicolson(x_0) 
# prob_end_1_3 = mod_1_3.CrankNicolson(x_0) 
 
# prob_end_2_1 = mod_2_1.CrankNicolson(x_0)
# prob_end_2_2 = mod_2_2.CrankNicolson(x_0)
# prob_end_2_3 = mod_2_3.CrankNicolson(x_0)

# prob_end_3_1 = mod_3_1.CrankNicolson(x_0)
# prob_end_3_2 = mod_3_2.CrankNicolson(x_0)
# prob_end_3_3 = mod_3_3.CrankNicolson(x_0)

# fig, axs = plt.subplots(3,1)
# #####################################################################
# line1_1, = axs[0].plot(mod_1_1.x,prob_end_1_1, color='black',
#             label=r'$\alpha_0 = 0.00$')

# line1_2, = axs[0].plot(mod_1_2.x,prob_end_1_2, color='black',
#             label=r'$\alpha_0 = 0.05$')
# line1_2.set_dashes([10, 2, 10, 2])

# line1_3, = axs[0].plot(mod_1_3.x,prob_end_1_3, color='black',
#             label=r'$\alpha_0 = -0.05$')
# line1_3.set_dashes([2, 2, 2, 2])

# axs[0].legend(loc='best', fontsize = 6)
# ########################################################################
# line2_1, = axs[1].plot(mod_2_1.x,prob_end_2_1, color='black',
#             label=r'$\alpha_0 = 0.00$')

# line2_2, = axs[1].plot(mod_2_2.x,prob_end_2_2, color='black',
#             label=r'$\alpha_0 = 0.02$')
# line2_2.set_dashes([10, 2, 10, 2])

# line2_3, = axs[1].plot(mod_2_3.x,prob_end_2_3, color='black',
#             label=r'$\alpha_0 = -0.02$')
# line2_3.set_dashes([2, 2, 2, 2])

# axs[1].legend(loc='best', fontsize = 6)
# ########################################################################
# line3_1, = axs[2].plot(mod_3_1.x,prob_end_3_1, color='black',
#             label=r'$\alpha_0 = 0.00$')

# line3_2, = axs[2].plot(mod_3_2.x,prob_end_3_2, color='black',
#             label=r'$\alpha_0 = 0.06$')
# line3_2.set_dashes([10, 2, 10, 2])

# line3_3, = axs[2].plot(mod_3_3.x,prob_end_3_3, color='black',
#             label=r'$\alpha_0 = -0.06$')
# line3_3.set_dashes([2, 2, 2, 2])

# axs[2].legend(loc='best', fontsize = 6)

# for ax in axs.flat:
#     ax.set(xlabel='x', ylabel='Density')

# axs[0].text(0.5,1.1, "(a)  " + r'$\alpha_1 = 0.8$', size=8, ha="center", 
#          transform=axs[0].transAxes)
# axs[1].text(0.5,1.1, "(b)  " +  r'$\alpha_1 = 1.2$', size=8, ha="center", 
#          transform=axs[1].transAxes)
# axs[2].text(0.5,1.1, "(c)  " + r'$\alpha_1 = 1.2$' , size=8, ha="center", 
#          transform=axs[2].transAxes)
# fig.set_size_inches(4, 5)
# fig.tight_layout()
# plt.savefig('Figure_1.png', dpi=600)
# plt.show()


##########################################################################################################
#                                          Figure 2                                                      #
##########################################################################################################
mod_1 = OpinionFormation(N = 50, T =3, nu = 3 , alpha0 = 0. , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.001, deltat = 1/16, model_type =0)  
area_1, prob_1,prob_end_1 = mod_1.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

mod_2 = OpinionFormation(N = 50, T =3, nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.001, deltat = 1/16, model_type =0)  
area_2, prob_2,prob_end_2 = mod_2.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})

time, space = np.meshgrid(mod_1.t, mod_1.x)

# Plot the surface.
surf = ax[0].plot_surface(time, space, prob_1, 
                linewidth=0, antialiased=False)

surf = ax[1].plot_surface(time, space, prob_2, 
                linewidth=0, antialiased=False)

ax[0].set_zlabel("Probability Density")
ax[0].set_ylabel("x")
ax[0].set_yticks((-1,0,1))
ax[0].view_init(10,40)

ax[1].set_ylabel("x")
ax[1].set_yticks((-1,0,1))
ax[1].view_init(10,40)

plt.savefig('Figure_2.png', dpi=600,bbox_inches='tight')


