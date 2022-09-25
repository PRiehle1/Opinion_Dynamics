from model import OpinionFormation
import plot 
import sim
import matplotlib.pyplot as plt 
from matplotlib import rc
import numpy as np
from tqdm import tqdm
import random
#import montecarlo
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
# mod_1 = OpinionFormation(N = 50, T =5, nu = 3 , alpha0 = 0. , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.001, deltat = 1/16, model_type =0)  
# area_1, prob_1,prob_end_1 = mod_1.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

# mod_2 = OpinionFormation(N = 50, T =5, nu = 1 , alpha0 = 0.00 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.001, deltat = 1/16, model_type =0)  
# area_2, prob_2,prob_end_2 = mod_2.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

# fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})

# time, space = np.meshgrid(mod_1.t, mod_1.x)

# # Plot the surface.
# surf = ax[0].plot_surface(time, space, prob_1, 
#                 linewidth=0, antialiased=False)

# surf = ax[1].plot_surface(time, space, prob_2, 
#                 linewidth=0, antialiased=False, color = "grey")

# ax[0].set_zlabel("Probability Density")
# ax[0].set_ylabel("x")
# ax[0].set_xlabel("Time", rotation = 0.5)
# ax[0].set_yticks((-1,0,1))
# ax[0].view_init(10,40)
# ax[0].text2D(0.2, 0.7, "(a) " + r'$\nu = 3$', transform=ax[0].transAxes)

# ax[1].set_ylabel("x")
# ax[1].set_yticks((-1,0,1))
# ax[1].view_init(10,40)
# ax[1].set_xlabel("Time")
# ax[1].text2D(0.2, 0.7, "(b) " + r'$\nu = 1$', transform=ax[1].transAxes)

# plt.savefig('Figure_2.png', dpi=600,bbox_inches='tight')


# ##########################################################################################################
# #                                          Figure 3                                                      #
# ##########################################################################################################
# from data_reader import data_reader
# import pandas as pd 

# month = pd.date_range('1991-12-01','2022-03-30', freq='MS').strftime("%Y-%b").tolist()

# data_1 = data_reader(time_start= 0, time_end= 176)
# zew_1 = data_1.zew()/100
# ip_1 = data_1.industrial_production()

# data_2 = data_reader(time_start= 176, time_end= 364)
# zew_2 = data_2.zew()/100
# ip_2 = data_2.industrial_production()

# fig, axs = plt.subplots(2,1)

# axs[0].plot(month[0:len(zew_1)],zew_1, color = "black",linestyle="--", linewidth=2, label="Period 1")
# axs[0].hlines(y=zew_1.mean(), xmin=0, xmax=len(zew_1), linewidth=1,linestyle="-.", color='black', label= "Mean Period 1")
# axs[0].plot(month[len(zew_1):], zew_2, color = "black",linestyle=":", label = "Period 2", linewidth=2)
# axs[0].hlines(y=zew_2.mean(), xmin=len(zew_1), xmax=len(zew_1)+len(zew_2) , linewidth=1, color='black', label= "Mean Period 2")
# axs[0].set_xticks([0,109,169+60,169+180],rotation=0)
# axs[0].set_yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
# axs[0].set_ylabel("x")
# axs[0].set_xlabel("Time")
# axs[0].legend(loc='lower left', fontsize = 8)

# axs[1].plot(month[0:len(zew_1)],ip_1, color = "black",linestyle="--", linewidth=2, label="Period 1")

# axs[1].plot(month[len(zew_1):], ip_2, color = "black",linestyle=":", label = "Period 2", linewidth=2)

# axs[1].set_xticks([0,109,169+60,169+180],rotation=0)
# axs[1].set_ylabel("Log IP")
# axs[1].set_xlabel("Time")
# axs[1].legend(loc ='lower left', fontsize = 8)

# axs[0].text(0.5,1.1, "(a)", size=10, ha="center", 
#          transform=axs[0].transAxes)
# axs[1].text(0.5,1.1, "(b)", size=10, ha="center", 
#          transform=axs[1].transAxes)

# fig.tight_layout()
# fig.set_size_inches(8, 6)
# plt.savefig('Figure_3.png', dpi=600,bbox_inches='tight')
# plt.show()

##########################################################################################################
#                                          Figure 4                                                      #
##########################################################################################################

rolling_window_data = pd.read_csv(r"Estimation\Model_3\Rolling_Window\estimates_rolling_window_final.csv",header = None)
rolling_window_data_variance = pd.read_csv(r"Estimation\Model_3\Rolling_Window\variance_est_rolling_window_final.csv",header = None)
rolling_window_data_variance = rolling_window_data_variance.to_numpy()
rolling_window_data_se = np.sqrt(rolling_window_data_variance)/(np.sqrt(175))

nu = rolling_window_data.iloc[:,0]
a0 =  rolling_window_data.iloc[:,1]
a1 = rolling_window_data.iloc[:,2]
N = rolling_window_data.iloc[:,3]
a2 = rolling_window_data.iloc[:,4]
a3 = rolling_window_data.iloc[:,5]

fig, axs = plt.subplots(2,3)
axs[0,0].plot(nu + rolling_window_data_se[:,0] * 1.96, color = "black",linestyle="--")
axs[0,0].plot(nu, color = "black")
axs[0,0].plot(nu - rolling_window_data_se[:,0] * 1.96,color = "black",linestyle="--")
axs[0,0].set_xlabel("Window")
axs[0,0].set_ylabel(r"$\nu$")

axs[0,1].plot(a0 - rolling_window_data_se[:,1] * 1.96,color = "black",linestyle="--")
axs[0,1].plot(a0, color = "black")
axs[0,1].set_ylabel(r"$\alpha_0$")
axs[0,1].set_xlabel("Window")
axs[0,1].plot(a0 + rolling_window_data_se[:,1] * 1.96,color = "black",linestyle="--")

axs[0,2].plot(a1 - rolling_window_data_se[:,2] * 1.96,color = "black",linestyle="--")
axs[0,2].plot(a1, color = "black")
axs[0,2].set_ylabel(r"$\alpha_1$")
axs[0,2].set_xlabel("Window")
axs[0,2].plot(a1 + rolling_window_data_se[:,2] * 1.96,color = "black",linestyle="--")

axs[1,0].plot(N - rolling_window_data_se[:,3] * 1.96,color = "black",linestyle="--")
axs[1,0].plot(N, color = "black")
axs[1,0].set_ylabel(r"$N$")
axs[1,0].set_xlabel("Window")
axs[1,0].plot(N + rolling_window_data_se[:,3] * 1.96,color = "black",linestyle="--")

axs[1,1].plot(a3 - rolling_window_data_se[:,5] * 1.96,color = "black",linestyle="--")
axs[1,1].plot(a3, color = "black")
axs[1,1].set_ylabel(r"$\alpha_3$")
axs[1,1].set_xlabel("Window")
axs[1,1].plot(a3 + rolling_window_data_se[:,5] * 1.96,color = "black",linestyle="--")

axs[1,2].plot(a2 - rolling_window_data_se[:,4] * 1.96,color = "black",linestyle="--")
axs[1,2].plot(a2, color = "black")
axs[1,2].set_ylabel(r"$\alpha_2$")
axs[1,2].set_xlabel("Window")
axs[1,2].plot(a2 + rolling_window_data_se[:,4] * 1.96,color = "black",linestyle="--")

fig.tight_layout()
fig.set_size_inches(9, 4)
plt.savefig('Figure_4.png', dpi=600,bbox_inches='tight')
plt.show()


rolling_window_staistics = pd.read_csv(r"Validation_and_Statistics\Real_Data\real_statistics_rolling.csv",header = None)
mean = rolling_window_staistics.iloc[:,0]
std =  rolling_window_staistics.iloc[:,1]
skew = rolling_window_staistics.iloc[:,2]
kurt = rolling_window_staistics.iloc[:,3]
rel_deviation = rolling_window_staistics.iloc[:,4]


fig, axs = plt.subplots(2,2)
axs[0,0].plot(mean)
axs[0,0].set_xlabel("Year")
axs[0,0].set_ylabel("mean")

axs[0,1].plot(skew)
axs[0,1].set_ylabel("skewness")

axs[1,0].plot(kurt)
axs[1,0].set_ylabel("kurtosis")

axs[1,1].plot(std)
axs[1,1].set_ylabel("std")


plt.show()

