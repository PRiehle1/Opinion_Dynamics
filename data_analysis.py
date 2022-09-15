### Import Packages ###
from cmath import sqrt
import pandas as pd 
import numpy as np
'''
This file anaylizes the Estimates of the Monte Carlo Experiment for the case of T =200 and for the case of T = 400

Results are stored in the Folder: 
'''

##########################################################################################################################################
#                                                  T = 200                                                                               #
##########################################################################################################################################

# # Data Set 1 # # # 

true_param = [3,0,0.8]

data_set_1 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set1_T200_final.csv")
nu_1 = data_set_1.iloc[0:200,0]
a0_1 = data_set_1.iloc[0:200,1]
a1_1 = data_set_1.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_1))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_1))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_1))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 1 \n" )
print(np.round((nu_1.mean(),nu_1.median(), nu_1.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_1.mean(), a0_1.median(),a0_1.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_1.mean(), a1_1.median(),a1_1.std()/np.sqrt(200), RMSE_a1),3))
#################################################################################################################################

# Data Set 2 #

true_param = [3,0.08,0.8]
data_set_2 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set2_T200_final.csv")
nu_2 = data_set_2.iloc[0:200,0]
a0_2 = data_set_2.iloc[0:200,1]
a1_2 = data_set_2.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_2))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_2))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_2))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 2 \n" )
print(np.round((nu_2.mean(),nu_2.median(), nu_2.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_2.mean(), a0_2.median(),a0_2.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_2.mean(), a1_2.median(),a1_2.std()/np.sqrt(200), RMSE_a1),3))

#######################################################################################################################################
# Data Set 3

true_param = [3,0.00,1.2]
data_set_3 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set3_T200_final.csv")
nu_3 = data_set_3.iloc[0:200,0]
a0_3 = data_set_3.iloc[0:200,1]
a1_3 = data_set_3.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_3))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_3))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_3))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 3 \n" )
print(np.round((nu_3.mean(),nu_3.median(), nu_3.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_3.mean(), a0_3.median(),a0_3.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_3.mean(), a1_3.median(),a1_3.std()/np.sqrt(200), RMSE_a1),3))
################################################################################################################################################
# Data Set 4 #
true_param = [3,0.08,1.2]

data_set_4 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set4_T200_final.csv")
nu_4 = data_set_4.iloc[0:200,0]
a0_4 = data_set_4.iloc[0:200,1]
a1_4 = data_set_4.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_4))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_4))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_4))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 4 \n" )
print(np.round((nu_4.mean(),nu_4.median(), nu_4.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_4.mean(), a0_4.median(),a0_4.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_4.mean(), a1_4.median(),a1_4.std()/np.sqrt(200), RMSE_a1),3))

#####################################################################################################################################
#                                      T = 400                                                                                      #
#####################################################################################################################################
# # Data Set 1 # # # 

true_param = [3,0,0.8]

data_set_1 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set1_T400_final.csv")
nu_1 = data_set_1.iloc[0:200,0]
a0_1 = data_set_1.iloc[0:200,1]
a1_1 = data_set_1.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_1))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_1))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_1))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 1 \n" )
print(np.round((nu_1.mean(),nu_1.median(), nu_1.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_1.mean(), a0_1.median(),a0_1.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_1.mean(), a1_1.median(),a1_1.std()/np.sqrt(200), RMSE_a1),3))
#################################################################################################################################

# Data Set 2 #

true_param = [3,0.08,0.8]
data_set_2 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set2_T400_final.csv")
nu_2 = data_set_2.iloc[0:200,0]
a0_2 = data_set_2.iloc[0:200,1]
a1_2 = data_set_2.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_2))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_2))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_2))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 2 \n" )
print(np.round((nu_2.mean(),nu_2.median(), nu_2.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_2.mean(), a0_2.median(),a0_2.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_2.mean(), a1_2.median(),a1_2.std()/np.sqrt(200), RMSE_a1),3))

#######################################################################################################################################
# Data Set 3

true_param = [3,0.00,1.2]
data_set_3 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set3_T400_final.csv")
nu_3 = data_set_3.iloc[0:200,0]
a0_3 = data_set_3.iloc[0:200,1]
a1_3 = data_set_3.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_3))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_3))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_3))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 3 \n" )
print(np.round((nu_3.mean(),nu_3.median(), nu_3.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_3.mean(), a0_3.median(),a0_3.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_3.mean(), a1_3.median(),a1_3.std()/np.sqrt(200), RMSE_a1),3))
################################################################################################################################################
# Data Set 4 #
true_param = [3,0.08,1.2]

data_set_4 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set4_T400_final.csv")
nu_4 = data_set_4.iloc[0:200,0]
a0_4 = data_set_4.iloc[0:200,1]
a1_4 = data_set_4.iloc[0:200,2]

MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_4))).mean() 
RMSE_nu = np.sqrt(MSE_nu)

MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_4))).mean() 
RMSE_a0 = np.sqrt(MSE_a0)

MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_4))).mean() 
RMSE_a1 = np.sqrt(MSE_a1)

print("\n Estimates Data Set 4 \n" )
print(np.round((nu_4.mean(),nu_4.median(), nu_4.std()/np.sqrt(200), RMSE_nu),3))
print(np.round((a0_4.mean(), a0_4.median(),a0_4.std()/np.sqrt(200), RMSE_a0),3))
print(np.round((a1_4.mean(), a1_4.median(),a1_4.std()/np.sqrt(200), RMSE_a1),3))