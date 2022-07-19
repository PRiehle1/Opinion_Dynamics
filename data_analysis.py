from cmath import sqrt
import pandas as pd 

# # Data Set 1 

# data_set_1 = pd.read_csv(r"C:\Users\Guest\Desktop\Phillip\Opinion_Dynamics\Opinion_Dynamics\Estimation\sim_Data\exoN\estimates_model_0.csv")
# nu_1 = data_set_1.iloc[0:202,0]
# a0_1 = data_set_1.iloc[0:202,1]
# a1_1 = data_set_1.iloc[0:202,2]

# print("\n Estimates Data Set 1 \n" )
# print(nu_1.mean(), nu_1.var(), nu_1.std()/sqrt(len(nu_1)))
# print(a0_1.mean(), a0_1.var(), a0_1.std()/sqrt(len(nu_1)))
# print(a1_1.mean(), a1_1.var(), a1_1.std()/sqrt(len(nu_1)))


# # Data Set 2

# data_set_2 = pd.read_csv(r"C:\Users\Guest\Desktop\Phillip\Opinion_Dynamics\Opinion_Dynamics\Estimation\sim_Data\exoN\estimates_model_0.csv")
# nu_2 = data_set_1.iloc[202:,0]
# a0_2 = data_set_1.iloc[202:,1]
# a1_2 = data_set_1.iloc[202:,2]

# print("\n Estimates Data Set 2 \n" )
# print(nu_2.mean(), nu_2.var(), nu_2.std()/sqrt(len(nu_1)))
# print(a0_2.mean(), a0_2.var(), a0_2.std()/sqrt(len(nu_1)))
# print(a1_2.mean(), a1_2.var(), a1_2.std()/sqrt(len(nu_1)))

#####################################################################################################################################
#                                      REAL DATA                                                                                             #
#####################################################################################################################################

# Exogenous N = 175 
real_data_set_1 = pd.read_csv(r"Estimation\Model_0\estimates_model_0.csv")
nu_1 = real_data_set_1.iloc[0:202,0]
a0_1 = real_data_set_1.iloc[0:202,1]
a1_1 = real_data_set_1.iloc[0:202,2]

print("\n Estimates Data Set 1 \n" )
print(nu_1.mean(), nu_1.var(), nu_1.std()/sqrt(len(nu_1)))
print(a0_1.mean(), a0_1.var(), a0_1.std()/sqrt(len(nu_1)))
print(a1_1.mean(), a1_1.var(), a1_1.std()/sqrt(len(nu_1)))
