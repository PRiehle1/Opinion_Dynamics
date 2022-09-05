from cmath import sqrt
import pandas as pd 

# # Data Set 1 

data_set_1 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_model_0.csv")
nu_1 = data_set_1.iloc[0:200,0]
a0_1 = data_set_1.iloc[0:200,1]
a1_1 = data_set_1.iloc[0:200,2]

print("\n Estimates Data Set 1 \n" )
print(nu_1.mean(), nu_1.var(), nu_1.std()/sqrt(len(nu_1)))
print(a0_1.mean(), a0_1.var(), a0_1.std()/sqrt(len(nu_1)))
print(a1_1.mean(), a1_1.var(), a1_1.std()/sqrt(len(nu_1)))


# Data Set 2

#data_set_2 = pd.read_csv(r"C:\Users\Guest\Desktop\Phillip\Opinion_Dynamics\Opinion_Dynamics\Estimation\sim_Data\exoN\estimates_model_0.csv")
nu_2 = data_set_1.iloc[201:403,0]
a0_2 = data_set_1.iloc[201:403,1]
a1_2 = data_set_1.iloc[201:403,2]

print("\n Estimates Data Set 2 \n" )
print(nu_2.mean(), nu_2.var(), nu_2.std()/sqrt(len(nu_2)))
print(a0_2.mean(), a0_2.var(), a0_2.std()/sqrt(len(nu_2)))
print(a1_2.mean(), a1_2.var(), a1_2.std()/sqrt(len(nu_2)))

# Data Set 3

#data_set_2 = pd.read_csv(r"C:\Users\Guest\Desktop\Phillip\Opinion_Dynamics\Opinion_Dynamics\Estimation\sim_Data\exoN\estimates_model_0.csv")
nu_3 = data_set_1.iloc[404:,0]
a0_3 = data_set_1.iloc[404:,1]
a1_3 = data_set_1.iloc[404:,2]

print("\n Estimates Data Set 3 \n" )
print(nu_3.mean(), nu_3.var(), nu_3.std()/sqrt(len(nu_1)))
print(a0_3.mean(), a0_3.var(), a0_3.std()/sqrt(len(nu_1)))
print(a1_3.mean(), a1_3.var(), a1_3.std()/sqrt(len(nu_1)))

# Data Set 4

data_set_2 = pd.read_csv(r"C:\Users\Guest\Desktop\Phillip\Opinion_Dynamics\Opinion_Dynamics\Estimation\sim_Data\exoN\estimates_model_0.csv")
nu_4 = data_set_1.iloc[603:,0]
a0_4 = data_set_1.iloc[603:,1]
a1_4 = data_set_1.iloc[603:,2]

print("\n Estimates Data Set 4 \n" )
print(nu_4.mean(), nu_4.var(), nu_4.std()/sqrt(len(nu_1)))
print(a0_4.mean(), a0_4.var(), a0_4.std()/sqrt(len(nu_1)))
print(a1_4.mean(), a1_4.var(), a1_4.std()/sqrt(len(nu_1)))



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
