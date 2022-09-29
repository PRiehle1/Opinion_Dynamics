### Import Packages ###
from cmath import sqrt
import pandas as pd 
import numpy as np

def run_analysis() ->None:
    ##########################################################################################################################################
    #                                                  T = 200                                                                               #
    ##########################################################################################################################################
    # # Data Set 1 # # # 
    true_param = [3,0,0.8]
    data_set_1 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_1_1_T200_final.csv")
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
    nu_array = np.round((nu_1.mean(),nu_1.median(), nu_1.std(), RMSE_nu),3)
    a0_array = np.round((a0_1.mean(), a0_1.median(),a0_1.std(), RMSE_a0),3)
    a1_array = np.round((a1_1.mean(), a1_1.median(),a1_1.std(), RMSE_a1),3)
    print(np.round((nu_1.mean(),nu_1.median(), nu_1.std(), RMSE_nu),3))
    print(np.round((a0_1.mean(), a0_1.median(),a0_1.std(), RMSE_a0),3))
    print(np.round((a1_1.mean(), a1_1.median(),a1_1.std(), RMSE_a1),3))
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_1_1_T200.csv", (nu_array,a0_array,a1_array))
    #################################################################################################################################
    ###  Data Set 2 ####
    true_param = [3,0.08,0.8]
    data_set_2 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_2_1_T200_final.csv")
    nu_2 = data_set_2.iloc[0:200,0]
    a0_2 = data_set_2.iloc[0:200,1]
    a1_2 = data_set_2.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_2))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_2))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_2))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)
    nu_array = np.round((nu_2.mean(),nu_2.median(), nu_2.std(), RMSE_nu),3)
    a0_array = np.round((a0_2.mean(), a0_2.median(),a0_2.std(), RMSE_a0),3)
    a1_array = np.round((a1_2.mean(), a1_2.median(),a1_2.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_2_1_T200.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 2 \n" )
    print(np.round((nu_2.mean(),nu_2.median(), nu_2.std(), RMSE_nu),3))
    print(np.round((a0_2.mean(), a0_2.median(),a0_2.std(), RMSE_a0),3))
    print(np.round((a1_2.mean(), a1_2.median(),a1_2.std(), RMSE_a1),3))
    #######################################################################################################################################
    # Data Set 3
    true_param = [3,0.00,1.2]
    data_set_3 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_3_1_T200_final.csv")
    nu_3 = data_set_3.iloc[0:200,0]
    a0_3 = data_set_3.iloc[0:200,1]
    a1_3 = data_set_3.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_3))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_3))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_3))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_3.mean(),nu_3.median(), nu_3.std(), RMSE_nu),3)
    a0_array = np.round((a0_3.mean(), a0_3.median(),a0_3.std(), RMSE_a0),3)
    a1_array = np.round((a1_3.mean(), a1_3.median(),a1_3.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_3_1_T200.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 3 \n" )
    print(np.round((nu_3.mean(),nu_3.median(), nu_3.std(), RMSE_nu),3))
    print(np.round((a0_3.mean(), a0_3.median(),a0_3.std(), RMSE_a0),3))
    print(np.round((a1_3.mean(), a1_3.median(),a1_3.std(), RMSE_a1),3))
    ################################################################################################################################################
    # Data Set 4 #
    true_param = [3,0.08,1.2]
    data_set_4 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_4_1_T200_final.csv")
    nu_4 = data_set_4.iloc[0:200,0]
    a0_4 = data_set_4.iloc[0:200,1]
    a1_4 = data_set_4.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_4))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_4))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_4))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_4.mean(),nu_4.median(), nu_4.std(), RMSE_nu),3)
    a0_array = np.round((a0_4.mean(), a0_4.median(),a0_4.std(), RMSE_a0),3)
    a1_array = np.round((a1_4.mean(), a1_4.median(),a1_4.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_4_1_T200.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 4 \n" )
    print(np.round((nu_4.mean(),nu_4.median(), nu_4.std(), RMSE_nu),3))
    print(np.round((a0_4.mean(), a0_4.median(),a0_4.std(), RMSE_a0),3))
    print(np.round((a1_4.mean(), a1_4.median(),a1_4.std(), RMSE_a1),3))
    ################################################################################################################################################
    # Data Set 5 #
    true_param = [1,0,0.8]
    data_set_5 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_1_2_T200_final.csv")
    nu_5 = data_set_5.iloc[0:200,0]
    a0_5 = data_set_5.iloc[0:200,1]
    a1_5 = data_set_5.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_5))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_5))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_5))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_5.mean(),nu_5.median(), nu_5.std(), RMSE_nu),3)
    a0_array = np.round((a0_5.mean(), a0_5.median(),a0_5.std(), RMSE_a0),3)
    a1_array = np.round((a1_5.mean(), a1_5.median(),a1_5.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_1_2_T200.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 5 \n" )
    print(np.round((nu_5.mean(),nu_5.median(), nu_5.std(), RMSE_nu),3))
    print(np.round((a0_5.mean(), a0_5.median(),a0_5.std(), RMSE_a0),3))
    print(np.round((a1_5.mean(), a1_5.median(),a1_5.std(), RMSE_a1),3))
    ################################################################################################################################################
    # Data Set 6 #
    true_param = [1,0.08,0.8]
    data_set_6 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_2_2_T200_final.csv")
    nu_6 = data_set_6.iloc[0:200,0]
    a0_6 = data_set_6.iloc[0:200,1]
    a1_6 = data_set_6.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_6))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_6))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_6))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)
    nu_array = np.round((nu_6.mean(),nu_6.median(), nu_6.std(), RMSE_nu),3)
    a0_array = np.round((a0_6.mean(), a0_6.median(),a0_6.std(), RMSE_a0),3)
    a1_array = np.round((a1_6.mean(), a1_6.median(),a1_6.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_2_2_T200.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 6 \n" )
    print(np.round((nu_6.mean(),nu_6.median(), nu_6.std(), RMSE_nu),3))
    print(np.round((a0_6.mean(), a0_6.median(),a0_6.std(), RMSE_a0),3))
    print(np.round((a1_6.mean(), a1_6.median(),a1_6.std(), RMSE_a1),3))
    ######################################################################################################################################
    # Data Set 7 #
    true_param = [1,0,1.2]

    data_set_7 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_3_2_T200_final.csv")
    nu_7 = data_set_7.iloc[0:200,0]
    a0_7 = data_set_7.iloc[0:200,1]
    a1_7 = data_set_7.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_7))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_7))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_7))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_7.mean(),nu_7.median(), nu_7.std(), RMSE_nu),3)
    a0_array = np.round((a0_7.mean(), a0_7.median(),a0_7.std(), RMSE_a0),3)
    a1_array = np.round((a1_7.mean(), a1_7.median(),a1_7.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_3_2_T200.csv", (nu_array,a0_array,a1_array))


    print("\n Estimates Data Set 7 \n" )
    print(np.round((nu_7.mean(),nu_7.median(), nu_7.std(), RMSE_nu),3))
    print(np.round((a0_7.mean(), a0_7.median(),a0_7.std(), RMSE_a0),3))
    print(np.round((a1_7.mean(), a1_7.median(),a1_7.std(), RMSE_a1),3))
    ######################################################################################################################################
    # Data Set 8 #
    ####################################################################################################################################
    true_param = [1,0.08,1.2]
    data_set_8 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_4_2_T200_final.csv")
    nu_8 = data_set_8.iloc[0:200,0]
    a0_8 = data_set_8.iloc[0:200,1]
    a1_8 = data_set_8.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_8))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_8))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_8))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)
    nu_array = np.round((nu_8.mean(),nu_8.median(), nu_8.std(), RMSE_nu),3)
    a0_array = np.round((a0_8.mean(), a0_8.median(),a0_8.std(), RMSE_a0),3)
    a1_array = np.round((a1_8.mean(), a1_8.median(),a1_8.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_4_2_T200.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 8 \n" )
    print(np.round((nu_8.mean(),nu_8.median(), nu_8.std(), RMSE_nu),3))
    print(np.round((a0_8.mean(), a0_8.median(),a0_8.std(), RMSE_a0),3))
    print(np.round((a1_8.mean(), a1_8.median(),a1_8.std(), RMSE_a1),3))
    # #####################################################################################################################################
    # #                                      T = 400                                                                                      #
    # #####################################################################################################################################
    # # Data Set 1 # # # 

    true_param = [3,0,0.8]

    data_set_1 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_1_1_T400_final.csv")
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
    nu_array = np.round((nu_1.mean(),nu_1.median(), nu_1.std(), RMSE_nu),3)
    a0_array = np.round((a0_1.mean(), a0_1.median(),a0_1.std(), RMSE_a0),3)
    a1_array = np.round((a1_1.mean(), a1_1.median(),a1_1.std(), RMSE_a1),3)
    print(np.round((nu_1.mean(),nu_1.median(), nu_1.std(), RMSE_nu),3))
    print(np.round((a0_1.mean(), a0_1.median(),a0_1.std(), RMSE_a0),3))
    print(np.round((a1_1.mean(), a1_1.median(),a1_1.std(), RMSE_a1),3))
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_1_1_T400.csv", (nu_array,a0_array,a1_array))
    #################################################################################################################################
    ###  Data Set 2 ####

    true_param = [3,0.08,0.8]
    data_set_2 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_2_1_T400_final.csv")
    nu_2 = data_set_2.iloc[0:200,0]
    a0_2 = data_set_2.iloc[0:200,1]
    a1_2 = data_set_2.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_2))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_2))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_2))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)
    nu_array = np.round((nu_2.mean(),nu_2.median(), nu_2.std(), RMSE_nu),3)
    a0_array = np.round((a0_2.mean(), a0_2.median(),a0_2.std(), RMSE_a0),3)
    a1_array = np.round((a1_2.mean(), a1_2.median(),a1_2.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_2_1_T400.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 2 \n" )
    print(np.round((nu_2.mean(),nu_2.median(), nu_2.std(), RMSE_nu),3))
    print(np.round((a0_2.mean(), a0_2.median(),a0_2.std(), RMSE_a0),3))
    print(np.round((a1_2.mean(), a1_2.median(),a1_2.std(), RMSE_a1),3))
    #######################################################################################################################################
    # Data Set 3
    true_param = [3,0.00,1.2]
    data_set_3 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_3_1_T400_final.csv")
    nu_3 = data_set_3.iloc[0:200,0]
    a0_3 = data_set_3.iloc[0:200,1]
    a1_3 = data_set_3.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_3))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_3))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_3))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_3.mean(),nu_3.median(), nu_3.std(), RMSE_nu),3)
    a0_array = np.round((a0_3.mean(), a0_3.median(),a0_3.std(), RMSE_a0),3)
    a1_array = np.round((a1_3.mean(), a1_3.median(),a1_3.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_3_1_T400.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 3 \n" )
    print(np.round((nu_3.mean(),nu_3.median(), nu_3.std(), RMSE_nu),3))
    print(np.round((a0_3.mean(), a0_3.median(),a0_3.std(), RMSE_a0),3))
    print(np.round((a1_3.mean(), a1_3.median(),a1_3.std(), RMSE_a1),3))
    ################################################################################################################################################
    # Data Set 4 #
    true_param = [3,0.08,1.2]

    data_set_4 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_4_1_T400_final.csv")
    nu_4 = data_set_4.iloc[0:200,0]
    a0_4 = data_set_4.iloc[0:200,1]
    a1_4 = data_set_4.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_4))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_4))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_4))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_4.mean(),nu_4.median(), nu_4.std(), RMSE_nu),3)
    a0_array = np.round((a0_4.mean(), a0_4.median(),a0_4.std(), RMSE_a0),3)
    a1_array = np.round((a1_4.mean(), a1_4.median(),a1_4.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_4_1_T400.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 4 \n" )
    print(np.round((nu_4.mean(),nu_4.median(), nu_4.std(), RMSE_nu),3))
    print(np.round((a0_4.mean(), a0_4.median(),a0_4.std(), RMSE_a0),3))
    print(np.round((a1_4.mean(), a1_4.median(),a1_4.std(), RMSE_a1),3))
    ################################################################################################################################################
    # Data Set 5 #
    true_param = [1,0,0.8]

    data_set_5 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_1_2_T400_final.csv")
    nu_5 = data_set_5.iloc[0:200,0]
    a0_5 = data_set_5.iloc[0:200,1]
    a1_5 = data_set_5.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_5))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_5))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_5))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_5.mean(),nu_5.median(), nu_5.std(), RMSE_nu),3)
    a0_array = np.round((a0_5.mean(), a0_5.median(),a0_5.std(), RMSE_a0),3)
    a1_array = np.round((a1_5.mean(), a1_5.median(),a1_5.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_1_2_T400.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 5 \n" )
    print(np.round((nu_5.mean(),nu_5.median(), nu_5.std(), RMSE_nu),3))
    print(np.round((a0_5.mean(), a0_5.median(),a0_5.std(), RMSE_a0),3))
    print(np.round((a1_5.mean(), a1_5.median(),a1_5.std(), RMSE_a1),3))
    ################################################################################################################################################
    # Data Set 6 #
    true_param = [1,0.08,0.8]

    data_set_6 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_2_2_T400_final.csv")
    nu_6 = data_set_6.iloc[0:200,0]
    a0_6 = data_set_6.iloc[0:200,1]
    a1_6 = data_set_6.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_6))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_6))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_6))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)
    nu_array = np.round((nu_6.mean(),nu_6.median(), nu_6.std(), RMSE_nu),3)
    a0_array = np.round((a0_6.mean(), a0_6.median(),a0_6.std(), RMSE_a0),3)
    a1_array = np.round((a1_6.mean(), a1_6.median(),a1_6.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_2_2_T400.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 6 \n" )
    print(np.round((nu_6.mean(),nu_6.median(), nu_6.std(), RMSE_nu),3))
    print(np.round((a0_6.mean(), a0_6.median(),a0_6.std(), RMSE_a0),3))
    print(np.round((a1_6.mean(), a1_6.median(),a1_6.std(), RMSE_a1),3))
    ######################################################################################################################################
    # Data Set 7 #
    true_param = [1,0,1.2]

    data_set_7 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_3_2_T400_final.csv")
    nu_7 = data_set_7.iloc[0:200,0]
    a0_7 = data_set_7.iloc[0:200,1]
    a1_7 = data_set_7.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_7))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_7))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_7))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)

    nu_array = np.round((nu_7.mean(),nu_7.median(), nu_7.std(), RMSE_nu),3)
    a0_array = np.round((a0_7.mean(), a0_7.median(),a0_7.std(), RMSE_a0),3)
    a1_array = np.round((a1_7.mean(), a1_7.median(),a1_7.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_3_2_T400.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 7 \n" )
    print(np.round((nu_7.mean(),nu_7.median(), nu_7.std(), RMSE_nu),3))
    print(np.round((a0_7.mean(), a0_7.median(),a0_7.std(), RMSE_a0),3))
    print(np.round((a1_7.mean(), a1_7.median(),a1_7.std(), RMSE_a1),3))
    ######################################################################################################################################
    # Data Set 8 #
    ####################################################################################################################################
    true_param = [1,0.08,1.2]
    data_set_8 = pd.read_csv(r"Estimation\sim_Data\exoN\estimates_MC_set_4_2_T400_final.csv")
    nu_8 = data_set_8.iloc[0:200,0]
    a0_8 = data_set_8.iloc[0:200,1]
    a1_8 = data_set_8.iloc[0:200,2]

    MSE_nu = np.square(np.subtract(true_param[0],np.array(nu_8))).mean() 
    RMSE_nu = np.sqrt(MSE_nu)

    MSE_a0 = np.square(np.subtract(true_param[1],np.array(a0_8))).mean() 
    RMSE_a0 = np.sqrt(MSE_a0)

    MSE_a1 = np.square(np.subtract(true_param[2],np.array(a1_8))).mean() 
    RMSE_a1 = np.sqrt(MSE_a1)
    nu_array = np.round((nu_8.mean(),nu_8.median(), nu_8.std(), RMSE_nu),3)
    a0_array = np.round((a0_8.mean(), a0_8.median(),a0_8.std(), RMSE_a0),3)
    a1_array = np.round((a1_8.mean(), a1_8.median(),a1_8.std(), RMSE_a1),3)
    np.savetxt("Validation_and_Statistics/Monte_Carlo/Statistics_MC_set_4_2_T400.csv", (nu_array,a0_array,a1_array))

    print("\n Estimates Data Set 8 \n" )
    print(np.round((nu_8.mean(),nu_8.median(), nu_8.std(), RMSE_nu),3))
    print(np.round((a0_8.mean(), a0_8.median(),a0_8.std(), RMSE_a0),3))
    print(np.round((a1_8.mean(), a1_8.median(),a1_8.std(), RMSE_a1),3))