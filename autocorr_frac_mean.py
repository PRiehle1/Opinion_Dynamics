import pandas as pd 
import statsmodels.api as sm
import numpy as np
from scipy.stats import skew, kurtosis, norm

def run_auto_corr()   ->None:
    """
    It reads in a csv file, converts it to a numpy array, calculates the autocorrelation of each row,
    and then saves the mean and 95% confidence interval of the autocorrelation of each lag to a csv file
    """
    ########################################################################################
    # autocorrelation                                                                        #
    ########################################################################################

    ##################### Model 0 #####################################
    model_0_set_3_data = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_0\sim_0_set3.csv")
    model_0_set_3_data = model_0_set_3_data.to_numpy()
    lags = 10
    acor_0 = []
    for i in range(0,999):
        acor_0.append(sm.tsa.acf(model_0_set_3_data[i,:], nlags = 10))
    acor_0 = np.asarray(acor_0)
    acor_0_final= []
    for i in range(0,11):
        acor_0_final.append(acor_0[:,i].mean())
        acor_0_final.append(norm.interval(0.95, loc=acor_0[:,i].mean() , scale=acor_0[:,i].std()))
    np.savetxt("Validation_and_Statistics/Model_Simulations/Model_0/acor_set_3.csv", acor_0_final, delimiter=",",fmt ='% s')
    ##################### Model 1 #####################################
    model_1_set_3_data = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_1\sim_1_set3.csv")
    model_1_set_3_data = model_1_set_3_data.to_numpy()
    lags = 10
    acor_1 = []
    for i in range(0,999):
        acor_1.append(sm.tsa.acf(model_1_set_3_data[i,:], nlags = 10))
    acor_1 = np.asarray(acor_1)
    acor_1_final= []
    for i in range(0,11):
        acor_1_final.append(acor_1[:,i].mean())
        acor_1_final.append(norm.interval(0.95, loc=acor_1[:,i].mean() , scale=acor_1[:,i].std()))
    np.savetxt("Validation_and_Statistics/Model_Simulations/Model_1/acor_set_3.csv", acor_1_final, delimiter=",",fmt ='% s')
    ##################### Model 2 #####################################
    model_2_set_3_data = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_2\sim_2_set3.csv")
    model_2_set_3_data = model_2_set_3_data.to_numpy()
    lags = 10
    acor_2 = []
    for i in range(0,999):
        acor_2.append(sm.tsa.acf(model_2_set_3_data[i,:], nlags = 10))
    acor_2 = np.asarray(acor_2)
    acor_2_final= []
    for i in range(0,11):
        acor_2_final.append(acor_2[:,i].mean())
        acor_2_final.append(norm.interval(0.95, loc=acor_2[:,i].mean() , scale=acor_2[:,i].std()))
    np.savetxt("Validation_and_Statistics/Model_Simulations/Model_2/acor_set_3.csv", acor_2_final, delimiter=",",fmt ='% s')
    ##################### Model 3 #####################################
    model_3_set_3_data = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_3\sim_3_set3.csv")
    model_3_set_3_data = model_3_set_3_data.to_numpy()
    lags = 10
    acor_3 = []
    for i in range(0,999):
        acor_3.append(sm.tsa.acf(model_3_set_3_data[i,:], nlags = 10))
    acor_3 = np.asarray(acor_3)
    acor_3_final= []
    for i in range(0,11):
        acor_3_final.append(acor_3[:,i].mean())
        acor_3_final.append(norm.interval(0.95, loc=acor_3[:,i].mean() , scale=acor_3[:,i].std()))
    np.savetxt("Validation_and_Statistics/Model_Simulations/Model_3/acor_set_3.csv", acor_3_final, delimiter=",",fmt ='% s')
    ##################### Model 4 #####################################
    model_4_set_3_data = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_4\sim_4_set3.csv")
    model_4_set_3_data = model_4_set_3_data.to_numpy()
    lags = 10
    acor_4 = []
    for i in range(0,999):
        acor_4.append(sm.tsa.acf(model_4_set_3_data[i,:], nlags = 10))
    acor_4 = np.asarray(acor_4)
    acor_4_final= []
    for i in range(0,11):
        acor_4_final.append(acor_4[:,i].mean())
        acor_4_final.append(norm.interval(0.95, loc=acor_4[:,i].mean() , scale=acor_4[:,i].std()))
    np.savetxt("Validation_and_Statistics/Model_Simulations/Model_4/acor_set_3.csv", acor_4_final, delimiter=",",fmt ='% s')
    ##################### Model 5 #####################################
    model_5_set_3_data = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_5\sim_5_set3.csv")
    model_5_set_3_data = model_5_set_3_data.to_numpy()
    lags = 10
    acor_5 = []
    for i in range(0,999):
        acor_5.append(sm.tsa.acf(model_5_set_3_data[i,:], nlags = 10))
    acor_5 = np.asarray(acor_5)
    acor_5_final= []
    for i in range(0,11):
        acor_5_final.append(acor_5[:,i].mean())
        acor_5_final.append(norm.interval(0.95, loc=acor_5[:,i].mean() , scale=acor_5[:,i].std()))
    np.savetxt("Validation_and_Statistics/Model_Simulations/Model_5/acor_set_3.csv", acor_5_final, delimiter=",",fmt ='% s')
    ##################### Model 6 #####################################
    model_6_set_3_data = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_6\sim_6_set3.csv")
    model_6_set_3_data = model_6_set_3_data.to_numpy()
    lags = 10
    acor_6 = []
    for i in range(0,999):
        acor_6.append(sm.tsa.acf(model_6_set_3_data[i,:], nlags = 10))
    acor_6 = np.asarray(acor_6)
    acor_6_final= []
    for i in range(0,11):
        acor_6_final.append(acor_6[:,i].mean())
        acor_6_final.append(norm.interval(0.95, loc=acor_6[:,i].mean() , scale=acor_6[:,i].std()))
    np.savetxt("Validation_and_Statistics/Model_Simulations/Model_6/acor_set_3.csv", np.round(acor_6_final,3), delimiter=",",fmt ='% s')

    ##################### Model ZEW #####################################
    from data_reader import data_reader
    import pandas as pd 
    data_1 = data_reader(time_start= 0, time_end= 364)
    zew_1 = data_1.zew()/100
    acor_5_zew = []
    acor_5_zew.append(sm.tsa.acf(zew_1, nlags = 10))
    np.savetxt("Validation_and_Statistics/Real_Data/acor_zew_period_3.csv", acor_5_zew, delimiter=",",fmt ='% s')

def run_frac_diff_mean()  -> None:
    """
    > This function reads in the fractional differentiation values for each model and prints the mean
    and 95% confidence interval for each model
    """
    ########################################################################################
    # mean fractional differentiation                                                      #
    ########################################################################################
    ##################### Model 0 #####################################
    frac_d_0 = pd.read_csv(r"d_sim_3_set_1.csv").to_numpy()
    print(frac_d_0.mean())
    print(norm.interval(0.95, loc=frac_d_0.mean() , scale=frac_d_0.std()))
    ##################### Model 1 #####################################
    frac_d_1 = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_1\d_sim_1_set_3.csv").to_numpy()
    print(frac_d_1.mean())
    print(norm.interval(0.95, loc=frac_d_1.mean() , scale=frac_d_1.std()))
    ##################### Model 2 #####################################
    frac_d_2 = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_2\d_sim_2_set_3.csv").to_numpy()
    print(frac_d_2.mean())
    print(norm.interval(0.95, loc=frac_d_2.mean() , scale=frac_d_2.std()))
    ##################### Model 3 #####################################
    frac_d_3 = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_3\d_sim_3_set_3.csv").to_numpy()
    print(frac_d_3.mean())
    print(norm.interval(0.95, loc=frac_d_3.mean() , scale=frac_d_3.std()))
    ##################### Model 4 #####################################
    frac_d_4 = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_4\d_sim_4_set_3.csv").to_numpy()
    print(frac_d_4.mean())
    print(norm.interval(0.95, loc=frac_d_4.mean() , scale=frac_d_4.std()))
    ##################### Model 5 #####################################
    frac_d_5 = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_5\d_sim_5_set_3.csv").to_numpy()
    print(frac_d_5.mean())
    print(norm.interval(0.95, loc=frac_d_5.mean() , scale=frac_d_5.std()))
    ##################### Model 6 #####################################
    frac_d_6 = pd.read_csv(r"Validation_and_Statistics\Model_Simulations\Model_6\d_sim_6_set_3.csv").to_numpy()
    print(frac_d_6.mean())
    print(norm.interval(0.95, loc=frac_d_6.mean() , scale=frac_d_6.std()))
