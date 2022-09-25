###################################################################################################
#       Analysis of the Correlation of Parameter Estimates and Statistics                         #
#                                                                                                 # 
###################################################################################################
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts


############ The Estimation Data ###############
rolling_window_data = pd.read_csv(r"Estimation\Model_3\Rolling_Window\estimates_rolling_window_final.csv",header = None)
nu = rolling_window_data.iloc[:,0]
a0 =  rolling_window_data.iloc[:,1]
a1 = rolling_window_data.iloc[:,2]
N = rolling_window_data.iloc[:,3]
a2 = rolling_window_data.iloc[:,4]
a3 = rolling_window_data.iloc[:,5]
############ The Time series Statistics ############
rolling_window_staistics = pd.read_csv(r"Validation_and_Statistics\Real_Data\real_statistics_rolling.csv",header = None)
mean = rolling_window_staistics.iloc[:,0]
std =  rolling_window_staistics.iloc[:,1]
skew = rolling_window_staistics.iloc[:,2]
kurt = rolling_window_staistics.iloc[:,3]
rel_deviation = rolling_window_staistics.iloc[:,4]
########### Correlation #############
# nu
corr_nu_mean, _ = pearsonr(nu, mean)
corr_nu_std, _ = pearsonr(nu, std)
corr_nu_skew, _ = pearsonr(nu, skew)
corr_nu_kurt, _ = pearsonr(nu, kurt)
# alpha_0
corr_a0_mean, _ = pearsonr(a0, mean)
corr_a0_std, _ = pearsonr(a0, std)
corr_a0_skew, _ = pearsonr(a0, skew)
corr_a0_kurt, _ = pearsonr(a0, kurt)
# alpha_1
corr_a1_mean, _ = pearsonr(a1, mean)
corr_a1_std, _ = pearsonr(a1, std)
corr_a1_skew, _ = pearsonr(a1, skew)
corr_a1_kurt, _ = pearsonr(a1, kurt)
# alpha_2
corr_a2_mean, _ = pearsonr(a2, mean)
corr_a2_std, _ = pearsonr(a2, std)
corr_a2_skew, _ = pearsonr(a2, skew)
corr_a2_kurt, _ = pearsonr(a2, kurt)
# alpha_3
corr_a3_mean, _ = pearsonr(a3, mean)
corr_a3_std, _ = pearsonr(a3, std)
corr_a3_skew, _ = pearsonr(a3, skew)
corr_a3_kurt, _ = pearsonr(a3, kurt)
# N
corr_N_mean, _ = pearsonr(N, mean)
corr_N_std, _ = pearsonr(N, std)
corr_N_skew, _ = pearsonr(N, skew)
corr_N_kurt, _ = pearsonr(N, kurt)