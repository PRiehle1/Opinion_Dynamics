##########################
#  Import Packages       #
##########################
import os
import order_of_accuracy
import validation
import rpy2.robjects as robjects
import autocorr_frac_mean
import statistics_time_series
import Plots
import mc_estimator_analysis

"""
This file explains how to obtain the results:  
1: Order of accuracy Crank-Nicolson scheme
2: Monte Carlo Experiment and Results interpretation  
3: Estimation ZEW Data (Due to multiprocessing, the file has to be startet from the terminal or from itslellf)
    3.1: Estimation Period I
    3.2: Estimation Period II
    3.3: Estimation Period III
4: Monte Carlo Validation Estimated Data 
    4.1: Validation Unconditional Moments Period I
    4.2: Validation Unconditional Moments Period II
    4.3: Validation Unconditional Moments Period III
    4.4: Unconditional Moments ZEW
    4.5: Fractional Differentiation and Autocorrelation
5: Figures 

To get the estimators for the Monte Carlo estimation and the estimators for the samples of real data: run run_estimation.py
(Cannot be called from this file due to multiprocessing; Thus, it would reload the file again and again and dont actually start the estimation. 

"""
#### Insert number from the Table above ###
process = 2
#### Insert number from the Table above ###

if process == 1:
    # 1: Order of accuracy  
    order_of_accuracy.run()
elif process == 2:
    user_input = input('"Please run run_estimation.py and uncomment estimate_simulated_Data()!\n Results obtained?  (yes/no): ')
    if user_input.lower() == 'yes':
        # Analyse the results
        mc_estimator_analysis.run_analysis()
    else:
        print("Please obtain results first")
elif process == 3.1:
    # Estimation Period I #
    user_input = input('"Please run run_estimation.py and uncomment estimate_real_Data(period =1)!\n Results obtained?  (yes/no): ')
    if user_input.lower() == 'yes':
        print("Great")
    else: 
        print("Pleas run run.estimation.estimate_real_Data(period =1) ")
elif process == 3.2:
    # Estimation Period II
    user_input = input('"Please run run_estimation.py and uncomment estimate_real_Data(period =2)!\n Results obtained?  (yes/no): ')
    if user_input.lower() == 'yes':
        print("Great")
    else: 
        print("Pleas run run.estimation.estimate_real_Data(period =2) ")
elif process == 3.3:
    # Estimation Period III #
    user_input = input('"Please run run_estimation.py and uncomment estimate_real_Data(period =3)!\n Results obtained?  (yes/no): ')
    if user_input.lower() == 'yes':
        print("Great")
    else: 
        print("Pleas run run.estimation.estimate_real_Data(period =3) ")
elif process == 4.1: 
    # Validation Period I by 1000 Simmulations for each model
    validation.validation(period= 1)
elif process == 4.2: 
    # Validation Period II by 1000 Simmulations for each model
    validation.validation(period= 2)
elif process == 4.3: 
    # Validation Period III by 1000 Simmulations for each model
    validation.validation(period= 3)
elif process == 4.4:
    # Here the statistics for the ZEW and IP are calculated 
    statistics_time_series.run()
elif process == 4.5:
    # R file for the estimation of fractional differentiation for the simulated and real data for sample 3
    r = robjects.r
    r.source('frac_diff.R')
    # Analyze the results
    autocorr_frac_mean.run_frac_diff_mean()
    autocorr_frac_mean.run_auto_corr()
elif process == 5:
    # Creat all Figures 
    Plots.plot_figures()