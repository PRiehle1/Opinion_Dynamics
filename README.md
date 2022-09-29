# Opinion_Dynamics
Repository for the Analysis of Opinion Diffusion Processes

To reproduce all results, run reproduce_all_results.py and select the specific process.

The final results shown in the thesis are stored in Fianal_Results_Thesis.

List of python scripts and purpose:
    autocorr_frac_mean.py: Stores the mean and confidence interval for the autocorrelation of the models from sample 3 and the mean fractional difference estimator for the samples
    data_reader.py: Python Class to Load the Data (ZEW and IP) FRED API KEY might have to be changed 
    errors.py: Some errors called in the model class 
    estimation.py: Python Class for the Approximate Maximum Likelihood Estimation based on the Model Class
    frac_diff.r: R script to estimate the fractional difference
    mc_estimator_analysis: Returns the mean and standard errors for the parameter estimators of the MC experiment
    model.py: Modell Class for the Weidlich and Haag model, the Crank-Nicolson scheme is also stored here
    order_of_accuracy.py: Calculates the order of accuracy of the Crank-Nicolson scheme in time and space direction 
    plot.py: Python Class for Plots; is  used for the 3D Plot mainly
    reproduce_all_results.py: File how to reproduce all results in the Thesis 
    run_estimation.py: File for the estimation of the simulated as well as the empirical models
    sim.py: Python class for the simulation of the Opinion Process
    statistics_time_series.py Returns the statistics for the ZEW and IP series
    validation.py: Returns the unconditional moments of 1000 simulated models
    
List of Folders: 
    Figures: Contains the Figures in the thesis 
    Estimation: Contains the parameter estimates for the 7 models and for the MC Experiment
    Validation_and_Statistics: 
        Crank_Nicolson: order of accuracy in time and space direction 
        Model_Simulation: Statistics of the simulated model data with estimators 
        Monte_Carlo: Results of the MC Experiment 
        Real_Data: Statistics of the Real Data

