##############################################################################################################
#                                        Model Validation                                                    #
##############################################################################################################

#############################################################
#                   Import Packages                         #
#############################################################

from tkinter import N
import sim 
import numpy as np 
from scipy.stats import skew, kurtosis, norm
from data_reader import data_reader
import matplotlib.pyplot as plt
import os

############################################################
#               Helper Functions                           #
############################################################
def distance(est_array: np.array, time_series:np.array) -> float:
    dummy = []
    for i in range(0,len(est_array)-1):
        dummy.append(np.abs(est_array[i]- time_series[i]))
    dist = 1/len(time_series) * np.sum(dummy)

    return dist
############################################################
#                Main Function                             #
############################################################
def get_statistics(sim_model:object, numSim:int, init_value:float, time_length:int):
    simu_0 = []
    mean_0 =  []
    std_0 = []
    skw_0 = []
    kurt_0 = []
    rel_dev_0 = []
    dist_0 = []
    model_0_statistics =[]
    for i in range(0,numSim):
        # Simulation
        simu_0.append(sim_model.simulation(init_value,time_length))
        # Moments
        simu_0_array = np.asarray(simu_0[i], dtype=object)
        mean_0.append(simu_0_array.mean())
        std_0.append(simu_0_array.std())
        skw_0.append(skew(simu_0_array, axis=0, bias=True))
        kurt_0.append(kurtosis(simu_0_array, axis = 0, bias = True))
        rel_dev_0.append((mean_0[i]**2)/simu_0_array.var())
        dist_0.append(distance(simu_0_array, zew))
    #Mean
    mean_0_mean = np.asarray(mean_0).mean()
    mean_0_std = np.asarray(mean_0).std()
    mean_0_conf = norm.interval(0.95, loc=mean_0_mean , scale=mean_0_std)
    model_0_statistics.append([mean_0_mean, mean_0_conf])
    # Standard deviation
    std_0_mean = np.asarray(std_0).mean()
    std_0_std = np.asarray(std_0).std()
    std_0_conf = norm.interval(0.95, loc=std_0_mean , scale=std_0_std )
    model_0_statistics.append([std_0_mean, std_0_conf])
    # Skewness
    skw_0_mean = np.asarray(skw_0).mean()
    skw_0_std = np.asarray(skw_0).std()
    skw_0_conf = norm.interval(0.95, loc=skw_0_mean , scale=skw_0_std )
    model_0_statistics.append([skw_0_mean, skw_0_conf])
    # Kurtosis
    kurt_0_mean = np.asarray(kurt_0).mean()
    kurt_0_std = np.asarray(kurt_0).std()
    kurt_0_conf = norm.interval(0.95, loc=kurt_0_mean , scale=kurt_0_std )
    model_0_statistics.append([kurt_0_mean, kurt_0_conf])
    # Relative Deviation
    rel_dev_0_mean = np.asarray(rel_dev_0).mean()
    rel_dev_0_std = np.asarray(rel_dev_0).std()
    rel_dev_0_conf = norm.interval(0.95, loc=rel_dev_0_mean , scale=rel_dev_0_std )
    model_0_statistics.append([rel_dev_0_mean, rel_dev_0_conf])
    #Distance
    dist_0_mean = np.asarray(dist_0).mean()
    dist_0_std = np.asarray(dist_0).std()
    dist_0_conf = norm.interval(0.95, loc=dist_0_mean , scale=dist_0_std)
    model_0_statistics.append([dist_0_mean, dist_0_conf])

    return model_0_statistics,simu_0
################################################################################################################################################
#  DATA SET 1 (20.12.1991:18.07.2006)
################################################################################################################################################

###########################################################
#           Real Data                                     #
###########################################################
data = data_reader(time_start= 0, time_end= 175)
zew = data.zew()/100
zew_fw = zew[1:]
ip = data.industrial_production()
real_statistics = []

zew_mean = zew.mean(axis = 0)

zew_mean =  zew.mean()
real_statistics.append(zew_mean)

zew_std = zew.std()
real_statistics.append(zew_std)

zew_skw = skew(zew, axis=0, bias=True)
real_statistics.append(zew_skw)

zew_kurt = kurtosis(zew, axis = 0, bias = True)
real_statistics.append(zew_kurt)

zew_rel_dev = (zew_mean**2)/zew.var()
real_statistics.append(zew_rel_dev)

np.savetxt("Estimation/real_statistics_set1.csv", real_statistics, delimiter=",",fmt ='% s')

# ############################################################
# #                       Model 0                            #
# ############################################################
# param = [7.345487901182219392e-01,1.210034441730688075e-02,1.192063076939320343e+00]
# sim_0 = sim.Simulation(N = 175, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= None, alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 0, seed = np.random.random_integers(0,600))
# numSim = 1000

# statistics,simu_0 = get_statistics(sim_0, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_0/sim_statistics_model_0_set1.csv", statistics, delimiter=",",fmt ='% s')
# simu_0 = np.asarray(simu_0)
# np.savetxt("Estimation/Model_0/sim_0_set1.csv", simu_0, delimiter=",",fmt ='% s')

# ############################################################
# #                       Model 1                            #
# ############################################################
# param = [1.428409879579108366e-01,1.099258353878485306e-01,9.985028997509437509e-01,2.155265043587821339e+01]
# sim_1 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= None, alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 1, seed = np.random.random_integers(0,600))
# numSim = 1000
# statistics, simu_1 = get_statistics(sim_1, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_1/sim_statistics_model_1_set1.csv", statistics, delimiter=",",fmt ='% s')
# simu_1 = np.asarray(simu_1)
# np.savetxt("Estimation/Model_1/sim_1_set1.csv", simu_1, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 2                            #
# # ############################################################
# param = [1.218685131203422756e-01,1.248188944999606559e-01,9.650341221432480188e-01,2.016967392187644847e+01,-6.728591198477081647e+00
# ]
# sim_2 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= param[4], alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 2, seed = np.random.random_integers(0,600), y = ip)
# numSim = 1000
# statistics, simu_2 = get_statistics(sim_2, numSim, -0.59, 175)
# np.savetxt("Estimation/Model_2/sim_statistics_model_2_set1.csv", statistics, delimiter=",",fmt ='% s')
# simu_2 = np.asarray(simu_2)
# np.savetxt("Estimation/Model_2/sim_2_set1.csv", simu_2, delimiter=",",fmt ='% s')

############################################################
#                       Model 3                            #
############################################################

# param = [9.047996351698969764e-02,1.495954758529130790e-01,8.745193519211922339e-01,3.187844863450407118e+01,-5.184832000945729824e+00,2.125734982220825575e+00]
# sim_3 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= param[4], alpha3=param[5], deltax= 0.01, deltat= 1/100, model_type= 3, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics,simu_3 = get_statistics(sim_3, numSim, -0.59, 175)
# np.savetxt("Estimation/Model_3/sim_statistics_model_3_set1.csv", statistics, delimiter=",",fmt ='% s')
# simu_3 = np.asarray(simu_3)
# np.savetxt("Estimation/Model_3/sim_3_set1.csv", simu_3, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 4                            #
# # ############################################################

# param = [9.927718324518330917e-02,1.328325475477330764e-01,9.217582211083190646e-01,3.419952589519457575e+01,2.143775742274823148e+00]
# sim_4 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=None, alpha3=param[4], deltax= 0.01, deltat= 1/100, model_type= 4, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics, simu_4 = get_statistics(sim_4, numSim, -0.59, 175)
# np.savetxt("Estimation/Model_4/sim_statistics_model_4_set1.csv", statistics, delimiter=",",fmt ='% s')
# simu_4 = np.asarray(simu_4)
# np.savetxt("Estimation/Model_4/sim_4_set1.csv", simu_4, delimiter=",",fmt ='% s')

# # # ############################################################
# # # #                       Model 5                            #
# # # ############################################################
# param = [6.481269069841348596e-02,1.954777342680067975e-01,7.704788094116641339e-01,2.993329386808641690e+00]
# sim_5 = sim.Simulation(N = 22, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=None, alpha3=param[3], deltax= 0.01, deltat= 1/100, model_type= 5, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics, simu_5 = get_statistics(sim_5, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_5/sim_statistics_model_5_set1.csv", statistics, delimiter=",",fmt ='% s')
# simu_5 = np.asarray(simu_5)
# np.savetxt("Estimation/Model_5/sim_5_set1.csv", simu_5, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 6                           #
# # ############################################################

param = [6.331940450689063637e-02,2.026316488839882413e-01,7.396393454904490738e-01,-7.328853008299797800e+00,2.764240234051995593e+00]
sim_6 = sim.Simulation(N = 22, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=param[3], alpha3=param[4], deltax= 0.01, deltat= 1/100, model_type= 6, seed = np.random.random_integers(0,600), y = ip) 
numSim = 1000
statistics, simu_6 = get_statistics(sim_6, numSim, zew[0], len(zew))
np.savetxt("Estimation/Model_6/sim_statistics_model_6_set1.csv", statistics, delimiter=",",fmt ='% s')
simu_5 = np.asarray(simu_6)
np.savetxt("Estimation/Model_6/sim_6_set1.csv", simu_6, delimiter=",",fmt ='% s')

# ################################################################################################################################################
# #  DATA SET 2 (22.08.2006:15.03.2022)
# ################################################################################################################################################

# ###########################################################
# #           Real Data                                     #
# ###########################################################
data = data_reader(time_start= 176, time_end= -1)
ip = data.industrial_production()
zew = data.zew()/100
zew = zew[0:len(ip)]
zew_fw = zew[1:]
real_statistics = []

zew_mean = zew.mean(axis = 0)

zew_mean =  zew.mean()
real_statistics.append(zew_mean)

zew_std = zew.std()
real_statistics.append(zew_std)

zew_skw = skew(zew, axis=0, bias=True)
real_statistics.append(zew_skw)

zew_kurt = kurtosis(zew, axis = 0, bias = True)
real_statistics.append(zew_kurt)

zew_rel_dev = (zew_mean**2)/zew.var()
real_statistics.append(zew_rel_dev)

np.savetxt("Estimation/real_statistics_set2.csv", real_statistics, delimiter=",",fmt ='% s')

# ############################################################
# #                       Model 0                            #
# ############################################################
# param = [1.490003417794093732e+00,5.165238326396261528e-03,1.085673714369213005e+00]
# sim_0 = sim.Simulation(N = 175, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= None, alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 0, seed = np.random.random_integers(0,600))
# numSim = 1000
# statistics,simu_0 = get_statistics(sim_0, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_0/sim_statistics_model_0_set2.csv", statistics, delimiter=",",fmt ='% s')
# simu_0 = np.asarray(simu_0)
# np.savetxt("Estimation/Model_0/sim_0_set2.csv", simu_0, delimiter=",",fmt ='% s')
# ############################################################
# #                       Model 1                            #
# ############################################################
# param = [2.419691842964410422e-01,2.865422704207985782e-02,9.168349398705467612e-01,2.244431800720106907e+01]
# sim_1 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= None, alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 1, seed = np.random.random_integers(0,600))
# numSim = 1000
# statistics, simu_1 = get_statistics(sim_1, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_1/sim_statistics_model_1_set2.csv", statistics, delimiter=",",fmt ='% s')
# simu_1 = np.asarray(simu_1)
# np.savetxt("Estimation/Model_1/sim_1_set2.csv", simu_1, delimiter=",",fmt ='% s')

# ############################################################
# #                       Model 2                            #
# ############################################################
# param = [1.265690361344460335e-01,7.198959257050341343e-02,4.866135558191909127e-01,1.342627566862939581e+01,-4.834381737020470915e+00
# ]
# sim_2 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= param[4], alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 2, seed = np.random.random_integers(0,600), y = ip)
# numSim = 1000
# statistics, simu_2 = get_statistics(sim_2, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_2/sim_statistics_model_2_set2.csv", statistics, delimiter=",",fmt ='% s')
# simu_2 = np.asarray(simu_2)
# np.savetxt("Estimation/Model_2/sim_2_set2.csv", simu_2, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 3                            #
# # ############################################################

# param = [1.493466698593123171e-02,-6.493910351455471630e-02,-7.471621546742781561e-01,3.027093689498677076e+00,-9.957757769642693546e+00,4.172092832848550259e+00]
# sim_3 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= param[4], alpha3=param[5], deltax= 0.01, deltat= 1/100, model_type= 3, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics,simu_3 = get_statistics(sim_3, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_3/sim_statistics_model_3_set2.csv", statistics, delimiter=",",fmt ='% s')
# simu_3 = np.asarray(simu_3)
# np.savetxt("Estimation/Model_3/sim_3_set2.csv", simu_3, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 4                            #
# # ############################################################
# param = [1.408948434364905994e-02,-7.568828975300316564e-02,-2.345414365596940887e-01,2.691052862727618677e+00,4.651749873609083430e+00]

# sim_4 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=None, alpha3=param[4], deltax= 0.01, deltat= 1/100, model_type= 4, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics, simu_4 = get_statistics(sim_4, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_4/sim_statistics_model_4_set2.csv", statistics, delimiter=",",fmt ='% s')
# simu_4 = np.asarray(simu_4)
# np.savetxt("Estimation/Model_4/sim_4_set2.csv", simu_4, delimiter=",",fmt ='% s')

# # # ############################################################
# # # #                       Model 5                            #
# # # ############################################################
# param = [1.687044347992531501e-01,4.437435164575289498e-02,8.081715128334789888e-01,6.943424950364560644e-01]
# sim_5 = sim.Simulation(N = 22, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=None, alpha3=param[3], deltax= 0.01, deltat= 1/100, model_type= 5, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics, simu_5 = get_statistics(sim_5, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_5/sim_statistics_model_5_set2.csv", statistics, delimiter=",",fmt ='% s')
# simu_5 = np.asarray(simu_5)
#np.savetxt("Estimation/Model_5/sim_5_set2.csv", simu_5, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 6                           #
# # ############################################################

param = [1.758980936653743443e-01,5.425385194302516367e-02,6.907932872040989380e-01,-1.955077894801849236e+00,5.575924232595009800e-01]
sim_6 = sim.Simulation(N = 22, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=param[3], alpha3=param[4], deltax= 0.01, deltat= 1/100, model_type= 6, seed = np.random.random_integers(0,600), y = ip) 
numSim = 1000
statistics, simu_6 = get_statistics(sim_6, numSim, zew[0], len(zew))
np.savetxt("Estimation/Model_6/sim_statistics_model_6_set2.csv", statistics, delimiter=",",fmt ='% s')
simu_5 = np.asarray(simu_6)
np.savetxt("Estimation/Model_6/sim_6_set2.csv", simu_6, delimiter=",",fmt ='% s')

# ################################################################################################################################################
# #  DATA SET 3 (20.12.1991:15.03.2022)
# ################################################################################################################################################

###########################################################
#           Real Data                                     #
###########################################################
data = data_reader(time_start= 0, time_end= -1)
ip = data.industrial_production()
zew = data.zew()/100
zew = zew[0:len(ip)]
zew_fw = zew[1:]
real_statistics = []

zew_mean = zew.mean(axis = 0)

zew_mean =  zew.mean()
real_statistics.append(zew_mean)

zew_std = zew.std()
real_statistics.append(zew_std)

zew_skw = skew(zew, axis=0, bias=True)
real_statistics.append(zew_skw)

zew_kurt = kurtosis(zew, axis = 0, bias = True)
real_statistics.append(zew_kurt)

zew_rel_dev = (zew_mean**2)/zew.var()
real_statistics.append(zew_rel_dev)

np.savetxt("Estimation/real_statistics_set3.csv", real_statistics, delimiter=",",fmt ='% s')

############################################################
#                       Model 0                            #
############################################################
# param = [9.669776636390868818e-01,9.053076346303937094e-03,1.150891765214812734e+00]
# sim_0 = sim.Simulation(N = 175, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= None, alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 0, seed = np.random.random_integers(0,600))
# numSim = 1000
# statistics,simu_0 = get_statistics(sim_0, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_0/sim_statistics_model_0_set3.csv", statistics, delimiter=",",fmt ='% s')
# simu_0 = np.asarray(simu_0)
# np.savetxt("Estimation/Model_0/sim_0_set3.csv", simu_0, delimiter=",",fmt ='% s')
############################################################
#                       Model 1                            #
############################################################
# param = [1.486484022727795618e-01,6.890586885379698656e-02,9.515725459789808882e-01,1.706971621101071079e+01]
# sim_1 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= None, alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 1, seed = np.random.random_integers(0,600))
# numSim = 1000
# statistics, simu_1 = get_statistics(sim_1, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_1/sim_statistics_model_1_set3.csv", statistics, delimiter=",",fmt ='% s')
# simu_1 = np.asarray(simu_1)
# np.savetxt("Estimation/Model_1/sim_1_set3.csv", simu_1, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 2                            #
# # ############################################################
# param = [1.179441737539584506e-01,1.058794799751657245e-01,7.726591629031178687e-01,1.493749840485062563e+01,-4.616748011303377197e+00
# ]
# sim_2 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= param[4], alpha3= None, deltax= 0.01, deltat= 1/100, model_type= 2, seed = np.random.random_integers(0,600), y = ip)
# numSim = 1000
# statistics, simu_2 = get_statistics(sim_2, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_2/sim_statistics_model_2_set3.csv", statistics, delimiter=",",fmt ='% s')
# simu_2 = np.asarray(simu_2)
# np.savetxt("Estimation/Model_2/sim_2_set3.csv", simu_2, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 3                            #
# # ############################################################

# param = [3.603835661125873863e-02,1.465722907414069298e-01,3.363059051870710969e-01,8.038854823426559548e+00,-5.176101086238097615e+00,3.297306859294678372e+00]

# sim_3 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2= param[4], alpha3=param[5], deltax= 0.01, deltat= 1/100, model_type= 3, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics,simu_3 = get_statistics(sim_3, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_3/sim_statistics_model_3_set3.csv", statistics, delimiter=",",fmt ='% s')
# simu_3 = np.asarray(simu_3)
# np.savetxt("Estimation/Model_3/sim_3_set3.csv", simu_3, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 4                            #
# # ############################################################

# param = [3.704345199123725008e-02,1.326440333420852991e-01,4.847468151983388984e-01,8.219977911128516723e+00,3.492394787584649052e+00]

# sim_4 = sim.Simulation(N = param[3], T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=None, alpha3=param[4], deltax= 0.01, deltat= 1/100, model_type= 4, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics, simu_4 = get_statistics(sim_4, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_4/sim_statistics_model_4_set3.csv", statistics, delimiter=",",fmt ='% s')
# simu_4 = np.asarray(simu_4)
# np.savetxt("Estimation/Model_4/sim_4_set3.csv", simu_4, delimiter=",",fmt ='% s')

# # # ############################################################
# # # #                       Model 5                            #
# # # ############################################################
# param = [1.064613358748091598e-01,8.529230383282541961e-02,8.604054948168503580e-01,1.556818372688179242e+00]
# sim_5 = sim.Simulation(N = 22, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=None, alpha3=param[3], deltax= 0.01, deltat= 1/100, model_type= 5, seed = np.random.random_integers(0,600), y = ip) 
# numSim = 1000
# statistics, simu_5 = get_statistics(sim_5, numSim, zew[0], len(zew))
# np.savetxt("Estimation/Model_5/sim_statistics_model_5_set3.csv", statistics, delimiter=",",fmt ='% s')
# simu_5 = np.asarray(simu_5)
# np.savetxt("Estimation/Model_5/sim_5_set3.csv", simu_5, delimiter=",",fmt ='% s')

# # ############################################################
# # #                       Model 6                           #
# # ############################################################

param = [4.931107104610878145e-02,1.374961485430753472e-01,7.009806378582285058e-01,-2.019033674796051336e+00,4.821084414937820029e+00]

sim_6 = sim.Simulation(N = 22, T = 1, nu = param[0], alpha0= param[1], alpha1= param[2], alpha2=param[3], alpha3=param[4], deltax= 0.01, deltat= 1/100, model_type= 6, seed = np.random.random_integers(0,600), y = ip) 
numSim = 1000
statistics, simu_6 = get_statistics(sim_6, numSim, zew[0], len(zew))
np.savetxt("Estimation/Model_6/sim_statistics_model_6_set3.csv", statistics, delimiter=",",fmt ='% s')
simu_5 = np.asarray(simu_6)
np.savetxt("Estimation/Model_6/sim_6_set3.csv", simu_6, delimiter=",",fmt ='% s')