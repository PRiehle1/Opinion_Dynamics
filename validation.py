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


    
############################################################
#               Helper Functions                           #
############################################################
def distance(est_array: np.array, time_series:np.array) -> float:
    dummy = []
    for i in range(0,len(est_array)-1):
        dummy.append(np.abs(est_array[i]- time_series[i]))
    dist = 1/len(time_series) * np.sum(dummy)

    return dist

###########################################################
#           Real Data                                     #
###########################################################
data = data_reader(time_start= 0, time_end= 175)
zew = data.zew()/100
zew_fw = zew[1:]
ip = data.industrial_production()

zew_mean = zew.mean(axis = 0)
zew_mean =  zew.mean()
zew_std = zew.std()
zew_skw = skew(zew, axis=0, bias=True)
zew_kurt = kurtosis(zew, axis = 0, bias = True)
zew_rel_dev = (zew_mean**2)/zew.var()

############################################################
#                       Model 0                            #
############################################################

### Simulations ###
sim_0 = sim.Simulation(N = 175, T = 1, nu = 7.641222632245864288e-01, alpha0= 8.672915546100912546e-03, alpha1= 1.190665833267387841, alpha2= None, alpha3= None, deltax= 0.02, deltat= 1/100, model_type= 0, seed = np.random.random_integers(0,600))
numSim = 1000
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
    simu_0.append(sim_0.simulation(-0.59, 175))
    # Moments
    simu_0_array = np.asarray(simu_0[i])
    mean_0.append(simu_0_array.mean())
    std_0.append(simu_0_array.std())
    skw_0.append(skew(simu_0_array, axis=0, bias=True))
    kurt_0.append(kurtosis(simu_0_array, axis = 0, bias = True))
    rel_dev_0.append((mean_0[i]**2)/simu_0_array.var())
    dist_0.append(distance(simu_0_array, zew))

mean_0_mean = np.asarray(mean_0).mean()
mean_0_std = np.asarray(mean_0).std()
mean_0_conf = norm.interval(0.95, loc=mean_0_mean , scale=mean_0_std)
model_0_statistics.append([mean_0_mean, mean_0_conf])

std_0_mean = np.asarray(std_0).mean()
std_0_std = np.asarray(std_0).std()
std_0_conf = norm.interval(0.95, loc=std_0_mean , scale=std_0_std )
model_0_statistics.append([std_0_mean, std_0_conf])

skw_0_mean = np.asarray(skw_0).mean()
skw_0_std = np.asarray(skw_0).std()
skw_0_conf = norm.interval(0.95, loc=skw_0_mean , scale=skw_0_std )
model_0_statistics.append([skw_0_mean, skw_0_conf])

kurt_0_mean = np.asarray(kurt_0).mean()
kurt_0_std = np.asarray(kurt_0).std()
kurt_0_conf = norm.interval(0.95, loc=kurt_0_mean , scale=kurt_0_std )
model_0_statistics.append([kurt_0_mean, kurt_0_conf])

rel_dev_0_mean = np.asarray(rel_dev_0).mean()
rel_dev_0_std = np.asarray(rel_dev_0).std()
rel_dev_0_conf = norm.interval(0.95, loc=rel_dev_0_mean , scale=rel_dev_0_std)
model_0_statistics.append([rel_dev_0_mean, rel_dev_0_conf])

dist_0_mean = np.asarray(dist_0).mean()
dist_0_std = np.asarray(dist_0).std()
dist_0_conf = norm.interval(0.95, loc=dist_0_mean , scale=dist_0_std )
model_0_statistics.append([dist_0_mean, dist_0_conf])


print(model_0_statistics)
