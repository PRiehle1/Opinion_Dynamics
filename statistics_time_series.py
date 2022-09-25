from tkinter import N
import sim 
import numpy as np 
from scipy.stats import skew, kurtosis, norm
from data_reader import data_reader
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


###########################################################
#           Real Data      Period 1                       #
###########################################################
data = data_reader(time_start= 0, time_end= 176)
zew = data.zew()/100
zew_fw = zew[1:]
ip = data.industrial_production()
real_statistics = []
real_statistics_ip = []

## Mean ##
zew_mean =  zew.mean()
real_statistics.append(zew_mean)

ip_mean =  ip.mean()
real_statistics_ip.append(ip_mean)

## Std ##
zew_std = zew.std()
real_statistics.append(zew_std)

ip_std = ip.std()
real_statistics_ip.append(ip_std)

## Skewness ##
zew_skw = skew(zew, axis=0, bias=True)
real_statistics.append(zew_skw)

ip_skw = skew(ip, axis=0, bias=True)
real_statistics_ip.append(ip_skw)

## Kurt ##
ip_kurt = kurtosis(ip, axis = 0, bias = True)
real_statistics_ip.append(ip_kurt)

## Rel Deviation ##
zew_rel_dev = (zew_mean**2)/zew.var()
real_statistics.append(zew_rel_dev)

ip_rel_dev = (ip_mean**2)/ip.var()
real_statistics_ip.append(ip_rel_dev)

result = adfuller(zew, autolag = "AIC")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

np.savetxt("Estimation/adf_Period1_zew.csv", result, delimiter=",",fmt ='% s')
np.savetxt("Estimation/real_statistics_Period1_zew.csv", real_statistics, delimiter=",",fmt ='% s')
np.savetxt("Estimation/real_statistics_Period1_ip.csv", real_statistics_ip, delimiter=",",fmt ='% s')
###########################################################
data = data_reader(time_start= 176, time_end= 364)
ip = data.industrial_production()
zew = data.zew()/100
zew = zew[0:len(ip)]
zew_fw = zew[1:]
real_statistics = []
real_statistics_ip = []

## Mean ##
zew_mean =  zew.mean()
real_statistics.append(zew_mean)

ip_mean =  ip.mean()
real_statistics_ip.append(ip_mean)

## Std ##
zew_std = zew.std()
real_statistics.append(zew_std)

ip_std = ip.std()
real_statistics_ip.append(ip_std)

## Skewness ##
zew_skw = skew(zew, axis=0, bias=True)
real_statistics.append(zew_skw)

ip_skw = skew(ip, axis=0, bias=True)
real_statistics_ip.append(ip_skw)

## Kurt ##
ip_kurt = kurtosis(ip, axis = 0, bias = True)
real_statistics_ip.append(ip_kurt)

## Rel Deviation ##
zew_rel_dev = (zew_mean**2)/zew.var()
real_statistics.append(zew_rel_dev)

ip_rel_dev = (ip_mean**2)/ip.var()
real_statistics_ip.append(ip_rel_dev)

result = adfuller(zew)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

np.savetxt("Estimation/adf_Period2_zew.csv", result, delimiter=",",fmt ='% s')

np.savetxt("Estimation/real_statistics_Period2_zew.csv", real_statistics, delimiter=",",fmt ='% s')
np.savetxt("Estimation/real_statistics_Period2_ip.csv", real_statistics_ip, delimiter=",",fmt ='% s')
#############################################################################################
data = data_reader(time_start= 0, time_end= 364)
ip = data.industrial_production()
zew = data.zew()/100
zew = zew[0:len(ip)]
zew_fw = zew[1:]
real_statistics = []
real_statistics_ip = []

## Mean ##
zew_mean =  zew.mean()
real_statistics.append(zew_mean)

ip_mean =  ip.mean()
real_statistics_ip.append(ip_mean)

## Std ##
zew_std = zew.std()
real_statistics.append(zew_std)

ip_std = ip.std()
real_statistics_ip.append(ip_std)

## Skewness ##
zew_skw = skew(zew, axis=0, bias=True)
real_statistics.append(zew_skw)

ip_skw = skew(ip, axis=0, bias=True)
real_statistics_ip.append(ip_skw)

## Kurt ##
ip_kurt = kurtosis(ip, axis = 0, bias = True)
real_statistics_ip.append(ip_kurt)

## Rel Deviation ##
zew_rel_dev = (zew_mean**2)/zew.var()
real_statistics.append(zew_rel_dev)

ip_rel_dev = (ip_mean**2)/ip.var()
real_statistics_ip.append(ip_rel_dev)

result = adfuller(zew)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

np.savetxt("Estimation/adf_Period3_zew.csv", result, delimiter=",",fmt ='% s')

np.savetxt("Estimation/real_statistics_Period3_zew.csv", real_statistics, delimiter=",",fmt ='% s')
np.savetxt("Estimation/real_statistics_Period3_ip.csv", real_statistics_ip, delimiter=",",fmt ='% s')

