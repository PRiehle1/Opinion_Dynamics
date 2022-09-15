import pandas as pd
import wget
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter
import os.path
import numpy as np


class data_reader():
    
    def __init__(self, time_end=int, time_start= int):
        self.time_start = time_start
        self.time_end = time_end
    def zew(self):
        
        # Define the URL of the File
        URL = "https://ftp.zew.de/pub/zew-docs/div/konjunktur.xls"
        # 2. download the data behind the URL
        if os.path.exists(r"ZEW_Sentiment.xls") == False:
            response = wget.download(URL, "ZEW_Sentiment.xls")
        # Open the xls file and store the time series
        zew = pd.read_excel("ZEW_Sentiment.xls", sheet_name= "data", names = ["Date","ZEW Indicator of Economic Sentiment Germany, balances"])
        # Change to numpy array
        zew = zew.iloc[:,0].to_numpy()
        zew = zew[self.time_start:self.time_end]
        
        return zew
    
    def industrial_production(self, start_period = "1991-12-01", hp_filter = True):
        
        fred = Fred(api_key='83638cb297f59ce5d318fa8e1deff61b')
        ip = fred.get_series('DEUPROINDMISMEI') 
        ip = ip.loc[start_period:].to_numpy()

        ip = np.log(ip[self.time_start:self.time_end])
        
        if hp_filter == True: 
            ip_cyle,_ = hpfilter(ip, 14400)
            return ip_cyle
        else: 
            return ip
        
        
        
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    reader = data_reader(175)
    #zew = reader.zew()
    ip = reader.industrial_production()
    plt.plot(ip/100)
    plt.show()
