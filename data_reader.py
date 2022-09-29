import pandas as pd
import wget
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter
import os.path
import numpy as np



class data_reader():
    
    def __init__(self, time_end=int, time_start= int):
        """
        This function takes in two integers, time_start and time_end, and sets them as the instance
        variables time_start and time_end.
        
        :param time_end: The time the user wants to end the timer
        :param time_start: The time the user wants to start the timer
        """
        self.time_start = time_start
        self.time_end = time_end
    def zew(self):
        """
        It downloads the data from the URL, stores it in a file called ZEW_Sentiment.xls, and then reads
        the data from the file and stores it in a numpy array. 
        
        The function is called by the following line of code:
        :return: the ZEW indicator of economic sentiment for Germany.
        """
        
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
        """
        This function takes the log of the industrial production index of Germany, and then applies a
        14400 day (40 year) Hodrick-Prescott filter to the data. 
        
        The function returns the cyclical component of the data. 
        
        The function takes two arguments: 
        
        1. start_period: the date from which the data is to be extracted. 
        2. hp_filter: whether or not to apply the Hodrick-Prescott filter. 
        
        The function returns the cyclical component of the data. 
        
        The function is called by the following code: 
        
        # Python
        ip = industrial_production(self, start_period = "1991-12-01", hp_filter = True)
        
        :param start_period: the date from which you want to start the data, defaults to 1991-12-01
        (optional)
        :param hp_filter: If True, the data is filtered with a Hodrick-Prescott filter, defaults to True
        (optional)
        :return: the industrial production index of Germany.
        """
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
    pass