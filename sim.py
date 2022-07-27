''' Class for Simulating the social model'''
# Import Packages
from model import OpinionFormation
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate

class Simulation(OpinionFormation):

    def __init__(self,N: int, T:int, nu: float, alpha0: float, alpha1: float, alpha2:float , alpha3:float, model_type:int, deltax: float, deltat: float, seed:int, y=0) -> None: 
        """
        Initialize the model class with listed input parameters. Furthermore generate empty ararys for the used variables

        Args:
            N (int): Number of Agents
            T (int): Total Amount of Time Steps
            nu (float): Flexibility Parameter
            alpha0 (float): Preference Parameter
            alpha1 (float): Adaptation Parameter
            alpha2 (float):
            alpha3 (float):
            y (float): 
            model_type(int):
            deltax (float): Discretization in space
            deltat (float): Discretization in time


        Return: 
            A reference to the newly created object
        """
        super().__init__(N, T, nu, alpha0, alpha1, alpha2, alpha3, deltax, deltat, model_type) 
        self.y = y
        self.seed = seed
        
    def eulermm(self, ic: float)    -> np.array: 
        """
        The eulermm function takes in a drift and diffusion function, as well as an initial condition. 
        It then uses the Euler-Maruyama method to numerically solve for the SDE. The eulermm function returns 
        the solution at each time step.
        
        Args:
            self: Refer to the object instance itself
            ic (float): Pass the initial condition of the simulation
        
        Return: 
            np.array: A vector of length t with the simulated paths
        """
        dt=self.dt
        t = self.t
        NumTstep = t.size

        sqrtdt = np.sqrt(dt)
        dummy = np.zeros(NumTstep)
        dummy[0] = ic

        est = np.zeros(self.T)
        est[0] = ic
        a = np.arange(0, self.T/ self.dt, step = 1/ self.dt)

        for i in range(1,NumTstep):
             # Set Random Seed
            np.random.seed(self.seed+i)
            dummy[i] = dummy[i-1] + (self.drift(dummy[i-1])) * dt + (np.sqrt(self.diffusion(dummy[i-1])))*np.random.normal(loc=0.0,scale=sqrtdt)
            # Take only the values at the integer t values 
            if i in a: 
                est[int(i*self.dt)] = dummy[i]  
       
        return est
    
    def simulation(self, initial_value: float, sim_length:int):
        time_series = []

        for i in tqdm(range(sim_length)):

            if i == 0:
                # Calculate the PDF 
                if self.model_type == 0:    
                    pdf = self.CrankNicolson(x_0 = initial_value)
                elif self.model_type == 1: 
                    pdf = self.CrankNicolson(x_0 = initial_value)
                elif self.model_type == 2: 
                    pdf = self.CrankNicolson(x_0 = initial_value, y = self.y[i])
                # Calculate the CDF
                cdf = np.cumsum(pdf)
                # Norm the CDF
                cdf = cdf/cdf[-1]
                # Draw a random uniform number
                u = np.around(np.random.uniform(), decimals= 10)
                # Interpolate the Function 
                cdf_new = interpolate.interp1d(cdf, self.x)
                # Insert u 
                time_series.append(initial_value)
                time_series.append(np.around(cdf_new(u),2))
            else:
                # Calculate the PDF 
                if self.model_type == 0:    
                    pdf = self.CrankNicolson(x_0 = time_series[i])
                elif self.model_type == 1: 
                    pdf = self.CrankNicolson(x_0 = time_series[i])
                elif self.model_type == 2: 
                    pdf = self.CrankNicolson(x_0 = time_series[i], y = self.y[i])

                # Calculate the CDF
                cdf = np.cumsum(pdf)
                # Norm the CDF
                cdf = cdf/cdf[-1]
                # Draw a random uniform number
                u = np.around(np.random.uniform(), decimals= 10)
                # Interpolate the Function 
                cdf_new = interpolate.interp1d(cdf, self.x)
                # Insert u 
                time_series.append(np.around(cdf_new(u),2))
        return time_series
    
