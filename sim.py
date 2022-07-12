''' Class for Simulating the social model'''
# Import Packages
import model
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

class Simulation(model.OpinionFormation):

    def __init__(self,N: int, T:int, nu: float, alpha0: float, alpha1: float, alpha2:float , alpha3:float, y:float, model_type:int, deltax: float, deltat: float, seed:int) -> None: 
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
        super().__init__(N, T, nu, alpha0, alpha1, alpha2, alpha3, y, deltax, deltat, model_type) 
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
                pdf = self.CrankNicolson(x_0 = initial_value)
                # Calculate the CDF
                cdf = np.cumsum(pdf)
                # Norm the CDF
                cdf = cdf/cdf[-1]
                # Take the inverse of the CDF
                cdf_inv = np.around(cdf.T, decimals = 10)
                # Draw a random uniform number
                u = np.around(np.random.uniform(), decimals= 10)
                # Search for the closest value in the CDF 
                absolute_difference_function = lambda list_value : abs(list_value - u)
                closest_value = min(list(cdf_inv), key=absolute_difference_function)
                # Take the value from the CDF at the Position of the closest value
                for x in range(len(cdf_inv)):
                    if cdf_inv[x] == closest_value:
                        next_value = self.x[x]
                time_series.append(initial_value)
                time_series.append(next_value)
            else:
                # Calculate the PDF 
                pdf = self.CrankNicolson(x_0 = time_series[i])
                # Calculate the CDF
                cdf = np.cumsum(pdf)
                # Norm the CDF
                cdf = cdf/cdf[-1]
                # Take the inverse of the CDF
                cdf_inv = np.around(cdf.T, decimals = 10)
                # plt.plot(cdf_inv)
                # plt.show()

                # Draw a random uniform number
                u = np.around(np.random.uniform(), decimals= 10)
                # Search for the closest value in the CDF 
                absolute_difference_function = lambda list_value : abs(list_value - u)
                closest_value = min(list(cdf_inv), key=absolute_difference_function)
                # Take the value from the CDF at the Position of the closest value
                for x in range(len(cdf_inv)):
                    if cdf_inv[x] == closest_value:
                        next_value = self.x[x]
                time_series.append(next_value)
                
        return time_series
    
