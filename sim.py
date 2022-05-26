''' Class for Simulating the social model'''
# Import Packages
import model
import numpy as np 
from tqdm import tqdm

class Simulation(model.OpinionFormation):

    def __init__(self,N: int, T:int, nu: float, alpha0: float, alpha1: float, deltax: float, deltat: float, seed) -> None: 
        """
        Initialize the model class with listed input parameters. Furthermore generate empty ararys for the used variables

        Args:
            N (int): Number of Agents
            T (int): Total Amount of Time Steps
            nu (float): Flexibility Parameter
            alpha0 (float): Preference Parameter
            alpha1 (float): Adaptation Parameter
            deltax (float): Discretization in space
            deltat (float): Discretization in time
            bright (float): Boundary condition right
            bleft (float):  Boundary condition left

        Return: 
            A reference to the newly created object
        """
        super().__init__(N, T, nu, alpha0, alpha1, deltax, deltat) 
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
        dt=self.dt, t = self.t, NumTstep = t.size

        sqrtdt = np.sqrt(dt)
        dummy = np.zeros(NumTstep)
        dummy[0] = ic

        est = np.zeros(self.T)
        est[0] = ic
        a = np.arange(0, self.T/ self.dt, step = 1/ self.dt)

        for i in range(1,NumTstep):
             # Set Random Seed
            np.random.seed(self.seed+i)
            dummy[i] = dummy[i-1] + (1/self.N*self.drift(dummy[i-1])) * dt + (np.sqrt(1/(self.N**2)*self.diffusion(dummy[i-1])))*np.random.normal(loc=0.0,scale=sqrtdt)
            # Take only the values at the integer t values 
            if i in a: 
                est[int(i*self.dt)] = dummy[i]  
       
        return est

