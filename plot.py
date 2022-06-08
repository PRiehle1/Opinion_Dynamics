import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class Plotting3D(object):
    
    def __init__(self, param: np.array, x: np.array, t:np.array) -> None:
        """ 
        Initialize the Plotting3D class with listed input parameters

        Args:
            param (np.array): Parameter for the y axis
            x (np.array): Parameter for the z axis 
            t (np.array): Parameter for the x axis
        """
        self.param = param
        self.x = x
        self.t = t
            
    def surface_plot(self) -> plt.plot: 
        """
        The surface_plot function plots the probability density of a 
        parameter as a function of time and space.

        Args:
            self: Access variables that belongs to the class
        Returns:
            Plot: The Surface Plot
        """

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        time, space = np.meshgrid(self.t, self.x)
        
        # Plot the surface.
        surf = ax.plot_surface(time, space, self.param, 
                        linewidth=0, antialiased=False)

        ax.set_xlabel("Time")
        ax.set_zlabel("Probability Density")
        ax.set_ylabel("x")
        
        ax.view_init(20,120)

        plt.show()

class Plotting2D(object):

    def __init__(self, x : np.array, y: np.array) -> None:
        """
        Initialize the Plotting2D class with listed input parameters

        Args:
            x (np.array): The x axis
            y (np.array): The y axis 
        Return: 
            None     
            
        """
        self.x = x
        self.y = y
          
    def sim_plot(self) -> plt.plot:
        """
        The sim_plot function plots the simulation results of the Euler-Maruyama method for a given number of time steps.
        
        Return:
            The simulation plot

        """       
        fig, ax = plt.subplots()
        ax.plot(self.x,self.y)
        ax.set(xlabel='t', ylabel='y',
            title='Euler-Maruyama-Method for simulation of the canonical social model ')
        ax.grid()
        plt.show()