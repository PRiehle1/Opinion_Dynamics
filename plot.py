import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class Plotting3D(object):
    
    def __init__(self, param: np.array, x: np.array, t:np.array):
        """_summary_

        Args:
            param (np.array): _description_
            x (np.array): _description_
        """
        self.param = param
        self.x = x
        self.t = t
            
    def surface_plot(self):

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
        self.x = x
        self.y = y
          
    def sim_plot(self):
        
        fig, ax = plt.subplots()
        ax.plot(self.x,self.y)
        ax.set(xlabel='t', ylabel='y',
            title='Euler-Maruyama-Method for simulation of the canonical social model ')
        ax.grid()
        plt.show()