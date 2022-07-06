from operator import mod
import numpy as np 
from math import *
from errors import * 
from scipy.integrate import simps
from tqdm import tqdm

class OpinionFormation():
    
    # Initialize the class
    def __init__(self, N: int, T:int, nu: float, alpha0: float, alpha1: float, alpha2:float, alpha3:float, y: np.array, deltax: float, deltat: float, model_type: int) -> None:
        """ Initialize the model class with listed input parameters. Furthermore generate empty ararys for the used variables
        Args:
            N (int): Number of Agents
            T (int): Total Amount of Time
            nu (float): Flexibility Parameter
            alpha0 (float): Preference Parameter
            alpha1 (float): Adaptation Parameter
            alpha2 (float): Assessment of the business cycle
            alpha3 (float): Momentum Effect
            y (np.array): Underlying Time Series
            deltax (float): Discretization in space
            deltat (float): Discretization in time
            model_type(int): Type of the Model (0: a2,a3 = 0 & N = 175; 1: a2,a3 = 0; 2: a3 = 0, 3: all parameters endogenous )
        """
 
        # Model input Parameter
        self.N      = N 
        self.T      = T 
        self.nu     = nu 
        self.alpha0 = alpha0
        self.alpha1 = alpha1 
        self.alpha2 = alpha2
        self.alpha3 = alpha3 
        self.y      = y
        self.model_type = model_type
        self.dx     = deltax
        self.dt     = deltat 
        
        # Model Parameter to be generated
        self.x      = np.around(np.arange(-1,1,self.dx, dtype= 'd'), decimals=3 )
        self.t      = np.arange(0,T,self.dt, dtype= 'd')
        self.prob   = np.zeros([len(self.x), len(self.t)], dtype= 'd')
    
    
    # Define the Model Functions

    def influence_function(self, x:float, y = 0, x_l = 0) -> float:
        """
        Calculates the influence based on the point in space, the value of the makro time series and the laged point in space

        Args:
            x (float): Point in space 
            y (float): Point in makro time series
            x_l (float): laged point in space

        Returns:
            float: The value of the influence function
        """
        if self.model_type == 0:
            return self.alpha0 + self.alpha1* x
        elif self.model_type == 1: 
            return self.alpha0 + self.alpha1* x
        elif self.model_type == 2: 
            return self.alpha0 + self.alpha1 * x + self.alpha2 * y
        elif self.model_type == 3: 
            return self.alpha0 + self.alpha1 * x + self.alpha2 * y + self.alpha3*(x - x_l)

    def drift(self, x: float) -> float:
        """
        The drift function is used to calculate the drift of a particle. 
        Args:
            x (float): Pass the current position of the particle
        Returns:
            float: The drift value
        """
        
        return 2 * self.nu*(np.sinh(self.alpha0 + self.alpha1 * x) - x * np.cosh(self.alpha0 + self.alpha1*x))

    def diffusion(self, x: float) -> float:
        
        """ The diffusion function takes a value x and returns the change in that value after one time step.
        Args:
            x (float): The input to the diffusion function. This is typically the current position of the particle
        Returns:
            float: The output from the diffusion function
        
        """
        return  2 * self.nu*(np.cosh(self.alpha0 + self.alpha1 * x) - x * np.sinh(self.alpha0 + self.alpha1*x))
    
    # Define the functions for the initial distribution 
    def normalPDF_1(self, x:float,  mean: float, variance: float) -> float:
        """
        The normalPDF function takes in a float x, the mean of a distribution μ and the variance σ. 
        It returns the value of probability density function for normal distribution at point x.
        Args:
            x (float): Represent the value of x_
            mean (float): Mean of the Distribution
            variance (float): Variance of the Distribution
        Returns:
            float: The value of the normal pdf at a given point
        """
        return 1/np.sqrt(variance*2*np.pi) * np.exp((-1/2)*((x-mean)/np.sqrt(variance))**2)

    
    def initialDistribution(self, x_initial:float) -> np.array:
        """ Calculates the initial distribution of the probability
        Returns:
            array: The values of the initial Probability at t=0 for every x
        """
        dummy = np.zeros(len(self.prob))

        for i in range(0,len(dummy)):
            dummy[i] = self.normalPDF_1(x = self.x[i],mean = x_initial + self.drift(x = self.x[i]) * self.dt, variance= self.diffusion(self.x[i])*self.dt ) 
        return dummy/np.sum(dummy)
    
    # Define the functions for the solution of the partial differential equaution
    def CrankNicolson(self, x_0:float, check_stability = False, calc_dens = False, converged =  True, fast_comp = True) -> np.array:
        """
        The CrankNicolson function takes in the initial conditions and sets up the characteristic matrix
         for a Crank Nicolson simulation. It then solves for each time step using a linear algebra solver.
        Args:
            x_0 (float): The initial condition 
            check_stability (bool): Check the stability of the flow matrix
            calc_dens (bool): Calculate the total Density
            converged (bool): Only return the converged PDF
        Raises:
            UnstableSolutionMethodError: Raises if the Solution is not stable
            WrongDensityValueError: Raises if the Area is lower than some eps 
        Returns:
            np.array: The probability distribution at time t for every point in the domain x
        """
        # Fixed Parametes and Vecotors 
        dx = self.dx
        dt = self.dt 
        x = self.x
        t = self.t 
        N = self.N 
        prob = self.prob


        # Initialize the Matrix for the solver 
        a = np.zeros([len(x), len(x)]) # LHS Matrix
        b = np.zeros([len(x), len(x)]) # RHS Matrix
        
        # Parameter
        p1 = self.dt/(2*self.dx)
        p2 = self.dt/(4*self.N*(self.dx**2))
        
        def Q(x):
            return self.diffusion(x)
        def K(x):
            return self.drift(x) 

        # Fill the matrices
        for elem in range(len(x)):
            if elem == 0:
                a[elem, elem] =   1 + 2*p2*Q(x[elem])
                a[elem, elem+1] = p1 * K(x[elem+1]) - p2 * Q(x[elem+1])

                b[elem, elem] = 1 - 2*p2*Q(x[elem])
                b[elem, elem+1] = -p1 * K(x[elem+1]) + p2 * Q(x[elem+1])

            
            elif elem == len(x)-1:
                a[elem,elem-1] = -p1* K(x[elem-1]) - p2* Q(x[elem-1])
                a[elem, elem] = 1 + 2*p2*Q(x[elem])

                b[elem,elem-1] = p1* K(x[elem-1]) + p2* Q(x[elem-1])
                b[elem, elem] =  1 - 2*p2*Q(x[elem])
                            
            else:                 
                a[elem,elem-1] = -p1* K(x[elem-1]) - p2* Q(x[elem-1])
                a[elem, elem] = 1 + 2*p2*Q(x[elem])
                a[elem, elem+1] = p1 * K(x[elem+1]) - p2 * Q(x[elem+1])

                b[elem,elem-1] = p1* K(x[elem-1]) + p2* Q(x[elem-1])
                b[elem, elem] = 1 - 2*p2*Q(x[elem])
                b[elem, elem+1] =  -p1 * K(x[elem+1]) + p2 * Q(x[elem+1])

        # Inverse of the Matrix 
        a_b = np.matmul(np.linalg.inv(a),b)
        
        # Initial Distribution 
        prob[:,0] = self.initialDistribution(x_0)
        print(len(self.t))
        for t in range(1,len(self.t)):
            self.prob[:,t] =  np.matmul(a_b,np.abs(self.prob[:,t-1]))  
        
        return self.prob,self.prob[:,-1]
