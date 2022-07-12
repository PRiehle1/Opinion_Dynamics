from operator import mod
from unicodedata import decimal
import numpy as np 
from math import *
from errors import * 
from scipy.integrate import simps
from tqdm import tqdm
import matplotlib.pyplot as plt 

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
        self.x      = np.arange(-1,1+self.dx,self.dx, dtype= 'd').T
        self.t      = np.arange(0,T,self.dt, dtype= 'd')
        self.prob   = np.zeros([len(self.x), len(self.t)], dtype= 'd')
    
    # Helper Functions
    def integrate(self, x: np.array, y: np.array) -> float:
        """ Calculates the area under the curve for given coordinates
        Args:
            x (array): X-Coordinate
            y (array): Y-Coordinate
        Returns:
            float: The area under the curve
        """
        area = np.trapz(y=y, x=x)
        return area
    
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

    def transition_rate_up(self, x: float) -> float:
        """ Calculates the Transition Probability for the whole socio-configuration to move upward
        Args:
            x (float): Point in space
        Returns:
            float: Transition Probability for a movement upward
        """
        return self.nu * (1-x)  * np.exp(self.influence_function(x))
    
    def transition_rate_down(self, x: float) -> float:
        """ Calculates the Transition Probability for the whole socio-configuration to move downward
        Args:
            x (float): Point in space
        Returns:
            float: Transition Probability for a movement downward
        """
        return self.nu * (1+x)  * np.exp(((-1)*self.influence_function(x)))
    
    def drift(self, x: float, y=0, x_l = 0 ) -> float:
        """
        The drift function is used to calculate the drift of a particle. 
        Args:
            x (float): Pass the current position of the particle
        Returns:
            float: The drift value
        """
        if self.model_type == 0:
            return 2 * self.nu*(np.cosh(self.alpha0 + self.alpha1 * x)*(np.tanh(self.alpha0 + self.alpha1 * x)-x))
        elif self.model_type == 1: 
            return 
        elif self.model_type == 2: 
            return 2 * self.nu*(np.sinh(self.alpha0 + self.alpha1 * x + self.alpha2*y) - x * np.cosh(self.alpha0 + self.alpha1*x + self.alpha2 *y))
        elif self.model_type == 3: 
            return 2 * self.nu*(np.sinh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3(x - x_l)) - x * np.cosh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3(x - x_l)))   
        

    
    def diffusion(self, x: float, y = 0, x_l = 0) -> float:
        
        """ The diffusion function takes a value x and returns the change in that value after one time step.
        Args:
            x (float): The input to the diffusion function. This is typically the current position of the particle
        Returns:
            float: The output from the diffusion function
        
        """
        if self.model_type == 0:
            return  2 * self.nu*(np.cosh(self.alpha0 + self.alpha1 * x)*(1 - x* np.tanh(self.alpha0 + self.alpha1 * x)))*(1/self.N)
        elif self.model_type == 1: 
            return  2 * self.nu*(np.cosh(self.alpha0 + self.alpha1 * x) - x * np.sinh(self.alpha0 + self.alpha1*x))
        elif self.model_type == 2: 
            return 2 * self.nu*(np.cosh(self.alpha0 + self.alpha1 * x + self.alpha2*y) - x * np.sinh(self.alpha0 + self.alpha1*x + self.alpha2 *y))
        elif self.model_type == 3: 
            return 2 * self.nu*(np.cosh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3(x - x_l)) - x * np.sinh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3(x - x_l))) 
        

    
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
        return  np.exp((-1)*(((x-mean)**2)/(2*variance)))/(np.sqrt(variance)*np.sqrt(2*np.pi))
    
    def normalPDF_2(self, epsilon: float) -> float:
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
        return 1/np.sqrt(2*np.pi) * np.exp((-1/2)*epsilon**2)

    def normalDistributionCDF(self, x: float) -> float: 
        """The normalDistributionCDF function takes a float x as input and returns the cumulative distribution function \
            of the normal distribution with mean μ=0 and standard deviation σ=1.
        Args:
            x (float): Represent the value of x for which we want to calculate the probability
        Returns:
            float: The probability that a random variable x will be less than or equal to x
        """
        return (1.0 + erf(x / sqrt(2.0))) / 2.0
    
    def truncatednormalDistributionPDF(self, x: float, x_0:float, bound_right: float, bound_left: float, y = 0, x_l = 0) -> float: 
        """
        The truncatednormalDistributionPDF function takes in the following parameters: 
        x, x_0, bound_right, and bound_left. It returns a float that represents the value of 
        the truncated normal distribution at point x. The function is defined by:
        
            f(x) = ( 1 / (N * sqrt(2*pi)) ) * exp(-((x-(x0 + drift*dt))^2)/(2*diffusion^2 dt) ) / [ F(bound_right)-F(bound-left)]
            
        where N is the number of steps taken in each simulation path; drift is equal to 1/N times 
        the change in price over time; diffusion is equal to volatility divided by square root of time;  
        
            F represents the cumulative density function for a standard normal distribution with mean 0 and variance 1.  
        Args:
            x (float): Pass the current value of x to the function
            x_0 (float): Define the mean of the normal distribution
            bound_right (float): Define the upper bound of the truncated normal distribution
            bound_left (float): Set the lower bound of the truncated normal distribution
        Returns:
            float: The truncated normal distribution for a given x and its parameters
        """
        
        # Initialize the Variables
            
        drift = self.drift(x, y, x_l) 
            
        diffusion = self.diffusion(x, y, x_l) 
            
        normalDist = self.normalPDF_2((x-(x_0 + drift * self.dt))/np.sqrt(diffusion* self.dt)) 
        
        x_1 = (bound_right-(x_0 + drift * self.dt))/np.sqrt(diffusion*self.dt)    
        x_2 = (bound_left-(x_0 + drift * self.dt))/np.sqrt(diffusion*self.dt)
        
        trunormalDist = (1/np.sqrt(diffusion*self.dt)) * normalDist/(self.normalDistributionCDF(x = x_1) - self.normalDistributionCDF(x = x_2))
        
        return trunormalDist
    
    def initialDistribution(self, x_initial:float, truncated: bool, y = 0, x_l = 0) -> np.array:
        """ Calculates the initial distribution of the probability
        Returns:
            array: The values of the initial Probability at t=0 for every x
        """
        dummy = np.zeros(len(self.prob))

        for i in range(0,len(dummy)):
            if truncated == True:
                dummy[i] = self.truncatednormalDistributionPDF(x = self.x[i] ,x_0 = x_initial, bound_right = 1, bound_left = (-1))   
            else: 
                dummy[i] = self.normalPDF_1(x = self.x[i],mean = x_initial + (self.drift(x = self.x[i], y = y, x_l = x_l) * self.dt), variance= self.diffusion(self.x[i], y = y, x_l = x_l)*self.dt)
        
        return dummy
    
    # Define the functions for the solution of the partial differential equaution
    def CrankNicolson(self, x_0:float, y = 0, x_l = 0, check_stability = False, calc_dens = False, converged =  True, fast_comp = True) -> np.array:
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
        N = self.N 
        prob = self.prob


        # Initialize the Matrix for the solver 
        a = np.zeros([len(x), len(x)]) # LHS Matrix
        b = np.zeros([len(x), len(x)]) # RHS Matrix
        
        # Parameter
        p1 = self.dt/(2*self.dx)
        p2 = self.dt/(4*(self.dx**2))
        
        def Q(x):
            return self.diffusion(x, y, x_l)
        def K(x):
            return self.drift(x, y, x_l) 

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
                b[elem, elem] = 1 - 2*p2*Q(x[elem])
                            
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
        prob[:,0] = np.abs(self.initialDistribution(x_0, truncated= True))

        if fast_comp == True: 
            
           # x = np.zeros(len(self.t))
            for t in range(1,len(self.t)):
                self.prob[:,t] =  np.matmul(a_b,np.abs(self.prob[:,t-1]))  
            
            return self.prob[:,-1]
        else:
            
            # Check the Stability of the Matrix
            if check_stability == True:
                eigenvalues,_ = np.linalg.eig(a_b)
                if np.abs(eigenvalues).max() > 1.00000000009:             
                    print(np.abs(eigenvalues).max())
                    raise UnstableSolutionMethodError
            else: pass

            # Calulation of the Probability Flow with optional Density Calculation and Analysis
            if calc_dens == True:
                area = np.zeros(len(self.t))
                for t in range(1,len(self.t)): 
                    area[t-1] = simps(self.prob[:,t-1], x = self.x)
                    if  area[t-1] <= area[0] - 0.05 or area[t-1] >= area[0] + 0.05:           
                        raise WrongDensityValueError(area[t], t)
                    else: 
                        self.prob[:,t] =  np.matmul(a_b,np.abs(self.prob[:,t-1]))
                if converged == False:         
                    return area, self.prob, self.prob[:, -1]
                else: 
                    return area, self.prob[:,-1]
            else: 
                x = np.zeros(len(self.t))
                #import matplotlib.pyplot as plt 
                for t in range(1,len(self.t)):
                    self.prob[:,t] =  np.matmul(a_b,np.abs(self.prob[:,t-1]))
                if converged == False:         
                    return self.prob, self.prob[:, -1]
                else: 
                    
                    return self.prob[:,-1]