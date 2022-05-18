import numpy as np 
from math import *
from errors import * 

class OpinionFormation(object):
    
    # Initialize the class
    def __init__(self, N: int, T:int, nu: float, alpha0: float, alpha1: float, deltax: float, deltat: float, bright: float, bleft: float) -> None:
        """ Initialize the model class with listed input parameters. Furthermore generate empty ararys for the used variables

        Args:
            N (int): Number of Agents
            T (int): Total Amount of Time
            nu (float): Flexibility Parameter
            alpha0 (float): Preference Parameter
            alpha1 (float): Adaptation Parameter
            deltax (float): Discretization in space
            deltat (float): Discretization in time
            bright (float): Boundary condition right
            bleft (float):  Boundary condition left
        """
        
        # Model input Parameter
        self.N      = N 
        self.T      = T 
        self.nu     = nu 
        self.alpha0 = alpha0
        self.alpha1 = alpha1 
        self.dx     = deltax
        self.dt     = deltat 
        self.br     = bright
        self.bl     = bleft
        
        # Model Parameter to be generated
        self.x      = np.arange(-1,1,self.dx)
        self.t      = np.arange(0,T,self.dt)
        self.prob   = np.zeros([len(self.x), len(self.t)])
    
    # Helper Functions
    def integrate(self, x: np.array, y: np.array):
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
    def transition_probabilitie_up(self, x: float) -> float:
        """ Calculates the Transition Probability for the whole socio-configuration to move upward

        Args:
            x (float): Point in space

        Returns:
            float: Transition Probability for a movement upward
        """
        
        return self.nu * (1-x) * np.exp(self.alpha0 + self.alpha1 * x)
    
    def transition_probabilitie_down(self, x: float) -> float:
        """ Calculates the Transition Probability for the whole socio-configuration to move downward

        Args:
            x (float): Point in space

        Returns:
            float: Transition Probability for a movement downward
        """
        
        return self.nu * (1+x) * np.exp(-1*(self.alpha0 + self.alpha1 * x))
    
    def drift(self, x: float) -> float:
        """
        The drift function is used to calculate the drift of a particle. 

        Args:
            x (float): Pass the current position of the particle

        Returns:
            float: The drift value
        """
        
        return self.transition_probabilitie_up(x) - self.transition_probabilitie_down(x)
    
    def diffusion(self, x: float) -> float:
        
        """ The diffusion function takes a value x and returns the change in that value after one time step.

        Args:
            x (float): The input to the diffusion function. This is typically the current position of the particle

        Returns:
            float: The output from the diffusion function
        
        """
        return self.transition_probabilitie_up(x) + self.transition_probabilitie_down(x)
    # Define the functions for the initial distribution 
    
    def normalPDF(self, x:float,  mean: float, variance: float) -> float:
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

    def normalDistributionCDF(self, x: float) -> float: 
        """The normalDistributionCDF function takes a float x as input and returns the cumulative distribution function \
            of the normal distribution with mean μ=0 and standard deviation σ=1.
        Args:
            x (float): Represent the value of x for which we want to calculate the probability

        Returns:
            float: The probability that a random variable x will be less than or equal to x
        """
  
        return (1.0 + erf(x / sqrt(2.0))) / 2.0
    
    # Define the functions for the solution of the partial differential equaution
    
    def forwardDiffernece(self) -> np.array:
        """ The forwardEuler function takes in a drift function, diffusion function, and initial conditions. It then uses the forward Euler method to solve for the probability distribution of a stock price at time t. 
            The forward Euler method is an explicit finite difference scheme that approximates the solution to an initial value problem with one independent variable using linear interpolation between points on the solution curve. 
            This particular implementation of this scheme solves for probabilities at discrete points in space (i.e., stock prices) rather than continuous values.

        Returns:
            np.array: A matrix with the probability distribution at every point in time
        """

        
        # Draw numbers from the initial distribution
        for i in range(1,len(self.x)-1):
            
            self.prob[i,0] = self.normalPDF(self.x[i], self.drift(self.x[i]) * self.dt, self.diffusion(self.x[i])* self.dt)        
        
        # Set the Boundary Conditions 
        self.prob[0,0] = self.bl
        self.prob[len(self.x)-1,0] = self.br
        
        # Loop over every point in time
        for time in range(len(self.t)-1):   
            
            for elem in range(1,len(self.x)-1):
    
                self.prob[elem,time+1] = self.prob[elem,time]  - self.dt/self.dx *(self.drift(self.x[elem+1])* self.prob[elem+1,time] - self.drift(self.x[elem])* self.prob[elem,time]) \
                    + 1/(2*self.N) * self.dt/(self.dx**2) *(self.diffusion(self.x[elem+1])* self.prob[elem+1,time] - 2*self.diffusion(self.x[elem])* self.prob[elem,time]+ self.diffusion(self.x[elem-1])* self.prob[elem-1,time])
                                                                                                                                  
            # Boundary Conditios
            self.prob[0,time] = self.bl
            self.prob[len(self.x)-1,time] = self.br
            
        return self.prob
                  
    def backwardDifference(self) -> np.array: 
        
        """ The backwardDifference function takes in a drift function, diffusion function, and initial conditions. 
            It then solves the Fokker Planck PDE using the backward difference method. The output is an array of 
            probabilities at each time step for every x value.

        Returns:
            np.array: A matrix of the probability distribution at each time step
        """
    
        # Draw initial Probabilities from the initial distribution
        for i in range(1,len(self.x)-1):
            self.prob[i,0] = self.normalPDF(self.x[i], self.drift(self.x[i]) * self.dx, self.diffusion(self.x[i])* self.dx)        
        # Set the Boundary Conditions 
        self.prob[0,0] = self.bl
        self.prob[len(self.x)-1,0] = self.br
        
        # Initialize the characteristical Matrix 
        a = np.zeros([len(self.x),len(self.x)])
        
        for elem in range(len(self.prob)):
            
            if elem == 0:
                a[elem,elem] = 1
            elif elem == len(self.prob)-1:
                a[elem, elem] = 1
            else: 
                a[elem,elem-1] = (-self.dt/(2*self.N*self.dx**2)) * self.diffusion(self.x[elem-1])
                a[elem,elem] = 1 - (self.dt/self.dx) * self.drift(self.x[elem]) + (self.dt/(self.N*self.dx**2)) * self.diffusion(self.x[elem])
                a[elem,elem+1] = (self.dt/self.dx) * self.drift(self.x[elem+1]) - (self.dt/(2*self.N*self.dx**2)) * self.diffusion(self.x[elem+1])
        
        # Solve the system
        a_inv = np.linalg.pinv(a)
        
        # Check the Stability
        eigenvalues,_ = np.linalg.eig(a_inv)
        for eig in np.abs(eigenvalues):
            if eig > 1.0000000000002:              # Correction based on rounding errors
                print(eig)
                raise UnstableSolutionMethodError
            else: pass
        
        # Loop through all time steps    
        for t in range(1,len(self.t)):
            # Density Check
            area = self.integrate(x = self.x, y= self.prob[:,t-1])
            if  0.99>= area or area >=1.01:                               # Check up to which value the integration is a valid approximation  
                raise WrongDensityValueError(area)
            else:            
                self.prob[:,t] = np.matmul(a_inv, self.prob[:,t-1])
                # Boundary Conditions
                self.prob[0,t] = self.br
                self.prob[len(self.x)-1,t] = self.bl            
        
        return self.prob

    def CrankNicolson(self) -> np.array:
        
        """ The CrankNicolson function takes in the initial conditions and sets up the 
            characteristic matrix for a Crank Nicolson simulation. It then solves for each time step using a linear algebra solver.

        Returns:
            np.array: The probability distribution at time t for every point in the domain x
        """
        
        # Draw initial Probabilities from the initial distribution
        for i in range(1,len(self.x)-1):
            self.prob[i,0] = self.normalPDF(self.x[i], self.drift(self.x[i]) * self.dx, self.diffusion(self.x[i])* self.dx)        
        # Set the Boundary Conditions 
        self.prob[0,0] = self.bl
        self.prob[len(self.x)-1,0] = self.br
        
        # Initialize the characteristical Matrix 
        a = np.zeros([len(self.x),len(self.x)])
        b = np.zeros([len(self.x),len(self.x)])
        
        
        for elem in range(len(self.prob)):
            
            if elem == 0:
                a[elem,elem] = 1
                b[elem,elem] = 1
            elif elem == len(self.prob)-1:
                a[elem, elem] = 1
                b[elem, elem] = 1
            else: 
                b[elem,elem-1] = (-self.dt/(4*self.N*self.dx**2)) * self.diffusion(self.x[elem-1])
                
                b[elem,elem] = 1 - (self.dt/(2*self.dx)) * self.drift(self.x[elem]) + (self.dt/(2*self.N*self.dx**2)) * self.diffusion(self.x[elem])
                
                b[elem,elem+1] = (self.dt/(2*self.dx)) * self.drift(self.x[elem+1]) - (self.dt/(4*self.N*self.dx**2)) * self.diffusion(self.x[elem+1])
                
                a[elem,elem-1] = (self.dt/(4*self.N*self.dx**2)) * self.diffusion(self.x[elem-1])
                
                a[elem,elem] = 1 + (self.dt/(2*self.dx)) * self.drift(self.x[elem]) - (self.dt/(2*self.N*self.dx**2)) * self.diffusion(self.x[elem])
                
                a[elem,elem+1] = (-self.dt/(2*self.dx)) * self.drift(self.x[elem+1]) + (self.dt/(4*self.N*self.dx**2)) * self.diffusion(self.x[elem+1])
        
        a_b = np.matmul(np.linalg.inv(b),a)
        
        # Check the Stability
        eigenvalues,_ = np.linalg.eig(a_b)
        for eig in np.abs(eigenvalues):
            if eig > 1.00000000009:              # Correction based on rounding errors
                print(eig)
                raise UnstableSolutionMethodError
            else: pass
        
        
        for t in range(1,len(self.t)):
            # Density Check
            area = self.integrate(x = self.x, y= self.prob[:,t-1])
            if  0.99>= area or area >=1.01:           
                raise WrongDensityValueError(area)
            else:
                self.prob[:,t] = np.matmul(a_b, self.prob[:,t-1])
                # Boundary Conditions
                self.prob[0,t] = self.br
                self.prob[len(self.x)-1,t] = self.bl            
        
        return self.prob, self.prob[:, -1]
    