import numpy as np 
from math import *
from errors import * 
from scipy.integrate import simps
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt 
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class OpinionFormation():
    
    # Initialize the class
    def __init__(self, N: int, T:int, nu: float, alpha0: float, alpha1: float, alpha2:float, alpha3:float, deltax: float, deltat: float, model_type: int) -> None:
        """ Initialize the model class with listed input parameters. Furthermore generate empty ararys for the used variables
        Args:
            N (int): Number of Agents
            T (int): Total Amount of Time
            nu (float): Flexibility Parameter
            alpha0 (float): Preference Parameter
            alpha1 (float): Adaptation Parameter
            alpha2 (float): Assessment of the business cycle
            alpha3 (float): Momentum Effect
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

        self.model_type = model_type
        self.dx     = deltax
        self.dt     = deltat 
        
        # Model Parameter to be generated
        self.x      = np.arange(-1,1+self.dx,self.dx, dtype = 'd')
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
    
    def drift(self, x: float, y=0, x_l = 0 ) -> float:
        """
        The drift function is used to calculate the drift of a particle. 
        Args:
            x (float): Pass the current position of the particle
        Returns:
            float: The drift value
        """
        if self.model_type == 0:
            return (2*self.nu) * (np.sinh(self.alpha0 + self.alpha1 * x) - (x * np.cosh(self.alpha0 + self.alpha1*x)))
        elif self.model_type == 1: 
            return (2*self.nu) * (np.sinh(self.alpha0 + self.alpha1 * x) - (x * np.cosh(self.alpha0 + self.alpha1*x)))
        elif self.model_type == 2: 
            return (2*self.nu) * (np.sinh(self.alpha0 + self.alpha1 * x + self.alpha2 * y) - (x * np.cosh(self.alpha0 + self.alpha1*x + self.alpha2 * y)))
        elif self.model_type == 3 or self.model_type == 6: 
            return 2 * self.nu*(np.sinh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3*(x - x_l)) - x * np.cosh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3*(x - x_l))) 
        elif self.model_type == 4 or self.model_type ==5: 
            return 2 * self.nu*(np.sinh(self.alpha0 + self.alpha1 * x + self.alpha3*(x - x_l)) - x * np.cosh(self.alpha0 + self.alpha1 * x + self.alpha3*(x - x_l)))    
        

    def diffusion(self, x: float, y = 0, x_l = 0) -> float:
        
        """ The diffusion function takes a value x and returns the change in that value after one time step.
        Args:
            x (float): The input to the diffusion function. This is typically the current position of the particle
        Returns:
            float: The output from the diffusion function
        
        """
        if self.model_type == 0:
            return  (2 * self.nu/self.N) *(np.cosh(self.alpha0 + self.alpha1 * x) - (x * np.sinh(self.alpha0 + self.alpha1*x)))
        elif self.model_type == 1: 
            return   (2 * self.nu/self.N) *(np.cosh(self.alpha0 + self.alpha1 * x) - (x * np.sinh(self.alpha0 + self.alpha1*x))) 
        elif self.model_type == 2: 
            return  (2 * self.nu/self.N) *(np.cosh(self.alpha0 + self.alpha1 * x + self.alpha2*y) - (x * np.sinh(self.alpha0 + self.alpha1*x + self.alpha2*y)))
        elif self.model_type == 3 or self.model_type == 6: 
            return (2 * self.nu/self.N)*(np.cosh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3*(x - x_l)) - x * np.sinh(self.alpha0 + self.alpha1 * x + self.alpha2*y + self.alpha3*(x - x_l))) 
        elif self.model_type == 4 or self.model_type ==5: 
            return (2 * self.nu/self.N)*(np.cosh(self.alpha0 + self.alpha1 * x + self.alpha3*(x - x_l)) - x * np.sinh(self.alpha0 + self.alpha1 * x  + self.alpha3*(x - x_l))) 
        
    # Define the functions for the initial distribution 
    
    def normalDistributionPDF(self,mean: float, sd:float, x:float) -> float: 

        return (1/(sd*np.sqrt(2*np.pi))) * np.exp((-1/2)* ((x-mean)/sd)**2)  
        
    def initialDistribution(self, x_initial:float, y = 0, x_l = 0) -> np.array:
        """ Calculates the initial distribution of the probability
        Returns:
            array: The values of the initial Probability at t=0 for every x
        """

        pdf = np.zeros(len(self.prob))
 
        for i in range(0,len(pdf)):
            mean = ((x_initial + ((self.drift(x = x_initial, y = y, x_l = x_l)) * self.dt)))
            sd =(np.sqrt(((self.diffusion(x_initial, y = y, x_l = x_l)*self.dt))))
            pdf[i] = self.normalDistributionPDF(mean, sd, self.x[i]) 
        return pdf 

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

        ########################################################
        #                    Lux 2012 Scheme                   #         
        ########################################################
        #Fixed Parametes and Vecotors 
        # For the first order derivative 
        p_1 = self.dt/(8.00*self.dx)
        # For the second order derivative
        p_2 = self.dt/(2.0*(self.dx**2))

        # Initialize the Matrix for the solver 
        lhs = np.zeros([len(self.x), len(self.x)]) # LHS Matrix
        rhs = np.zeros([len(self.x), len(self.x)]) # RHS Matrix
        
        # Functions Inside the Fokker Planck Equation
        def g(x):
            return  (1.0/(2.0)) * self.diffusion(x, y = y, x_l= x_l)

        def mu(x):
            return  (-1.0) * self.drift(x, y = y, x_l= x_l )

        # Fill the matrices
        x = self.x
        for elem in range(len(self.prob)):

            if elem == 0:

                lhs[elem, elem] = (1 - p_1*(mu(x[elem+1]) + mu(x[elem])) +  p_2 * g(x = x[elem]))
                lhs[elem, elem+1] = (-p_1* (mu(x[elem+1]) + mu(x[elem])) - p_2*g(x = x[elem+1]))

                rhs[elem, elem] = (1 + p_1*(mu(x[elem+1]) + mu(x[elem])) - p_2 * g(x = x[elem]))
                rhs[elem, elem+1] = (p_1* (mu(x[elem+1]) + mu(x[elem])) + p_2*g(x = x[elem+1]))

            
            elif elem == len(self.x)-1:
                lhs[elem,elem-1] = (p_1 * (mu(x = x[elem]) + mu(x[elem-1])) - p_2 * g(x = x[elem-1]))
                lhs[elem, elem] = (1 + p_1*(mu(x[elem]) + mu(x[elem-1])) +  p_2 * g(x = x[elem]))

                rhs[elem,elem-1] = (-p_1 * (mu(x = x[elem]) + mu(x[elem-1])) + p_2 * g(x = x[elem-1]))
                rhs[elem, elem] = (1 - p_1*(mu(x[elem]) + mu(x[elem-1])) - p_2 * g(x = x[elem]))
                            
            else:                 
                lhs[elem,elem-1] = (p_1 * (mu(x[elem]) + mu(self.x[elem-1])) - p_2 * g(x = self.x[elem-1]))
                lhs[elem, elem] = (1 - p_1*(mu(x[elem+1]) - mu(self.x[elem-1])) +  2* p_2 * g(x = self.x[elem]))
                lhs[elem, elem+1] = (-p_1* (mu(x[elem+1]) + mu(self.x[elem])) - p_2*g(x = self.x[elem+1]))

                rhs[elem,elem-1] = (-p_1 * (mu(x[elem]) + mu(x[elem-1])) + p_2 * g(x = x[elem-1]))
                rhs[elem, elem] = (1 + p_1*(mu(x[elem+1]) - mu(x[elem-1])) - 2 * p_2 * g(x = x[elem]))
                rhs[elem, elem+1] = (p_1* (mu(x[elem+1]) + mu(x[elem])) + p_2*g(x = x[elem+1]))
        
        ##############################################################################
        #                   Crank-Nicoloson as in Nicolas(2022)                      #
        ##############################################################################
        # # Functions Inside the Fokker Planck Equation
        # def D(x):
        #      return  self.diffusion(x, y = y, x_l= x_l)

        # def A(x):
        #      return  self.drift(x, y = y, x_l= x_l )
        # x = self.x
        # h = self.dx
        # k = self.dt 

        # for elem in range(len(self.x)):

        #     if elem == 0:

        #         lhs[elem, elem] = 4*(h**2) + 2*k * D(x[elem])
        #         lhs[elem, elem+1] = k*h* A(x[elem+1]) - k* D(x[elem+1])

        #         rhs[elem, elem] = 4*(h**2) - 2*k * D(x[elem])
        #         rhs[elem, elem+1] = -k*h* A(x[elem+1]) + k* D(x[elem+1])
            
        #     elif elem == len(self.x)-1:
        #         lhs[elem,elem-1] = -k * D(x[elem-1]) - k*h *A(x[elem-1])
        #         lhs[elem, elem] = 4*(h**2) + 2*k * D(x[elem])

        #         rhs[elem,elem-1] = +k * D(x[elem-1]) + k*h *A(x[elem-1])
        #         rhs[elem, elem] = 4*(h**2) - 2*k * D(x[elem])
        #     else:                 
        #         lhs[elem,elem-1] = -k * D(x[elem-1]) - k*h *A(x[elem-1])
        #         lhs[elem, elem] = 4*(h**2) + 2*k * D(x[elem])
        #         lhs[elem, elem+1] = k*h* A(x[elem+1]) - k* D(x[elem+1])

        #         rhs[elem,elem-1] = +k * D(x[elem-1]) + k*h *A(x[elem-1])
        #         rhs[elem, elem] = 4*(h**2) - 2*k * D(x[elem])
        #         rhs[elem, elem+1] = -k*h* A(x[elem+1]) + k* D(x[elem+1])
                
        # Initial Distribution 
        if self.model_type == 0: 
            self.prob[:,0] = np.abs(self.initialDistribution(x_0))
        elif self.model_type == 1: 
            self.prob[:,0] = np.abs(self.initialDistribution(x_0))
        elif self.model_type == 2:
            self.prob[:,0] = np.abs(self.initialDistribution(x_0, y = y))
        elif self.model_type == 3 or self.model_type == 6:
            self.prob[:,0] = np.abs(self.initialDistribution(x_0, y = y, x_l= x_l))
        elif self.model_type == 4 or self.model_type ==5:
            self.prob[:,0] = np.abs(self.initialDistribution(x_0, x_l= x_l))

        rhs = coo_matrix(rhs).tocsr()
        lhs = coo_matrix(lhs).tocsr()

        if fast_comp == True: 
            area = np.zeros(len(self.t))
            for t in range(1,len(self.t)):
                self.prob[:,t]  = spsolve(lhs, rhs @ (self.prob[:,t-1]))
            return self.prob[:,-1]
        else:
            
            # Check the Stability of the Matrix
            if check_stability == True:
                eigenvalues,_ = np.linalg.eig(np.linalg.inv(lhs) @ rhs)
                if np.abs(eigenvalues).max() > 1.00000000009:             
                    print(np.abs(eigenvalues).max())
                    raise UnstableSolutionMethodError
            else: pass

            # Calulation of the Probability Flow with optional Density Calculation and Analysis
            if calc_dens == True:
                area = np.zeros(len(self.t))
                for t in range(1,len(self.t)): 
                    area[t-1] = simps(self.prob[:,t-1], x = self.x) 
                    if  area[t-1] <= 1 - 0.05 or area[t-1] >= 1 + 0.05:     
                        raise WrongDensityValueError(area[t-1], t-1)
                    else: 
                        self.prob[:,t]  = spsolve(lhs, rhs @ self.prob[:,t-1])
                # uncoment for fancy pictures ;)
                        plt.plot(self.prob[:,t-1])
                plt.show()            
                if converged == False:         
                    return area, self.prob, self.prob[:, -1]
                else: 
                    return area, self.prob[:,-1]
            else: 
                for t in range(1,len(self.t)):
                        self.prob[:,t] = spsolve(lhs, rhs * self.prob[:,t-1])
                if converged == False:         
                    return self.prob, self.prob[:, -1]
                else: 
                    
                    return self.prob[:,-1]