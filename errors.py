class WrongDensityValueError(Exception):
    """ Exception for the occurance of a wrong density fuction

    Args:
        Exception (_type_): Value of the sum of all densities is not equal to one
    """

    def __init__(self, density, time_step, message = "The sum of the densities is unequal to 1 at time step:") -> None:
        self.density = density
        self.message = message
        self.time_step = time_step
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        The function returns a string that contains the density, message, and time step of the object
        :return: The density, message, and time step.
        """
        return f'{self.density} -> {self.message} {self.time_step}'
    
# > This class is used to raise an exception when a solution method is unstable
class UnstableSolutionMethodError(Exception):
    
    def __init__(self, message = "The eigenvalues lie outside the unit circle") -> None:
        """
        The __init__ function is a constructor that initializes the object
        
        :param message: The message to be printed when the Exception occurs, defaults to The eigenvalues
        lie outside the unit circle (optional)
        """
        self.message = message
        super().__init__(self.message)
        
    def __str__(self) -> str: 
        """
        The function returns a string representation of the object
        :return: The message
        """
        return self.message

# It's an exception that's raised when the likelihood function is not complete
class UncompleteLikelihoodError(Exception):
    
    def __init__(self, message = "The Likelihood has missing values") -> None:
        """
        The function __init__ is a constructor that takes in a message and sets it to the message
        attribute of the class
        
        :param message: The message to be printed when the exception occurs, defaults to The Likelihood
        has missing values (optional)
        """
        self.message = message
        super().__init__(self.message)
        
    def __str__(self) -> str: 
        """
        The function returns a string representation of the object
        :return: The message
        """
        return self.message
