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
        return f'{self.density} -> {self.message} {self.time_step}'
    
class UnstableSolutionMethodError(Exception):
    
    def __init__(self, message = "The eigenvalues lie outside the unit circle") -> None:
        self.message = message
        super().__init__(self.message)
        
    def __str__(self) -> str: 
        return self.message