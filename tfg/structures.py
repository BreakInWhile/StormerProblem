from numpy import ndarray
from abc import ABC,abstractmethod
from enum import IntEnum

class ReducedDimension(IntEnum):
    RHO = 0
    Z = 1
    P_RHO = 2
    P_Z = 3

class Dimension(IntEnum):
    X = 0
    Y = 1
    Z = 2
    DX = 3
    DY = 4
    DZ = 5

class StormerSolvingMethod(ABC):
    """
    Abstract class defining the structure of the methods for solving
    Störmer problem.
    """

    def __init__(self,name):
        self.name = name
    
    @abstractmethod
    def simulate(
            self,
            initial_values: ndarray,
            time_step: int,
            iterations: int,
            verbose: bool = True
    ) -> ndarray:
        """
        Simulate the trajectory for given initial values
        """
        pass
    
    @classmethod
    def info(self):
        print(self.name)
