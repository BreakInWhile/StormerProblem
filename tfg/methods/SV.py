import numpy as np
from tqdm import tqdm
from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger

logger = create_logger("Störmer-Verlet Method")

class StormerVerletMethod(StormerSolvingMethod):
    """
    Solves the reduced Stormer model
    """
    def __init__(self):
        super().__init__("Störmer-Verlet Method")

    def simulate(self, initial_values: np.ndarray, time_step: int, iterations: int, verbose: bool = True) -> np.ndarray:
        if verbose: logger.info(f"Simulating for {iterations} iterations...")
        solutions = [initial_values]
        for _ in tqdm(range(iterations),disable=not verbose):
            q = solutions[-1][:2]
            p = solutions[-1][2:]
            ff = self._f(q)
            q_n = q + time_step*p + ((time_step**2)/2)*ff
            p_n = p + (time_step/2)*ff + (time_step/2)*self._f(q_n)
            solutions.append(np.concatenate((q_n,p_n)))
        return np.array(solutions)
    
    def _f(self,q):
        rho,z = q
        rad = np.sqrt(rho**2 + z**2)
        first_term = (-1/rho + rho/(rad**3))
        d_ro = first_term \
                * (-1/(rho**2) + 3*(rho**2)/(rad**5) - 1/(rad**3))
        d_z = first_term * 3*rho*z/(rad**5)

        return np.array([d_ro,d_z])
    
class StormerVerletMethod3D(StormerSolvingMethod):
    def __init__(self):
        super().__init__("Störmer-Verlet Method")

    def simulate(self, initial_values: np.ndarray, time_step: int, iterations: int, verbose: bool = True) -> np.ndarray:
        if verbose: logger.info(f"Simulating for {iterations} iterations...")
        solutions = [initial_values]
        for _ in tqdm(range(iterations),disable=not verbose):
            q = solutions[-1][:3]
            p = solutions[-1][3:]
            ff = self._f(q,p)
            q_n = q + time_step*p + ((time_step**2)/2)*ff
            p_n = p + (time_step/2)*ff + (time_step/2)*self._f(q_n,p)
            solutions.append(np.concatenate((q_n,p_n)))
        return np.array(solutions)
    
    def _f(self,q,p):
        x,y,z = q
        dx,dy,dz = p
        rad = np.sqrt(x**2.0 + y**2.0 + z**2.0)
        d2x = (3.0*z/rad**5.0)*(dy*z - dz*y)-dy/rad**3.0
        d2y = -(3.0*z/rad**5.0)*(dx*z - dz*x)+dx/rad**3.0
        d2z = (3.0*z/rad**5.0)*(dx*y - dy*x)
        return np.array([d2x,d2y,d2z])