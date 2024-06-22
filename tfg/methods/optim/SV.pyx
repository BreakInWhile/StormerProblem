import cython
import numpy as np
cimport numpy as np
from tqdm import tqdm
from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger

logger = create_logger("Störmer-Verlet Method")
DTYPE = np.float64

cdef extern from "math.h":
    double sqrt(double x)

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double f1(double rho,double z,double rad,double first_term):
    cdef double d_rho
    #first_term = (-1/rho + rho/(rad**3))
    d_rho = first_term * (-1/(rho**2) + 3*(rho**2)/(rad**5) - 1/(rad**3))
    return d_rho

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double f2(double rho,double z,double rad,double first_term):
    cdef double d_z
    #first_term = (-1/rho + rho/(rad**3))
    d_z = first_term * 3*rho*z/(rad**5)
    return d_z

# TODO: Quitar
def fun1(a,b,rad,ft):
    return f1(a,b,rad,ft)

# TODO: Quitar
def fun2(a,b,rad,ft):
    return f2(a,b,rad,ft)

class StormerVerletMethod(StormerSolvingMethod):
    def __init__(self):
        super().__init__("Störmer-Verlet Method")

    @cython.cdivision(True)
    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def simulate(
            self,
            double[:] initial_values,
            double time_step,
            int iterations,
            bint verbose):
        
        cdef list solutions
        cdef double rho,z,p_rho,p_z
        cdef double sol_f1,sol_f2,rad,first_term

        if verbose: logger.info(f"Simulating for {iterations} iterations...")
        rho = initial_values[0]
        z = initial_values[1]
        p_rho = initial_values[2]
        p_z = initial_values[3] 
        solutions = [np.array([rho,z,p_rho,p_z],dtype=DTYPE)]
        for _ in tqdm(range(iterations),disable=not verbose):
            rad = sqrt(rho**2 + z**2)
            first_term = (-1/rho + rho/(rad**3))
            sol_f1 = f1(rho,z,rad,first_term)
            sol_f2 = f2(rho,z,rad,first_term)

            rho = rho + time_step*p_rho + ((time_step**2)/2)*sol_f1
            z = z + time_step*p_z + ((time_step**2)/2)*sol_f2

            rad = sqrt(rho**2 + z**2)
            first_term = (-1/rho + rho/(rad**3))
            nsol_f1 = f1(rho,z,rad,first_term)
            nsol_f2 = f2(rho,z,rad,first_term)
            p_rho = p_rho + (time_step/2)*sol_f1 + (time_step/2)*nsol_f1
            p_z = p_z + (time_step/2)*sol_f2 + (time_step/2)*nsol_f2

            solutions.append(np.array((rho,z,p_rho,p_z),dtype=DTYPE))
        return np.array(solutions)
    

