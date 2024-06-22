import cython
import numpy as np
cimport numpy as np
from tqdm import tqdm
from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger
from tfg.methods.optim.SV import StormerVerletMethod

NAME = "Conservative Method"
logger = create_logger(NAME)
DTYPE = np.float64

cdef double THRESHOLD_NEWTON = 1e-13

cdef extern from "math.h":
    double sqrt(double x)
    double fabs(double x)

SEED_METHOD = StormerVerletMethod()

#sym_rho,sym_z = smp.symbols("rho z")
#U = smp.Rational(1,2)*(1/sym_rho - sym_rho/smp.sqrt(sym_rho**2 + sym_z**2)**3)**2
#get_ddrho = smp.lambdify((sym_rho,sym_z),-smp.diff(U,sym_rho))
#get_ddz = smp.lambdify((sym_rho,sym_z),-smp.diff(U,sym_z))

class ConservativeMethod(StormerSolvingMethod):
    def __init__(self):
        super().__init__(NAME)
    
    @cython.cdivision(True)
    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def simulate(
        self,
        double[:] initial_values,
        double time_step,
        int iterations,
        bint verbose
        ):

        cdef double rho0,z0,p_rho0,p_z0
        cdef double rho1,z1,rho2,z2
        cdef double[:] seed

        if verbose: logger.info(f"Simulating for {iterations} iterations...")
        rho0 = initial_values[0]
        z0 = initial_values[1]
        p_rho0 = initial_values[2]
        p_z0 = initial_values[3]
        rho1 = rho0 +time_step*p_rho0 + (get_ddrho(rho0, z0)*time_step**2)/2
        z1 = z0 +time_step*p_z0 + (get_ddz(rho0, z0)*time_step**2)/2
        #seed = SEED_METHOD.simulate(initial_values,time_step,2,verbose=False)[-1]
        seed = np.array([rho1,z1],dtype=DTYPE)
        solutions = [initial_values]
        for _ in tqdm(range(iterations),disable=not verbose):
            ### z/rho ###
            # Primero z
            z2 = newtonMethodZ(seed[1],100,z0,z1,rho0,time_step)
            if z2 == -1.0:
                print("Fallado newton")
                break
            
            # Ahora rho
            rho2 = newtonMethodRHO(seed[0],100,rho0,rho1,z2,time_step)
            if rho2 == -1.0:
                print("Fallado newton")
                break
            
            #############
            
            ### rho/z ###

            # Primero rho
            # rho2 = newtonMethodRHO(seed[0],100,rho0,rho1,z0,time_step)
            
            # Ahora z   
            # z2 = newtonMethodZ(seed[1],100,z0,z1,rho2,time_step)

            #############
            
            discrete_sol = np.array([rho1,z1,(rho2-rho0)/time_step,(z2-z0)/time_step],dtype=DTYPE)
            rho0 = rho1
            rho1 = rho2
            z0 = z1
            z1 = z2

            #discrete_sol = np.array([rho0,z0,(rho1-rho0)/time_step,(z1-z0)/time_step],dtype=DTYPE)
            #seed = SEED_METHOD.simulate(discrete_sol,time_step,2,verbose=False)[-1]
            seed = np.array([rho1,z1],dtype=DTYPE)
            solutions.append(discrete_sol)
        return np.array(solutions)

@cython.cdivision(True)
@cython.cpow(True)
cdef double get_ddrho(double rho,double z) noexcept:
    return -1.0/2.0*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) \
                + rho**(-1.0))*(6.0*rho**2.0/(rho**2.0 + z**2.0)**(5.0/2.0) \
                - 2.0/(rho**2.0 + z**2.0)**(3.0/2.0) - 2.0/rho**2.0)

@cython.cdivision(True)
@cython.cpow(True)
cdef double get_ddz(double rho,double z) noexcept:
    return -3.0*rho*z*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))/(rho**2.0 + z**2.0)**(5.0/2.0)

@cython.cdivision(True)
@cython.cpow(True)
cdef double F1(double rho_n,double rho_1n,double rho_2n,double z_n,double h) noexcept:
    return ((1.0/2.0)*(-rho_2n/(rho_2n**2.0 + z_n**2.0)**(3.0/2.0) \
                    + rho_2n**(-1.0))**2.0 - 1.0/2.0*(-rho_n/(rho_n**2.0 \
                    + z_n**2.0)**(3.0/2.0) + rho_n**(-1.0))**2.0)/(rho_2n - rho_n) \
                    + (rho_2n - 2.0*rho_1n + rho_n)/h**2.0

@cython.cdivision(True)
@cython.cpow(True)
cdef double F2(double z_n,double z_1n,double z_2n,double rho_2n,double h) noexcept:
    return ((1.0/2.0)*(-rho_2n/(z_2n**2.0 + rho_2n**2.0)**(3.0/2.0) \
                    + rho_2n**(-1.0))**2.0 - 1.0/2.0*(-rho_2n/(rho_2n**2.0 + z_n**2.0)**(3.0/2.0) \
                    + rho_2n**(-1.0))**2.0)/(z_2n - z_n) \
                    + (z_2n - 2.0*z_1n + z_n)/h**2.0

@cython.cdivision(True)
@cython.cpow(True)
cdef double F1_diff(double rho_n,double rho_1n,double rho_2n,double z_n,double h) noexcept:
    return (1.0/2.0)*(-rho_2n/(rho_2n**2.0 + z_n**2.0)**(3.0/2.0) \
                + rho_2n**(-1.0))*(6.0*rho_2n**2.0/(rho_2n**2.0 + z_n**2.0)**(5.0/2.0) \
                - 2.0/(rho_2n**2.0 + z_n**2.0)**(3.0/2.0) - 2.0/rho_2n**2.0)/(rho_2n - rho_n) \
                - ((1.0/2.0)*(-rho_2n/(rho_2n**2.0 + z_n**2.0)**(3.0/2.0) + rho_2n**(-1.0))**2.0 \
                - 1.0/2.0*(-rho_n/(rho_n**2.0 + z_n**2.0)**(3.0/2.0) + rho_n**(-1.0))**2.0)/(rho_2n - rho_n)**2.0 \
                + h**(-2.0)

@cython.cdivision(True)
@cython.cpow(True)
cdef double F2_diff(double z_n,double z_1n,double z_2n,double rho_2n,double h) noexcept:
    return 3.0*z_2n*rho_2n*(-rho_2n/(z_2n**2.0 + rho_2n**2.0)**(3.0/2.0) \
                        + rho_2n**(-1.0))/((z_2n - z_n)*(z_2n**2.0 \
                        + rho_2n**2.0)**(5.0/2.0)) - ((1.0/2.0)*(-rho_2n/(z_2n**2.0 + rho_2n**2.0)**(3.0/2.0) \
                        + rho_2n**(-1.0))**2.0 - 1.0/2.0*(-rho_2n/(rho_2n**2.0 + z_n**2.0)**(3.0/2.0) \
                        + rho_2n**(-1.0))**2.0)/(z_2n - z_n)**2.0 + h**(-2.0)

@cython.cdivision(True)
@cython.cpow(True)
cdef double newtonMethodRHO(double x0,int iterationNumber,double rho_n,double rho_1n,double z_n,double h) noexcept:
    cdef double x,f_eval,counter
    #x=x0
    counter = 0
    f_eval = F1(rho_n,rho_1n,x0,z_n,h)
    for _ in range(iterationNumber):
        x=x0-f_eval/F1_diff(rho_n,rho_1n,x0,z_n,h)
        f_eval = F1(rho_n,rho_1n,x,z_n,h)
        if fabs((x-x0)/x0) <= THRESHOLD_NEWTON:
            break
        x0 = x
        counter+=1
    return -1.0 if counter == iterationNumber else x

@cython.cdivision(True)
@cython.cpow(True)
cdef double newtonMethodZ(double x0,int iterationNumber,double z_n,double z_1n,double rho_2n,double h) noexcept:
    cdef double x,f_eval,counter
    #x=x0
    counter = 0
    f_eval = F2(z_n, z_1n, x0, rho_2n, h)
    for _ in range(iterationNumber):
        x=x0-f_eval/F2_diff(z_n, z_1n, x0, rho_2n, h)
        f_eval = F2(z_n, z_1n, x, rho_2n, h)
        if fabs((x-x0)/x0) <= THRESHOLD_NEWTON:
            break
        x0 = x
        counter+=1
    return -1.0 if counter == iterationNumber else x