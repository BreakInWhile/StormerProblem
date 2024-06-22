import numpy as np
import sympy as smp
from tqdm import tqdm
from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger
from tfg.methods.optim.SV import StormerVerletMethod

NAME = "Strauss-Vázquez"
logger = create_logger(NAME)
DTYPE = np.float64

SEED_METHOD = StormerVerletMethod()

#sym_rho,sym_z = smp.symbols("rho z")
#U = smp.Rational(1,2)*(1/sym_rho - sym_rho/smp.sqrt(sym_rho**2 + sym_z**2)**3)**2
#get_ddrho = smp.lambdify((sym_rho,sym_z),-smp.diff(U,sym_rho))
#get_ddz = smp.lambdify((sym_rho,sym_z),-smp.diff(U,sym_z))

class ConservativeMethod(StormerSolvingMethod):
    def __init__(self):
        super().__init__(NAME)

    def simulate(self, initial_values: np.ndarray, time_step: int, iterations: int, verbose: bool = True) -> np.ndarray:
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
            
            # Ahora rho
            rho2 = newtonMethodRHO(seed[0],100,rho0,rho1,z2,time_step)
            
            #############
            
            ### rho/z ###

            # Primero rho
            # rho2 = newtonMethodRHO(seed[0],100,rho0,rho1,z0,time_step)
            
            # Ahora z   
            # z2 = newtonMethodZ(seed[1],100,z0,z1,rho2,time_step)

            #############

            rho0 = rho1
            rho1 = rho2
            z0 = z1
            z1 = z2

            discrete_sol = np.array([rho0,z0,(rho1-rho0)/time_step,(z1-z0)/time_step],dtype=DTYPE)
            #seed = SEED_METHOD.simulate(discrete_sol,time_step,2,verbose=False)[-1]
            seed = np.array([rho1,z1],dtype=DTYPE)
            solutions.append(discrete_sol)
        return np.array(solutions)
    
def get_ddrho(rho, z):
    return -1.0/2.0*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) \
                + rho**(-1.0))*(6.0*rho**2.0/(rho**2.0 + z**2.0)**(5.0/2.0) \
                - 2.0/(rho**2 + z**2.0)**(3.0/2.0) - 2.0/rho**2.0)

def get_ddz(rho, z):
    return -3.0*rho*z*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))/(rho**2.0 + z**2.0)**(5.0/2.0)
    
def F1(rho_n, rho_1n, rho_2n, z_n, h):
    return ((1/2)*(-rho_2n/(rho_2n**2 + z_n**2)**(3/2) \
                    + rho_2n**(-1.0))**2 - 1/2*(-rho_n/(rho_n**2 \
                    + z_n**2)**(3/2) + rho_n**(-1.0))**2)/(rho_2n - rho_n) \
                    + (rho_2n - 2*rho_1n + rho_n)/h**2

def F2(z_n, z_1n, z_2n, rho_2n, h):
    return ((1/2)*(-rho_2n/(z_2n**2 + rho_2n**2)**(3/2) \
                    + rho_2n**(-1.0))**2 - 1/2*(-rho_2n/(rho_2n**2 + z_n**2)**(3/2) \
                    + rho_2n**(-1.0))**2)/(z_2n - z_n) \
                    + (z_2n - 2*z_1n + z_n)/h**2

def F1_diff(rho_n, rho_1n, rho_2n, z_n, h):
    return (1/2)*(-rho_2n/(rho_2n**2 + z_n**2)**(3/2) \
                + rho_2n**(-1.0))*(6*rho_2n**2/(rho_2n**2 + z_n**2)**(5/2) \
                - 2/(rho_2n**2 + z_n**2)**(3/2) - 2/rho_2n**2)/(rho_2n - rho_n) \
                - ((1/2)*(-rho_2n/(rho_2n**2 + z_n**2)**(3/2) + rho_2n**(-1.0))**2 \
                - 1/2*(-rho_n/(rho_n**2 + z_n**2)**(3/2) + rho_n**(-1.0))**2)/(rho_2n - rho_n)**2 \
                + h**(-2.0)

def F2_diff(z_n, z_1n, z_2n, rho_2n, h):
    return 3*z_2n*rho_2n*(-rho_2n/(z_2n**2 + rho_2n**2)**(3/2) \
                        + rho_2n**(-1.0))/((z_2n - z_n)*(z_2n**2 \
                        + rho_2n**2)**(5/2)) - ((1/2)*(-rho_2n/(z_2n**2 + rho_2n**2)**(3/2) \
                        + rho_2n**(-1.0))**2 - 1/2*(-rho_2n/(rho_2n**2 + z_n**2)**(3/2) \
                        + rho_2n**(-1.0))**2)/(z_2n - z_n)**2 + h**(-2.0)

def newtonMethodRHO(x0,iterationNumber,rho_n, rho_1n, z_n, h):
    x=x0
    f_eval = F1(rho_n,rho_1n,x,z_n,h)
    for _ in range(iterationNumber):
        x=x-f_eval/F1_diff(rho_n,rho_1n,x,z_n,h)
        f_eval = F1(rho_n,rho_1n,x,z_n,h)
        if np.abs(f_eval,dtype=DTYPE) <= 1e-10:
            break
    return x

def newtonMethodZ(x0,iterationNumber,z_n, z_1n, rho_2n, h):
    x=x0
    f_eval = F2(z_n, z_1n, x, rho_2n, h)
    for _ in range(iterationNumber):
        x=x-f_eval/F2_diff(z_n, z_1n, x, rho_2n, h)
        f_eval = F2(z_n, z_1n, x, rho_2n, h)
        if np.abs(f_eval,dtype=DTYPE) <= 1e-10:
            break
    return x