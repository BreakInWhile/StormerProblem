import numpy as np
from tqdm import tqdm
from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger
from tfg.methods import newton
from tfg.methods.optim.SV import StormerVerletMethod

NAME = "Gauss Method"

logger = create_logger(NAME)
DTYPE = np.float64
NEWTON_THRESHOLD = 1e-13
NEWTON_ITERS = 100

A = np.array([
    [1.0/4.0,1/4-np.sqrt(3.0,dtype=DTYPE)/6.0],
    [1.0/4.0+np.sqrt(3.0,dtype=DTYPE)/6.0,1.0/4.0]
])
b = np.array([1.0/2.0,1.0/2.0])

METHOD_FOR_SEED = StormerVerletMethod()

class GaussMethod(StormerSolvingMethod):
    def __init__(self):
        super().__init__(NAME)

    def simulate(self, initial_values: np.ndarray, time_step: int, iterations: int,verbose: bool = True) -> np.ndarray:
        logger.info(f"Simulating for {iterations} iterations...")
        rho = initial_values[0]
        z = initial_values[1]
        p_rho = initial_values[2]
        p_z = initial_values[3]
        solutions = [np.array([rho,z,p_rho,p_z])]
        for _ in tqdm(range(iterations)):

            rad = np.sqrt(rho**2+z**2)
            term = (rho/rad**3 - 1/rho)
            x0 = np.array([p_rho,p_z,f1(rho,rad,term),f2(rho,z,rad,term),p_rho,p_z,f1(rho,rad,term),f2(rho,z,rad,term)])

            # algo = METHOD_FOR_SEED
            #.simulate(np.array([rho,z,p_rho,p_z]),time_step,1,verbose=False)[-1]
            # left_side = 2/time_step * (algo - np.array([rho,z,p_rho,p_z]))
            # f_seed_eval = F_seed(x0,p_rho,p_z,time_step,left_side)
            # j_seed_eval = J_seed(time_step)
            # for _ in range(newton_iters):
            #     x0=x0-(np.linalg.inv(j_seed_eval)@f_seed_eval)[:,0]
            #     f_eval = F_seed(x0,p_rho,p_z,time_step,left_side)
            #     if np.sqrt(np.sum(f_eval**2)) <= threshold:
            #         break
            
            counter = 0
            for _ in range(NEWTON_ITERS):
                f = F(x0,rho,z,p_rho,p_z,time_step)
                j = J(x0,rho,z,time_step)
                x=x0-(np.linalg.inv(j)@f)[:,0]
                if np.sqrt(np.sum((x-x0)**2))/np.sqrt(np.sum(x0**2)) <= NEWTON_THRESHOLD:
                    break
                x0 = x
                counter+=1

            if counter == NEWTON_ITERS: logger.debug(f"{counter}: No ha convergido")
            
            slopes_sum = np.zeros((4,))
            for i in range(2):
                slopes_sum += time_step*b[i]*x[4*i:4*(1+i)]
            
            rho+=slopes_sum[0]
            z+=slopes_sum[1]
            p_rho+=slopes_sum[2]
            p_z+=slopes_sum[3]
            solutions.append(np.array((rho,z,p_rho,p_z)))
        return np.array(solutions)

def f1(rho,rad,term):
    return term*(-1.0/rho**2.0 - 1.0/rad**3.0 + (3.0*rho**2.0)/rad**5.0)

def f2(rho,z,rad,term):
    return term*((3.0*rho*z)/rad**5.0)

def F(k,rho,z,p_rho,p_z,h):
    y1_1 = rho+h*(A[0,0]*k[0]+A[0,1]*k[4])
    y2_1 = z+h*(A[0,0]*k[1]+A[0,1]*k[5])
    y1_2 = rho+h*(A[1,0]*k[0]+A[1,1]*k[4])
    y2_2 = z+h*(A[1,0]*k[1]+A[1,1]*k[5])
    rad_1 = np.sqrt(y1_1**2+y2_1**2)
    rad_2 = np.sqrt(y1_2**2+y2_2**2)
    term_1 = (y1_1/rad_1**3 - 1/y1_1)
    term_2 = (y1_2/rad_2**3 - 1/y1_2)
    res = np.array([
        [k[0]-(p_rho+h*(A[0,0]*k[2]+A[0,1]*k[6]))],
        [k[1]-(p_z+h*(A[0,0]*k[3]+A[0,1]*k[7]))],
        [k[2]-f1(y1_1,rad_1,term_1)],
        [k[3]-f2(y1_1,y2_1,rad_1,term_1)],
        [k[4]-(p_rho+h*(A[1,0]*k[2]+A[1,1]*k[6]))],
        [k[5]-(p_z+h*(A[1,0]*k[3]+A[1,1]*k[7]))],
        [k[6]-f1(y1_2,rad_2,term_2)],
        [k[7]-f2(y1_2,y2_2,rad_2,term_2)]
    ])
    return res

def f20(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return (1/2)*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(4*a_11*h/(a_11*h*k1 + a_12*h*k5 + rho)**3 + 18*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**3/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_11*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_11*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))

def f21(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return (3/2)*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_11*h*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2))*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))

def f24(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return (1/2)*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(4*a_12*h/(a_11*h*k1 + a_12*h*k5 + rho)**3 + 18*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**3/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_12*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_12*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))

def f25(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return (3/2)*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_12*h*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2))*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))

def f30(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return 3*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 3*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-a_11*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_11*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2)

def f31(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return 3*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 9*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**5

def f34(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return 3*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 3*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-a_12*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_12*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2)

def f35(rho, z, k1, k2, k5, k6, a_11, a_12, h):
    return 3*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 9*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**5

def f60(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return (1/2)*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(4*a_21*h/(a_21*h*k1 + a_22*h*k5 + rho)**3 + 18*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**3/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_21*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_21*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))

def f61(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return (3/2)*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_21*h*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2))*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))

def f64(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return (1/2)*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(4*a_22*h/(a_21*h*k1 + a_22*h*k5 + rho)**3 + 18*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**3/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_22*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_22*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))

def f65(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return (3/2)*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_22*h*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2))*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))

def f70(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return 3*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 3*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-a_21*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_21*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2)

def f71(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return 3*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 9*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**5

def f74(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return 3*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 3*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-a_22*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_22*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2)

def f75(rho, z, k1, k2, k5, k6, a_21, a_22, h):
    return 3*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 9*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**5

def J(k,rho,z,h):
    res = np.array([
        [1,0,-h*A[0,0],0,0,0,-h*A[0,1],0],
        [0,1,0,-h*A[0,0],0,0,0,-h*A[0,1]],
        [f20(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),f21(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),1,0,f24(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),f25(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),0,0],
        [f30(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),f31(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),0,1,f34(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),f35(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h),0,0],
        [0,0,-h*A[1,0],0,1,0,-h*A[1,1],0],
        [0,0,0,-h*A[1,0],0,1,0,-h*A[1,1]],
        [f60(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),f61(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),0,0,f64(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),f65(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),1,0],
        [f70(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),f71(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),0,0,f74(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),f75(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h),0,1]
    ])

    return res

def J_seed(time_step):
    res = np.array([
        [1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],
        [1,0,-time_step*A[0,0],0,0,0,-time_step*A[0,1],0],
        [0,1,0,-time_step*A[0,0],0,0,0,-time_step*A[0,1]],
        [0,0,-time_step*A[1,0],0,1,0,-time_step*A[1,1],0],
        [0,0,0,-time_step*A[1,0],0,1,0,-time_step*A[1,1]],
    ])
    return res

def F_seed(k,p_rho,p_z,time_step,left_side):
    res = np.array([
        [k[0]+k[4]-left_side[0]],
        [k[1]+k[5]-left_side[1]],
        [k[2]+k[6]-left_side[2]],
        [k[3]+k[7]-left_side[3]],
        [k[0]-(p_rho + time_step*(A[0,0]*k[2]+A[0,1]*k[6]))],
        [k[1]-(p_z + time_step*(A[0,0]*k[3]+A[0,1]*k[7]))],
        [k[4]-(p_rho + time_step*(A[1,0]*k[2]+A[1,1]*k[6]))],
        [k[5]-(p_z + time_step*(A[1,0]*k[3]+A[1,1]*k[7]))]
    ])
    return res