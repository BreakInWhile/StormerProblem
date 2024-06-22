import numpy as np
import sys
from tqdm import tqdm
from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger
from tfg.methods import newton
from tfg.methods.optim.SV import StormerVerletMethod

NAME = "Gauss Method (General)"

logger = create_logger(NAME)
DTYPE = np.float64
NEWTON_THRESHOLD = 1e-13
NEWTON_ITERS = 100

A = np.array([
    [1.0/4.0,1/4-np.sqrt(3.0,dtype=DTYPE)/6.0],
    [1.0/4.0+np.sqrt(3.0,dtype=DTYPE)/6.0,1.0/4.0]
])
b = np.array([1.0/2.0,1.0/2.0])

# (*) Uncomment to try a different seed for Newton-Raphson method
# METHOD_FOR_SEED = StormerVerletMethod()

class GaussMethod(StormerSolvingMethod):
    def __init__(self):
        super().__init__(NAME)

    def simulate(self, initial_values: np.ndarray, time_step: int, iterations: int,verbose: bool = True) -> np.ndarray:
        if verbose: logger.info(f"Simulating for {iterations} iterations...")
        x = initial_values[0]
        y = initial_values[1]
        z = initial_values[2]
        dx = initial_values[3]
        dy = initial_values[4]
        dz = initial_values[5]
        solutions = [initial_values]
        x0 = np.empty((12,),dtype=DTYPE)
        for _ in tqdm(range(iterations)):

            rad = np.sqrt(x**2.0+y**2.0+z**2.0)
            term = (3.0*z/rad**5.0)
            f1_eval = f1(y,z,dy,dz,rad,term)
            f2_eval = f2(x,z,dx,dz,rad,term)
            f3_eval = f3(x,y,dx,dy,term)
            x0[0] = dx
            x0[1] = dy
            x0[2] = dz
            x0[3] = f1_eval
            x0[4] = f2_eval
            x0[5] = f3_eval
            x0[6] = dx
            x0[7] = dy
            x0[8] = dz
            x0[9] = f1_eval
            x0[10] = f2_eval
            x0[11] = f3_eval

            # (*) Uncomment to try a different seed for Newton-Raphson method
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
                f = F(x0,x,y,z,dx,dy,dz,time_step)
                j = J(x0,x,y,z,dx,dy,dz,time_step)
                x_res=x0-(np.linalg.inv(j)@f)[:,0]
                if np.sqrt(np.sum((x_res-x0)**2))/np.sqrt(np.sum(x0**2)) <= NEWTON_THRESHOLD:
                    break
                x0 = x_res
                counter+=1
            if counter == NEWTON_THRESHOLD: logger.debug(f"{counter}: No ha convergido")
            
            slopes_sum = np.zeros((6,))
            for i in range(2):
                slopes_sum += time_step*b[i]*x_res[6*i:6*(1+i)]
            
            x+=slopes_sum[0]
            y+=slopes_sum[1]
            z+=slopes_sum[2]
            dx+=slopes_sum[3]
            dy+=slopes_sum[4]
            dz+=slopes_sum[5]
            solutions.append(np.array([x,y,z,dx,dy,dz]))
        return np.array(solutions)

def f1(y: float,z: float,dy: float,dz: float,rad: float,term: float) -> float:
    return term*(dy*z - dz*y)-dy/rad**3.0

def f2(x: float,z: float,dx: float,dz: float,rad: float,term: float) -> float:
    return -term*(dx*z - dz*x)+dx/rad**3.0

def f3(x: float,y: float,dx: float,dy: float,term: float) -> float:
    return term*(dx*y - dy*x)

def F(k: np.ndarray,x: float,y: float,z: float,dx: float,dy: float,dz: float,h: float):
    y1_1 = x+h*(A[0,0]*k[0]+A[0,1]*k[6])
    y2_1 = y+h*(A[0,0]*k[1]+A[0,1]*k[7])
    y3_1 = z+h*(A[0,0]*k[2]+A[0,1]*k[8])
    y4_1 = dx+h*(A[0,0]*k[3]+A[0,1]*k[9])
    y5_1 = dy+h*(A[0,0]*k[4]+A[0,1]*k[10])
    y6_1 = dz+h*(A[0,0]*k[5]+A[0,1]*k[11])

    y1_2 = x+h*(A[1,0]*k[0]+A[1,1]*k[6])
    y2_2 = y+h*(A[1,0]*k[1]+A[1,1]*k[7])
    y3_2 = z+h*(A[1,0]*k[2]+A[1,1]*k[8])
    y4_2 = dx+h*(A[1,0]*k[3]+A[1,1]*k[9])
    y5_2 = dy+h*(A[1,0]*k[4]+A[1,1]*k[10])
    y6_2 = dz+h*(A[1,0]*k[5]+A[1,1]*k[11])
    rad_1 = np.sqrt(y1_1**2.0+y2_1**2.0+y3_1**2.0)
    rad_2 = np.sqrt(y1_2**2.0+y2_2**2.0+y3_2**2.0)
    term_1 = (3.0*y3_1/rad_1**5.0)
    term_2 = (3.0*y3_2/rad_2**5.0)

    res = np.array([
        [k[0]-y4_1],
        [k[1]-y5_1],
        [k[2]-y6_1],
        [k[3]-f1(y2_1,y3_1,y5_1,y6_1,rad_1,term_1)],
        [k[4]-f2(y1_1,y3_1,y4_1,y6_1,rad_1,term_1)],
        [k[5]-f3(y1_1,y2_1,y4_1,y5_1,term_1)],
        [k[6]-y4_2],
        [k[7]-y5_2],
        [k[8]-y6_2],
        [k[9]-f1(y2_2,y3_2,y5_2,y6_2,rad_2,term_2)],
        [k[10]-f2(y1_2,y3_2,y4_2,y6_2,rad_2,term_2)],
        [k[11]-f3(y1_2,y2_2,y4_2,y5_2,term_2)],
    ],dtype=DTYPE)
    return res

def f11(x, y, z, dy, dz):
    return 3.0*dy*x*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*x*z*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

def f12(x, y, z, dy, dz):
    return 3.0*dy*y*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 3.0*dz*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*y*z*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

def f13(x, y, z, dy, dz):
    return 6.0*dy*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*z**2*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-3.5) + 3.0*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

def f15(x, y, z):
    return 3.0*z**2*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - (x**2.0 + y**2.0 + z**2.0)**(-1.5)

def f16(x, y, z):
    return -3.0*y*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

def f21(x, y, z, dx, dz):
    return -3.0*dx*x*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 3.0*dz*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 15.0*x*z*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

def f22(x, y, z, dx, dz):
    return -3.0*dx*y*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 15.0*y*z*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

def f23(x, y, z, dx, dz):
    return -6.0*dx*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 15.0*z**2*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5) - 3.0*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

def f24(x, y, z):
    return -3.0*z**2*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + (x**2.0 + y**2.0 + z**2.0)**(-1.5)

def f26(x, y, z):
    return 3.0*x*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

def f31(x, y, z, dx, dy):
    return -3.0*dy*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*x*z*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

def f32(x, y, z, dx, dy):
    return 3.0*dx*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*y*z*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

def f33(x, y, z, dx, dy):
    return -15.0*z**2*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5) + 3.0*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

def f34(x, y, z):
    return 3.0*y*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

def f35(x, y, z):
    return -3.0*x*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

def J(k: np.ndarray,x: float,y: float,z: float,dx: float,dy: float,dz: float,h: float):
    y1_1 = x+h*(A[0,0]*k[0]+A[0,1]*k[6])
    y2_1 = y+h*(A[0,0]*k[1]+A[0,1]*k[7])
    y3_1 = z+h*(A[0,0]*k[2]+A[0,1]*k[8])
    y4_1 = dx+h*(A[0,0]*k[3]+A[0,1]*k[9])
    y5_1 = dy+h*(A[0,0]*k[4]+A[0,1]*k[10])
    y6_1 = dz+h*(A[0,0]*k[5]+A[0,1]*k[11])

    y1_2 = x+h*(A[1,0]*k[0]+A[1,1]*k[6])
    y2_2 = y+h*(A[1,0]*k[1]+A[1,1]*k[7])
    y3_2 = z+h*(A[1,0]*k[2]+A[1,1]*k[8])
    y4_2 = dx+h*(A[1,0]*k[3]+A[1,1]*k[9])
    y5_2 = dy+h*(A[1,0]*k[4]+A[1,1]*k[10])
    y6_2 = dz+h*(A[1,0]*k[5]+A[1,1]*k[11])

    ha11 = -h*A[0,0]
    ha12 = -h*A[0,1]
    ha21 = -h*A[1,0]
    ha22 = -h*A[1,1]

    f11_1 = f11(y1_1,y2_1,y3_1,y5_1,y6_1)
    f12_1 = f12(y1_1,y2_1,y3_1,y5_1,y6_1)
    f13_1 = f13(y1_1,y2_1,y3_1,y5_1,y6_1)
    f15_1 = f15(y1_1,y2_1,y3_1)
    f16_1 = f16(y1_1,y2_1,y3_1)
    f21_1 = f21(y1_1,y2_1,y3_1,y4_1,y6_1)
    f22_1 = f22(y1_1,y2_1,y3_1,y4_1,y6_1)
    f23_1 = f23(y1_1,y2_1,y3_1,y4_1,y6_1)
    f24_1 = f24(y1_1,y2_1,y3_1)
    f26_1 = f26(y1_1,y2_1,y3_1)
    f31_1 = f31(y1_1,y2_1,y3_1,y4_1,y5_1)
    f32_1 = f32(y1_1,y2_1,y3_1,y4_1,y5_1)
    f33_1 = f33(y1_1,y2_1,y3_1,y4_1,y5_1)
    f34_1 = f34(y1_1,y2_1,y3_1)
    f35_1 = f35(y1_1,y2_1,y3_1)

    f11_2 = f11(y1_2,y2_2,y3_2,y5_2,y6_2)
    f12_2 = f12(y1_2,y2_2,y3_2,y5_2,y6_2)
    f13_2 = f13(y1_2,y2_2,y3_2,y5_2,y6_2)
    f15_2 = f15(y1_2,y2_2,y3_2)
    f16_2 = f16(y1_2,y2_2,y3_2)
    f21_2 = f21(y1_2,y2_2,y3_2,y4_2,y6_2)
    f22_2 = f22(y1_2,y2_2,y3_2,y4_2,y6_2)
    f23_2 = f23(y1_2,y2_2,y3_2,y4_2,y6_2)
    f24_2 = f24(y1_2,y2_2,y3_2)
    f26_2 = f26(y1_2,y2_2,y3_2)
    f31_2 = f31(y1_2,y2_2,y3_2,y4_2,y5_2)
    f32_2 = f32(y1_2,y2_2,y3_2,y4_2,y5_2)
    f33_2 = f33(y1_2,y2_2,y3_2,y4_2,y5_2)
    f34_2 = f34(y1_2,y2_2,y3_2)
    f35_2 = f35(y1_2,y2_2,y3_2)

    res = np.array([
        [1,0,0,ha11,0,0,0,0,0,ha12,0,0],
        [0,1,0,0,ha11,0,0,0,0,0,ha12,0],
        [0,0,1,0,0,ha11,0,0,0,0,0,ha12],
        [f11_1*ha11,f12_1*ha11,f13_1*ha11,1,f15_1*ha11,f16_1*ha11,f11_1*ha12,f12_1*ha12,f13_1*ha12,0,f15_1*ha12,f16_1*ha12],
        [f21_1*ha11,f22_1*ha21,f23_1*ha11,f24_1*ha11,1,f26_1*ha11,f21_1*ha12,f22_1*ha12,f23_1*ha12,f24_1*ha12,0,f26_1*ha12],
        [f31_1*ha11,f32_1*ha21,f33_1*ha11,f34_1*ha11,f35_1*ha11,1,f31_1*ha12,f32_1*ha12,f33_1*ha12,f34_1*ha12,f35_1*ha12,0],
        [0,0,0,ha21,0,0,1,0,0,ha22,0,0],
        [0,0,0,0,ha21,0,0,1,0,0,ha22,0],
        [0,0,0,0,0,ha21,0,0,1,0,0,ha22],
        [f11_2*ha21,f12_2*ha21,f13_2*ha21,0,f15_2*ha21,f16_2*ha21,f11_2*ha22,f12_2*ha22,f13_2*ha22,1,f15_2*ha22,f16_2*ha22],
        [f21_2*ha21,f22_2*ha21,f23_2*ha21,f24_2*ha21,0,f26_2*ha21,f21_2*ha22,f22_2*ha22,f23_2*ha22,f24_2*ha22,1,f26_2*ha22],
        [f31_2*ha21,f32_2*ha21,f33_2*ha21,f34_2*ha21,f35_2*ha21,0,f31_2*ha22,f32_2*ha22,f33_2*ha22,f34_2*ha22,f35_2*ha22,1]
    ],dtype=DTYPE)

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