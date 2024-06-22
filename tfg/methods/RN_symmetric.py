"""
Implements numeric methods in Cartesian coordinates proposed
by Ramírez-Nicolás et al. (2014). In this library the method
is symmetrized to reduce the error.
"""

import numpy as np

from tqdm import tqdm

from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger

### Constants ###
DTYPE=np.float64
NAME="Ramírez-Nicolás Method"
LOGGER=create_logger(NAME)
NEWTON_THRESHOLD = 1e-13
NEWTON_ITERS = 100

class RNMethod(StormerSolvingMethod):
    def __init__(self):
        super().__init__(NAME)

    def simulate(self, initial_values: np.ndarray, time_step: int, iterations: int,verbose: bool = True) -> np.ndarray:
        assert initial_values.shape == (6,), f"Initial values must have shape (6,) but {initial_values.shape} was given"
        LOGGER.info(f"Simulating for {iterations} iterations...")
        x0 = initial_values[:3]
        v0 = initial_values[3:]
        sols = [np.array([x0[0],x0[1],x0[2],v0[0],v0[1],v0[2]],dtype=DTYPE)]
        for _ in tqdm(range(iterations),disable=not verbose):
            # get seed
            xi = get_seed(x0,v0,time_step)
            
            # Newton-Raphson
            counter = 0
            for _ in range(NEWTON_ITERS):
                f = get_F(x0,xi,v0,time_step)
                j = get_J(x0,xi,time_step)
                xn=xi-(np.linalg.inv(j)@f)[:,0]
                if np.sqrt(np.sum((xn-xi)**2))/np.sqrt(np.sum(xi**2)) <= NEWTON_THRESHOLD:
                    break
                xi = xn
                counter+=1
            
            if counter == NEWTON_ITERS: 
                print("No ha convergido")
                return None

            vn = 2*(xn-x0)/time_step - v0
            sols.append(np.array([xn[0],xn[1],xn[2],vn[0],vn[1],vn[2]],dtype=DTYPE))

            x0 = xn
            v0 = vn
        return np.array(sols,dtype=DTYPE)

### Functions ###

# F

def f_F11(x, y, z, xn, yn, zn, vx, vy, vz, h):
    return xn - h*vx - 1/4*h*(-(zn - z)*(3*zn*yn/(zn**2 + yn**2 + xn**2)**(5/2) + 3*y*z/(x**2 + y**2 + z**2)**(5/2)) + (yn - y)*((-x**2 - y**2 + 2*z**2)/(x**2 + y**2 + z**2)**(5/2) + (2*zn**2 - yn**2 - xn**2)/(zn**2 + yn**2 + xn**2)**(5/2))) - x

def f_F12(x, y, z, xn, yn, zn, vx, vy, vz, h):
    return yn - h*vy - 1/4*h*((zn - z)*(3*zn*xn/(zn**2 + yn**2 + xn**2)**(5/2) + 3*x*z/(x**2 + y**2 + z**2)**(5/2)) - (xn - x)*((-x**2 - y**2 + 2*z**2)/(x**2 + y**2 + z**2)**(5/2) + (2*zn**2 - yn**2 - xn**2)/(zn**2 + yn**2 + xn**2)**(5/2))) - y

def f_F13(x, y, z, xn, yn, zn, vx, vy, vz, h):
    return zn - h*vz - 1/4*h*(-(yn - y)*(3*zn*xn/(zn**2 + yn**2 + xn**2)**(5/2) + 3*x*z/(x**2 + y**2 + z**2)**(5/2)) + (xn - x)*(3*zn*yn/(zn**2 + yn**2 + xn**2)**(5/2) + 3*y*z/(x**2 + y**2 + z**2)**(5/2))) - z

def get_F(x_vec, xn_vec, v_vec, h):
    x,y,z = x_vec
    xn,yn,zn = xn_vec
    vx,vy,vz = v_vec
    f1 = f_F11(x, y, z, xn, yn, zn, vx, vy, vz, h)
    f2 = f_F12(x, y, z, xn, yn, zn, vx, vy, vz, h)
    f3 = f_F13(x, y, z, xn, yn, zn, vx, vy, vz, h)

    return np.array([
        [f1],
        [f2],
        [f3]
    ],dtype=DTYPE)

# Jacobian of F

def f_J11(x, y, z, xn, yn, zn, h):
    return -1/4*h*(-15*zn*yn*xn*(zn - z)/(zn**2 + yn**2 + xn**2)**(7/2) + 3.0*xn*(-yn + y)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2)) + 1

def f_J12(x, y, z, xn, yn, zn, h):
    return -1/4*h*(3.0*zn*(zn - z)*(zn**2 - 4*yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + 3.0*yn*(-yn + y)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + (-x**2 - y**2 + 2*z**2)/(x**2 + y**2 + z**2)**(5/2) + (2*zn**2 - yn**2 - xn**2)/(zn**2 + yn**2 + xn**2)**(5/2))

def f_J13(x, y, z, xn, yn, zn, h):
    return -1/4*h*(-3*zn*yn/(zn**2 + yn**2 + xn**2)**(5/2) + 3.0*zn*(-yn + y)*(-2*zn**2 + 3*yn**2 + 3*xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + 3.0*yn*(zn - z)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) - 3*y*z/(x**2 + y**2 + z**2)**(5/2))

def f_J21(x, y, z, xn, yn, zn, h):
    return -1/4*h*(3.0*zn*(-zn + z)*(zn**2 + yn**2 - 4*xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + 3.0*xn*(xn - x)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) - (-x**2 - y**2 + 2*z**2)/(x**2 + y**2 + z**2)**(5/2) - (2*zn**2 - yn**2 - xn**2)/(zn**2 + yn**2 + xn**2)**(5/2))

def f_J22(x, y, z, xn, yn, zn, h):
    return -1/4*h*(-15*zn*yn*xn*(-zn + z)/(zn**2 + yn**2 + xn**2)**(7/2) + 3.0*yn*(xn - x)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2)) + 1

def f_J23(x, y, z, xn, yn, zn, h):
    return -1/4*h*(3*zn*xn/(zn**2 + yn**2 + xn**2)**(5/2) + 3.0*zn*(xn - x)*(-2*zn**2 + 3*yn**2 + 3*xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + 3.0*xn*(-zn + z)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + 3*x*z/(x**2 + y**2 + z**2)**(5/2))

def f_J31(x, y, z, xn, yn, zn, h):
    return -1/4*h*(-15*zn*yn*xn*(-xn + x)/(zn**2 + yn**2 + xn**2)**(7/2) + 3*zn*yn/(zn**2 + yn**2 + xn**2)**(5/2) + 3.0*zn*(yn - y)*(zn**2 + yn**2 - 4*xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + 3*y*z/(x**2 + y**2 + z**2)**(5/2))

def f_J32(x, y, z, xn, yn, zn, h):
    return -1/4*h*(-15*zn*yn*xn*(yn - y)/(zn**2 + yn**2 + xn**2)**(7/2) - 3*zn*xn/(zn**2 + yn**2 + xn**2)**(5/2) + 3.0*zn*(-xn + x)*(zn**2 - 4*yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) - 3*x*z/(x**2 + y**2 + z**2)**(5/2))

def f_J33(x, y, z, xn, yn, zn, h):
    return -1/4*h*(3.0*yn*(-xn + x)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2) + 3.0*xn*(yn - y)*(-4*zn**2 + yn**2 + xn**2)/(zn**2 + yn**2 + xn**2)**(7/2)) + 1

def get_J(x_vec, xn_vec, h):
    """
    Get the Jacobian matrix given x_{n} and x_{n+1}
    """

    x,y,z = x_vec
    xn,yn,zn = xn_vec

    j11 = f_J11(x, y, z, xn, yn, zn, h)
    j12 = f_J12(x, y, z, xn, yn, zn, h)
    j13 = f_J13(x, y, z, xn, yn, zn, h)

    j21 = f_J21(x, y, z, xn, yn, zn, h)
    j22 = f_J22(x, y, z, xn, yn, zn, h)
    j23 = f_J23(x, y, z, xn, yn, zn, h)

    j31 = f_J31(x, y, z, xn, yn, zn, h)
    j32 = f_J32(x, y, z, xn, yn, zn, h)
    j33 = f_J33(x, y, z, xn, yn, zn, h)

    return np.array([
        [j11,j12,j13],
        [j21,j22,j23],
        [j31,j32,j33],
    ],dtype=DTYPE)

# Seed

def f_k11(x, y, z, h):
    return 1

def f_k12(x, y, z, h):
    return -1/2*h*(-x**2 - y**2 + 2*z**2)/(x**2 + y**2 + z**2)**(5/2)

def f_k13(x, y, z, h):
    return (3/2)*h*y*z/(x**2 + y**2 + z**2)**(5/2)

def f_k21(x, y, z, h):
    return (1/2)*h*(-x**2 - y**2 + 2*z**2)/(x**2 + y**2 + z**2)**(5/2)

def f_k22(x, y, z, h):
    return 1

def f_k23(x, y, z, h):
    return -3/2*h*x*z/(x**2 + y**2 + z**2)**(5/2)

def f_k31(x, y, z, h):
    return -3/2*h*y*z/(x**2 + y**2 + z**2)**(5/2)

def f_k32(x, y, z, h):
    return (3/2)*h*x*z/(x**2 + y**2 + z**2)**(5/2)

def f_k33(x, y, z, h):
    return 1

def get_maux(x_vec,h):
    x,y,z = x_vec

    k11 = f_k11(x,y,z,h)
    k12 = f_k12(x,y,z,h)
    k13 = f_k13(x,y,z,h)

    k21 = f_k21(x,y,z,h)
    k22 = f_k22(x,y,z,h)
    k23 = f_k23(x,y,z,h)

    k31 = f_k31(x,y,z,h)
    k32 = f_k32(x,y,z,h)
    k33 = f_k33(x,y,z,h)

    return np.linalg.inv(np.array([
        [k11,k12,k13],
        [k21,k22,k23],
        [k31,k32,k33]
    ],dtype=DTYPE))

def get_seed(x_vec,v_vec,h):
    m_aux = get_maux(x_vec,h)
    return x_vec + h*(m_aux@v_vec)