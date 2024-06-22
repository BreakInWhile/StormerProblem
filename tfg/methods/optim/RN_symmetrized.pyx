"""
Implements numeric method in Cartesian coordinates proposed
by Ramírez-Nicolás et al. (2014). In this library the method
is symmetrized to reduce the error.
"""

import cython
import numpy as np
cimport numpy as np

from tqdm import tqdm

from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger

### External ###
cdef extern from "math.h":
    double sqrt(double x)

### Constants ###
DTYPE=np.float64
NAME="Ramírez-Nicolás Method (Optimized)"
LOGGER=create_logger(NAME)

cdef double NEWTON_THRESHOLD = 1e-13
cdef int NEWTON_ITERS = 100

class RNMethod(StormerSolvingMethod):
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
        bint verbose = True
        ):
        """
        Simulates the trajectory in 3 dimensions using method proposed
        by Ramírez-Nicolás et al. (2014).

        Parameters:
            - initial_values: initial values of the trajectory. [x,y,z,dx,dy,dx]
            - time_step: the discrete time step
            - iterations: number of iterations for the trajectory
            - verbose: whether or not to print out logs. Default to true. 
        """

        cdef Py_ssize_t i,nothing
        cdef int counter
        cdef double sum1,sum2
        cdef double[:] x0_view,v0_view,xi_view,xn_view,vn_view,mmul_view
        cdef double[:,:] F_view,J_view,J_augmented_view

        if verbose: LOGGER.info(f"Simulating for {iterations} iterations...")

        x0 = np.zeros((3,),dtype=DTYPE)
        v0 = np.zeros((3,),dtype=DTYPE)
        xi = np.zeros((3,),dtype=DTYPE)
        xn = np.zeros((3,),dtype=DTYPE)
        vn = np.zeros((3,),dtype=DTYPE)
        f = np.zeros((3,1),dtype=DTYPE)
        j = np.zeros((3,3),dtype=DTYPE)
        j_augmented = np.zeros((3,6),dtype=DTYPE)
        mmul = np.zeros((3,),dtype=DTYPE)

        #--- Memory views ---#
        x0_view = x0
        v0_view = v0
        xi_view = xi
        xn_view = xn
        vn_view = vn
        F_view = f
        J_view = j
        J_augmented_view = j_augmented
        mmul_view = mmul

        x0_view[0] = initial_values[0]
        x0_view[1] = initial_values[1]
        x0_view[2] = initial_values[2]
        v0_view[0] = initial_values[3]
        v0_view[1] = initial_values[4]
        v0_view[2] = initial_values[5]
        
        sols = [np.array([x0[0],x0[1],x0[2],v0[0],v0[1],v0[2]],dtype=DTYPE)]
        for _ in tqdm(range(iterations),disable=not verbose):
            # get seed
            get_seed(xi_view,x0_view,v0_view,time_step)
            
            # Newton-Raphson
            counter = 0
            for nothing in range(NEWTON_ITERS):
                get_F(F_view,x0_view,xi_view,v0_view,time_step)
                get_J(J_view,x0_view,xi_view,time_step)
                c_invert(J_augmented_view,J_view)
                c_matrix_multiply(mmul_view,J_view,F_view)
                for i in range(3):
                    xn_view[i] = xi_view[i]-mmul_view[i]
                
                sum1 = 0
                sum2 = 0
                for i in range(3):
                    sum1 += (xn_view[i]-xi_view[i])**2.0
                    sum2 += xi_view[i]**2.0
                if sqrt(sum1)/sqrt(sum2) <= NEWTON_THRESHOLD:
                    break

                for i in range(3):
                    xi_view[i] = xn_view[i]

                counter+=1
            
            if counter == NEWTON_ITERS: 
                LOGGER.info("Newton-Raphson could not converge")
                return None

            vn_view[0] = 2.0*(xn_view[0]-x0_view[0])/time_step - v0_view[0]
            vn_view[1] = 2.0*(xn_view[1]-x0_view[1])/time_step - v0_view[1]
            vn_view[2] = 2.0*(xn_view[2]-x0_view[2])/time_step - v0_view[2]

            sols.append(np.array([xn_view[0],xn_view[1],xn_view[2],vn_view[0],vn_view[1],vn_view[2]],dtype=DTYPE))
            
            for i in range(3):
                x0_view[i] = xn_view[i]

            for i in range(3):
                v0_view[i] = vn_view[i]

        return np.array(sols,dtype=DTYPE)

### Functions ###

# F
@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void get_F(double[:,:] res_view,double[:] x_vec,double[:] xn_vec,double[:] v_vec,double h):

    cdef double x,y,z,rad
    cdef double xn,yn,zn,radn
    cdef double vx,vy,vz

    x = x_vec[0]
    y = x_vec[1]
    z = x_vec[2]
    rad = sqrt(x**2.0+y**2.0+z**2.0)

    xn = xn_vec[0]
    yn = xn_vec[1]
    zn = xn_vec[2]
    radn = sqrt(xn**2.0+yn**2.0+zn**2.0)

    vx = v_vec[0]
    vy = v_vec[1]
    vz = v_vec[2]

    res_view[0][0] = xn - h*vx - 1.0/4.0*h*(-(zn - z)*(3.0*zn*yn/radn**5.0 + 3.0*y*z/rad**5.0) + (yn - y)*((-x**2.0 - y**2.0 + 2.0*z**2.0)/rad**5.0 + (2.0*zn**2.0 - yn**2.0 - xn**2.0)/radn**5.0)) - x
    res_view[1][0] = yn - h*vy - 1.0/4.0*h*((zn - z)*(3.0*zn*xn/radn**5.0 + 3.0*x*z/rad**5.0) - (xn - x)*((-x**2.0 - y**2.0 + 2.0*z**2.0)/rad**5.0 + (2.0*zn**2.0 - yn**2.0 - xn**2.0)/radn**5.0)) - y
    res_view[2][0] = zn - h*vz - 1.0/4.0*h*(-(yn - y)*(3.0*zn*xn/radn**5.0 + 3.0*x*z/rad**5.0) + (xn - x)*(3.0*zn*yn/radn**5.0 + 3.0*y*z/rad**5.0)) - z

# Jacobian of F
@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void get_J(double[:,:] res_view,double[:] x_vec,double[:] xn_vec,double h):
    """
    Get the Jacobian matrix given x_{n} and x_{n+1}
    """

    cdef double x,y,z,rad
    cdef double xn,yn,zn,radn,termn

    x = x_vec[0]
    y = x_vec[1]
    z = x_vec[2]
    rad = sqrt(x**2.0+y**2.0+z**2.0)

    xn = xn_vec[0]
    yn = xn_vec[1]
    zn = xn_vec[2]
    radn = sqrt(xn**2.0+yn**2.0+zn**2.0)
    termn = -4.0*zn**2.0 + yn**2.0 + xn**2.0

    res_view[0][0] = -1.0/4.0*h*(-15.0*zn*yn*xn*(zn - z)/radn**7.0 + 3.0*xn*(-yn + y)*termn/radn**7.0) + 1.0
    res_view[0][1] = -1.0/4.0*h*(3.0*zn*(zn - z)*(zn**2.0 - 4.0*yn**2.0 + xn**2.0)/radn**7.0 + 3.0*yn*(-yn + y)*termn/radn**7.0 + (-x**2.0 - y**2.0 + 2.0*z**2.0)/rad**5.0 + (2.0*zn**2.0 - yn**2.0 - xn**2.0)/radn**5.0)
    res_view[0][2] = -1.0/4.0*h*(-3.0*zn*yn/radn**5.0 + 3.0*zn*(-yn + y)*(-2.0*zn**2.0 + 3.0*yn**2.0 + 3.0*xn**2.0)/radn**7.0 + 3.0*yn*(zn - z)*termn/radn**7.0 - 3.0*y*z/rad**5.0)

    res_view[1][0] = -1.0/4.0*h*(3.0*zn*(-zn + z)*(zn**2 + yn**2.0 - 4.0*xn**2.0)/radn**7.0 + 3.0*xn*(xn - x)*termn/radn**7.0 - (-x**2.0 - y**2.0 + 2.0*z**2.0)/rad**5.0 - (2.0*zn**2.0 - yn**2.0 - xn**2.0)/radn**5.0)
    res_view[1][1] = -1.0/4.0*h*(-15.0*zn*yn*xn*(-zn + z)/radn**7.0 + 3.0*yn*(xn - x)*termn/radn**7.0) + 1.0
    res_view[1][2] = -1.0/4.0*h*(3.0*zn*xn/radn**5.0 + 3.0*zn*(xn - x)*(-2.0*zn**2.0 + 3.0*yn**2.0 + 3.0*xn**2.0)/radn**7.0 + 3.0*xn*(-zn + z)*termn/radn**7.0 + 3.0*x*z/rad**5.0)

    res_view[2][0] = -1.0/4.0*h*(-15.0*zn*yn*xn*(-xn + x)/radn**7.0 + 3.0*zn*yn/radn**5.0 + 3.0*zn*(yn - y)*(zn**2.0 + yn**2.0 - 4.0*xn**2.0)/radn**7.0 + 3.0*y*z/rad**5.0)
    res_view[2][1] = -1.0/4.0*h*(-15.0*zn*yn*xn*(yn - y)/radn**7.0 - 3.0*zn*xn/radn**5.0 + 3.0*zn*(-xn + x)*(zn**2.0 - 4.0*yn**2.0 + xn**2.0)/radn**7.0 - 3.0*x*z/rad**5.0)
    res_view[2][2] = -1.0/4.0*h*(3.0*yn*(-xn + x)*termn/radn**7.0 + 3.0*xn*(yn - y)*termn/radn**7.0) + 1.0

# Seed
@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void get_seed(double[:] seed_view, double[:] x_vec,double[:] v_vec,double h):
    cdef double x,y,z,rad
    cdef double[:,:] maux_view,maux_augmented_view,v_vec_m

    v_vec_m = np.zeros((3,1),dtype=DTYPE)
    v_vec_m[0][0] = v_vec[0]
    v_vec_m[1][0] = v_vec[1]
    v_vec_m[2][0] = v_vec[2]

    x = x_vec[0]
    y = x_vec[1]
    z = x_vec[2]
    rad = sqrt(x**2.0+y**2.0+z**2.0)

    m_aux = np.zeros((3,3),dtype=DTYPE)
    maux_augmented_view = np.zeros((3,6),dtype=DTYPE)
    maux_view = m_aux

    maux_view[0][0] = 1.0
    maux_view[0][1] = -1.0/2.0*h*(-x**2.0 - y**2.0 + 2.0*z**2.0)/rad**5.0
    maux_view[0][2] = (3.0/2.0)*h*y*z/rad**5.0

    maux_view[1][0] = (1.0/2.0)*h*(-x**2.0 - y**2.0 + 2.0*z**2.0)/rad**5.0
    maux_view[1][1] = 1.0
    maux_view[1][2] = -3.0/2.0*h*x*z/rad**5.0

    maux_view[2][0] = -3.0/2.0*h*y*z/rad**5.0
    maux_view[2][1] = (3.0/2.0)*h*x*z/rad**5.0
    maux_view[2][2] = 1.0

    #maux_inv = np.linalg.inv(m_aux)

    c_invert(maux_augmented_view,maux_view)
    c_matrix_multiply(seed_view,maux_view,v_vec_m)

    #seed = x_vec + h*(m_aux@v_vec)

    #seed_view[0] = seed[0]
    #seed_view[1] = seed[1]
    #seed_view[2] = seed[2]

    seed_view[0] = x_vec[0] + h*seed_view[0]
    seed_view[1] = x_vec[1] + h*seed_view[1]
    seed_view[2] = x_vec[2] + h*seed_view[2]

#--- Auxiliar ---#

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void gauss_jordan(double[:,:] augmented_matrix,Py_ssize_t n) noexcept:
    cdef double ratio,divisor
    cdef Py_ssize_t i,j,k
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
                for k in range(n*2):
                    augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]

    for i in range(n):
        divisor = augmented_matrix[i][i]
        for j in range(n*2):
            augmented_matrix[i][j] /= divisor

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void c_invert(double[:,:] augmented_matrix,double[:,:] matrix) noexcept:
    cdef Py_ssize_t i,j,n
    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            augmented_matrix[i,j] = matrix[i,j]
            augmented_matrix[i][j+n] = 1.0 if i == j else  0.0

    gauss_jordan(augmented_matrix,n)

    for i in range(n):
        for j in range(n,2*n):
            matrix[i,j-n] = augmented_matrix[i,j]

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void c_matrix_multiply(double[:] result,double[:,:] mat1,double[:,:] mat2):
    cdef Py_ssize_t i,j,k,r1,c1,c2
    r1 = mat1.shape[0]
    c1 = mat1.shape[1]
    c2 = mat2.shape[1]
    for i in range(r1):
        for j in range(c2):
            result[i] = 0.0;
            for k in range(c1):
                result[i] += mat1[i][k] * mat2[k][j];