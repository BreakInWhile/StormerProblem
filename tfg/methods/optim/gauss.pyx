import logging
import cython
import numpy as np
cimport numpy as np
from tqdm import tqdm
from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger
from tfg.methods.optim.SV import StormerVerletMethod

NAME = "Gauss Method"

logger = create_logger(NAME)
DTYPE = np.float64

# (*) Uncomment if you want to try different seed for Newton-Raphson method
#METHOD_FOR_SEED = StormerVerletMethod()

cdef double NEWTON_THRESHOLD = 1e-13
cdef int NEWTON_ITERS = 100

A = np.array([
    [1.0/4.0,1/4-np.sqrt(3.0,dtype=DTYPE)/6.0],
    [1.0/4.0+np.sqrt(3.0,dtype=DTYPE)/6.0,1.0/4.0]
])
b = np.array([1.0/2.0,1.0/2.0])

cdef extern from "math.h":
    double sqrt(double x)
    double fabs(double x)

class GaussMethod(StormerSolvingMethod):
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
        cdef double rho,z,p_rho,p_z
        cdef double rad,term
        cdef double y1_1,y1_2,y2_1,y2_2
        cdef int counter
        cdef Py_ssize_t i

        f = np.zeros((8,1),dtype=DTYPE)
        j = np.zeros((8,8),dtype=DTYPE)
        j_augmented = np.zeros((8,16),dtype=DTYPE)
        mmul = np.zeros((8,),dtype=DTYPE)
        x = np.zeros((8,),dtype=DTYPE)
        x0 = np.zeros((8,),dtype=DTYPE)
        cdef double[:,:] f_view = f
        cdef double[:,:] j_view = j
        cdef double[:,:] j_augmented_view = j_augmented
        cdef double[:] mmul_view = mmul
        cdef double[:] x_view = x
        cdef double[:] x0_view = x
        cdef double[:,:] A_view = A
        cdef double[:] b_view = b

        if verbose: logger.info(f"Simulating for {iterations} iterations...")
        rho = initial_values[0]
        z = initial_values[1]
        p_rho = initial_values[2]
        p_z = initial_values[3]
        solutions = [np.array([rho,z,p_rho,p_z])]
        for _ in tqdm(range(iterations),disable=not verbose):

            rad = sqrt(rho**2+z**2)
            term = (rho/rad**3 - 1/rho)
            x0 = np.array([p_rho,p_z,f1(rho,rad,term),f2(rho,z,rad,term),p_rho,p_z,f1(rho,rad,term),f2(rho,z,rad,term)])

            # x0_view[0] = p_rho
            # x0_view[1] = p_z
            # x0_view[2] = f1(rho,rad,term)
            # x0_view[3] = f2(rho,z,rad,term)
            # x0_view[4] = p_rho
            # x0_view[5] = p_z
            # x0_view[6] = f1(rho,rad,term)
            # x0_view[7] = f2(rho,z,rad,term)

            # (*) Uncomment if you want to try different seed for Newton-Raphson method
            # algo = METHOD_FOR_SEED.simulate(np.array([rho,z,p_rho,p_z]),time_step,1,verbose=False)[-1]
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
                F(f_view,x0,rho,z,p_rho,p_z,A_view,time_step)
                J(j_view,x0,rho,z,A_view,time_step)
                c_invert(j_augmented_view,j_view)
                c_matrix_multiply(mmul_view,j_view,f_view)
                x=x0-mmul
                if sqrt(np.sum((x-x0)**2))/sqrt(np.sum(x0**2)) <= NEWTON_THRESHOLD:
                    break
                x0 = x
                counter+=1

            if counter == NEWTON_THRESHOLD: logger.debug(f"{counter}: No ha convergido")

            rho+=time_step*(b_view[0]*x[0]+b_view[1]*x[4])
            z+=time_step*(b_view[0]*x[1]+b_view[1]*x[5])
            p_rho+=time_step*(b_view[0]*x[2]+b_view[1]*x[6])
            p_z+=time_step*(b_view[0]*x[3]+b_view[1]*x[7])
            solutions.append(np.array((rho,z,p_rho,p_z)))
        return np.array(solutions)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f1(double rho,double rad,double term) noexcept:
    return term*(-1.0/rho**2.0 - 1.0/rad**3.0 + (3.0*rho**2.0)/rad**5.0)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f2(double rho,double z,double rad,double term) noexcept:
    return term*((3.0*rho*z)/rad**5.0)

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void F(double[:,:] f_view,double[:] k,double rho,double z,double p_rho,double p_z,double[:,:] A,double h) noexcept:
    cdef double y1_y,y2_1,y1_2,y2_2,rad_1,rad_2,term_1,term_2

    y1_1 = rho+h*(A[0,0]*k[0]+A[0,1]*k[4])
    y2_1 = z+h*(A[0,0]*k[1]+A[0,1]*k[5])
    y1_2 = rho+h*(A[1,0]*k[0]+A[1,1]*k[4])
    y2_2 = z+h*(A[1,0]*k[1]+A[1,1]*k[5])
    rad_1 = sqrt(y1_1**2+y2_1**2)
    rad_2 = sqrt(y1_2**2+y2_2**2)
    term_1 = (y1_1/rad_1**3 - 1/y1_1)
    term_2 = (y1_2/rad_2**3 - 1/y1_2)
    f_view[0,0] = k[0]-(p_rho+h*(A[0,0]*k[2]+A[0,1]*k[6]))
    f_view[1,0] = k[1]-(p_z+h*(A[0,0]*k[3]+A[0,1]*k[7]))
    f_view[2,0] = k[2]-f1(y1_1,rad_1,term_1)
    f_view[3,0] = k[3]-f2(y1_1,y2_1,rad_1,term_1)
    f_view[4,0] = k[4]-(p_rho+h*(A[1,0]*k[2]+A[1,1]*k[6]))
    f_view[5,0] = k[5]-(p_z+h*(A[1,0]*k[3]+A[1,1]*k[7]))
    f_view[6,0] = k[6]-f1(y1_2,rad_2,term_2)
    f_view[7,0] = k[7]-f2(y1_2,y2_2,rad_2,term_2)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f20(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return (1/2)*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(4*a_11*h/(a_11*h*k1 + a_12*h*k5 + rho)**3 + 18*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**3/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_11*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_11*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f21(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return (3/2)*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_11*h*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2))*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f24(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return (1/2)*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(4*a_12*h/(a_11*h*k1 + a_12*h*k5 + rho)**3 + 18*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**3/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_12*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_12*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f25(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return (3/2)*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-2/(a_11*h*k1 + a_12*h*k5 + rho)**2 - 2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 6*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_12*h*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 30*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2))*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f30(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return 3*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 3*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-a_11*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_11*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f31(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return 3*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_11*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 9*a_11*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**5

@cython.cdivision(True)
@cython.cpow(True)
cdef double f34(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return 3*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 3*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)*(-a_12*h/(a_11*h*k1 + a_12*h*k5 + rho)**2 - a_12*h/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2) + 3*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2))/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f35(double rho,double z,double k1,double k2,double k5,double k6,double a_11,double a_12,double h) noexcept:
    return 3*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(5/2) - 15*a_12*h*((a_11*h*k1 + a_12*h*k5 + rho)**(-1.0) - (a_11*h*k1 + a_12*h*k5 + rho)/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(3/2))*(a_11*h*k1 + a_12*h*k5 + rho)*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**(7/2) + 9*a_12*h*(a_11*h*k1 + a_12*h*k5 + rho)**2*(a_11*h*k2 + a_12*h*k6 + z)**2/((a_11*h*k1 + a_12*h*k5 + rho)**2 + (a_11*h*k2 + a_12*h*k6 + z)**2)**5

@cython.cdivision(True)
@cython.cpow(True)
cdef double f60(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return (1/2)*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(4*a_21*h/(a_21*h*k1 + a_22*h*k5 + rho)**3 + 18*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**3/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_21*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_21*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f61(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return (3/2)*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_21*h*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2))*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f64(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return (1/2)*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(4*a_22*h/(a_21*h*k1 + a_22*h*k5 + rho)**3 + 18*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**3/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2)) + (1/2)*(-a_22*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_22*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f65(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return (3/2)*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-2/(a_21*h*k1 + a_22*h*k5 + rho)**2 - 2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 6*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) + (1/2)*(6*a_22*h*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 30*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2))*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f70(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return 3*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 3*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-a_21*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_21*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f71(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return 3*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_21*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 9*a_21*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**5

@cython.cdivision(True)
@cython.cpow(True)
cdef double f74(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return 3*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 3*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)*(-a_22*h/(a_21*h*k1 + a_22*h*k5 + rho)**2 - a_22*h/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2) + 3*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2))/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f75(double rho,double z,double k1,double k2,double k5,double k6,double a_21,double a_22,double h) noexcept:
    return 3*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(5/2) - 15*a_22*h*((a_21*h*k1 + a_22*h*k5 + rho)**(-1.0) - (a_21*h*k1 + a_22*h*k5 + rho)/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(3/2))*(a_21*h*k1 + a_22*h*k5 + rho)*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**(7/2) + 9*a_22*h*(a_21*h*k1 + a_22*h*k5 + rho)**2*(a_21*h*k2 + a_22*h*k6 + z)**2/((a_21*h*k1 + a_22*h*k5 + rho)**2 + (a_21*h*k2 + a_22*h*k6 + z)**2)**5

@cython.cdivision(True)
@cython.cpow(True)
cdef double f_11(double rho,double z) noexcept:
    return -1.0/2.0*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))*(-30.0*rho**3.0/(rho**2.0 + z**2.0)**(7.0/2.0) + 18.0*rho/(rho**2.0 + z**2.0)**(5.0/2.0) + 4.0/rho**3.0) - 1.0/2.0*(3.0*rho**2.0/(rho**2.0 + z**2.0)**(5.0/2.0) - 1.0/(rho**2.0 + z**2.0)**(3.0/2.0) - 1.0/rho**2.0)*(6.0*rho**2.0/(rho**2.0 + z**2.0)**(5.0/2.0) - 2.0/(rho**2.0 + z**2.0)**(3.0/2.0) - 2.0/rho**2.0)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f_12(double rho,double z) noexcept:
    return -3.0/2.0*rho*z*(6.0*rho**2.0/(rho**2.0 + z**2.0)**(5.0/2.0) - 2.0/(rho**2.0 + z**2.0)**(3.0/2.0) - 2.0/rho**2.0)/(rho**2.0 + z**2.0)**(5.0/2.0) - 1.0/2.0*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))*(-30.0*rho**2.0*z/(rho**2.0 + z**2.0)**(7.0/2.0) + 6.0*z/(rho**2.0 + z**2.0)**(5.0/2.0))

@cython.cdivision(True)
@cython.cpow(True)
cdef double f_21(double rho,double z) noexcept:
    return 15.0*rho**2.0*z*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))/(rho**2.0 + z**2.0)**(7.0/2.0) - 3.0*rho*z*(3.0*rho**2.0/(rho**2.0 + z**2.0)**(5.0/2.0) - 1.0/(rho**2.0 + z**2.0)**(3.0/2.0) - 1.0/rho**2.0)/(rho**2.0 + z**2.0)**(5.0/2.0) - 3.0*z*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))/(rho**2.0 + z**2.0)**(5.0/2.0)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f_22(double rho,double z) noexcept:
    return -9.0*rho**2.0*z**2.0/(rho**2.0 + z**2.0)**5.0 + 15.0*rho*z**2.0*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))/(rho**2.0 + z**2.0)**(7.0/2.0) - 3.0*rho*(-rho/(rho**2.0 + z**2.0)**(3.0/2.0) + rho**(-1.0))/(rho**2.0 + z**2.0)**(5.0/2.0)

@cython.cdivision(True)
@cython.cpow(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void JOld(double[:,:] j_view,double[:] k,double rho,double z,double[:,:] A,double h) noexcept:
    j_view[0,0] = 1.0
    j_view[0,1] = 0.0
    j_view[0,2] = -h*A[0,0]
    j_view[0,3] = 0.0
    j_view[0,4] = 0.0
    j_view[0,5] = 0.0
    j_view[0,6] = -h*A[0,1]
    j_view[0,7] = 0.0

    j_view[1,0] = 0.0
    j_view[1,1] = 1.0
    j_view[1,2] = 0.0
    j_view[1,3] = -h*A[0,0]
    j_view[1,4] = 0.0
    j_view[1,5] = 0.0
    j_view[1,6] = 0.0
    j_view[1,7] = -h*A[0,1]

    j_view[2,0] = f20(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[2,1] = f21(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[2,2] = 1.0
    j_view[2,3] = 0.0
    j_view[2,4] = f24(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[2,5] = f25(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[2,6] = 0.0
    j_view[2,7] = 0.0

    j_view[3,0] = f30(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[3,1] = f31(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[3,2] = 0.0
    j_view[3,3] = 1.0
    j_view[3,4] = f34(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[3,5] = f35(rho,z,k[0],k[1],k[4],k[5],A[0,0],A[0,1],h)
    j_view[3,6] = 0.0
    j_view[3,7] = 0.0

    j_view[4,0] = 0.0
    j_view[4,1] = 0.0
    j_view[4,2] = -h*A[1,0]
    j_view[4,3] = 0.0
    j_view[4,4] = 1.0
    j_view[4,5] = 0.0
    j_view[4,6] = -h*A[1,1]
    j_view[4,7] = 0.0

    j_view[5,0] = 0.0
    j_view[5,1] = 0.0
    j_view[5,2] = 0.0
    j_view[5,3] = -h*A[1,0]
    j_view[5,4] = 0.0
    j_view[5,5] = 1.0
    j_view[5,6] = 0.0
    j_view[5,7] = -h*A[1,1]

    j_view[6,0] = f60(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[6,1] = f61(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[6,2] = 0.0
    j_view[6,3] = 0.0
    j_view[6,4] = f64(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[6,5] = f65(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[6,6] = 1.0
    j_view[6,7] = 0.0

    j_view[7,0] = f70(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[7,1] = f71(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[7,2] = 0.0
    j_view[7,3] = 0.0
    j_view[7,4] = f74(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[7,5] = f75(rho,z,k[0],k[1],k[4],k[5],A[1,0],A[1,1],h)
    j_view[7,6] = 0.0
    j_view[7,7] = 1.0

@cython.cdivision(True)
@cython.cpow(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void J(double[:,:] j_view,double[:] k,double rho,double z,double[:,:] A,double h) noexcept:
    cdef double y1_1,y2_1,y1_2,y2_2
    cdef double f11_1,f12_1,f21_1,f22_1
    cdef double f11_2,f12_2,f21_2,f22_2

    y1_1 = rho+h*(A[0,0]*k[0]+A[0,1]*k[4])
    y2_1 = z+h*(A[0,0]*k[1]+A[0,1]*k[5])
    y1_2 = rho+h*(A[1,0]*k[0]+A[1,1]*k[4])
    y2_2 = z+h*(A[1,0]*k[1]+A[1,1]*k[5])
    f11_1 = f_11(y1_1,y2_1)
    f12_1 = f_12(y1_1,y2_1)
    f21_1 = f_21(y1_1,y2_1)
    f22_1 = f_22(y1_1,y2_1)
    f11_2 = f_11(y1_2,y2_2)
    f12_2 = f_12(y1_2,y2_2)
    f21_2 = f_21(y1_2,y2_2)
    f22_2 = f_22(y1_2,y2_2)

    j_view[0,0] = 1.0
    j_view[0,1] = 0.0
    j_view[0,2] = -h*A[0,0]
    j_view[0,3] = 0.0
    j_view[0,4] = 0.0
    j_view[0,5] = 0.0
    j_view[0,6] = -h*A[0,1]
    j_view[0,7] = 0.0

    j_view[1,0] = 0.0
    j_view[1,1] = 1.0
    j_view[1,2] = 0.0
    j_view[1,3] = -h*A[0,0]
    j_view[1,4] = 0.0
    j_view[1,5] = 0.0
    j_view[1,6] = 0.0
    j_view[1,7] = -h*A[0,1]

    j_view[2,0] = -f11_1*h*A[0,0]
    j_view[2,1] = -f12_1*h*A[0,0]
    j_view[2,2] = 1.0
    j_view[2,3] = 0.0
    j_view[2,4] = -f11_1*h*A[0,1]
    j_view[2,5] = -f12_1*h*A[0,1]
    j_view[2,6] = 0.0
    j_view[2,7] = 0.0

    j_view[3,0] = -f21_1*h*A[0,0]
    j_view[3,1] = -f22_1*h*A[0,0]
    j_view[3,2] = 0.0
    j_view[3,3] = 1.0
    j_view[3,4] = -f21_1*h*A[0,1]
    j_view[3,5] = -f22_1*h*A[0,1]
    j_view[3,6] = 0.0
    j_view[3,7] = 0.0

    j_view[4,0] = 0.0
    j_view[4,1] = 0.0
    j_view[4,2] = -h*A[1,0]
    j_view[4,3] = 0.0
    j_view[4,4] = 1.0
    j_view[4,5] = 0.0
    j_view[4,6] = -h*A[1,1]
    j_view[4,7] = 0.0

    j_view[5,0] = 0.0
    j_view[5,1] = 0.0
    j_view[5,2] = 0.0
    j_view[5,3] = -h*A[1,0]
    j_view[5,4] = 0.0
    j_view[5,5] = 1.0
    j_view[5,6] = 0.0
    j_view[5,7] = -h*A[1,1]

    j_view[6,0] = -f11_2*h*A[1,0]
    j_view[6,1] = -f12_2*h*A[1,0]
    j_view[6,2] = 0.0
    j_view[6,3] = 0.0
    j_view[6,4] = -f11_2*h*A[1,1]
    j_view[6,5] = -f12_2*h*A[1,1]
    j_view[6,6] = 1.0
    j_view[6,7] = 0.0

    j_view[7,0] = -f21_2*h*A[1,0]
    j_view[7,1] = -f22_2*h*A[1,0]
    j_view[7,2] = 0.0
    j_view[7,3] = 0.0
    j_view[7,4] = -f21_2*h*A[1,1]
    j_view[7,5] = -f22_2*h*A[1,1]
    j_view[7,6] = 0.0
    j_view[7,7] = 1.0

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

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void c_matrix_sub(double[:] result,double[:] mat1,double[:] mat2):
    cdef Py_ssize_t i,r
    r = mat1.shape[0]
    for i in range(r):
        result[i] = mat1[i]-mat2[i]



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