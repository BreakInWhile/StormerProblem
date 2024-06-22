import sys
import cython
import numpy as np
cimport numpy as np

from tqdm import tqdm

from tfg.structures import StormerSolvingMethod
from tfg.utils.utils import create_logger

NAME = "Gauss Method (General)"

logger = create_logger(NAME)
DTYPE = np.float64

cdef double NEWTON_THRESHOLD = 1e-13
cdef int NEWTON_ITERS = 100

cdef extern from "math.h":
    double sqrt(double x)

class GaussMethod(StormerSolvingMethod):
    def __init__(self):
        super().__init__(NAME)

    @cython.cdivision(True)
    @cython.cpow(True)
    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def simulate(
        self,
        double[:] initial_values,
        double time_step,
        int iterations,
        bint verbose
    ):
        cdef double x,y,z,dx,dy,dz,rad,term,f1_eval,f2_eval,f3_eval
        cdef int counter
        cdef double[:,:] A_view,f_view,j_view,j_augmented_view
        cdef double[:] y1_view,y2_view,b_view,mmul_view

        if verbose: logger.info(f"Simulating for {iterations} iterations...")

        # ----- Views ----- #
        y1_view = np.zeros((6,),dtype=DTYPE)
        y2_view = np.zeros((6,),dtype=DTYPE)
        A_view = np.array([
            [1.0/4.0,1.0/4.0-sqrt(3.0)/6.0],
            [1.0/4.0+sqrt(3.0)/6.0,1.0/4.0]
        ])
        b_view = np.array([1.0/2.0,1.0/2.0])
        f = np.zeros((12,1),dtype=DTYPE)
        f_view = f
        j = np.zeros((12,12),dtype=DTYPE)
        j_view = j
        j_augmented_view = np.zeros((12,24),dtype=DTYPE)
        mmul = np.zeros((12,),dtype=DTYPE)
        mmul_view = mmul
        # ----------------- #

        x = initial_values[0]
        y = initial_values[1]
        z = initial_values[2]
        dx = initial_values[3]
        dy = initial_values[4]
        dz = initial_values[5]

        solutions = [initial_values]
        x0 = np.empty((12,),dtype=DTYPE)
        for _ in tqdm(range(iterations)):

            rad = sqrt(x**2.0+y**2.0+z**2.0)
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
            
            counter = 0
            for _ in range(NEWTON_ITERS):
                y1_view[0] = x+time_step*(A_view[0,0]*x0[0]+A_view[0,1]*x0[6])
                y1_view[1] = y+time_step*(A_view[0,0]*x0[1]+A_view[0,1]*x0[7])
                y1_view[2] = z+time_step*(A_view[0,0]*x0[2]+A_view[0,1]*x0[8])
                y1_view[3] = dx+time_step*(A_view[0,0]*x0[3]+A_view[0,1]*x0[9])
                y1_view[4] = dy+time_step*(A_view[0,0]*x0[4]+A_view[0,1]*x0[10])
                y1_view[5] = dz+time_step*(A_view[0,0]*x0[5]+A_view[0,1]*x0[11])

                y2_view[0] = x+time_step*(A_view[1,0]*x0[0]+A_view[1,1]*x0[6])
                y2_view[1] = y+time_step*(A_view[1,0]*x0[1]+A_view[1,1]*x0[7])
                y2_view[2] = z+time_step*(A_view[1,0]*x0[2]+A_view[1,1]*x0[8])
                y2_view[3] = dx+time_step*(A_view[1,0]*x0[3]+A_view[1,1]*x0[9])
                y2_view[4] = dy+time_step*(A_view[1,0]*x0[4]+A_view[1,1]*x0[10])
                y2_view[5] = dz+time_step*(A_view[1,0]*x0[5]+A_view[1,1]*x0[11])

                F(f_view,x0,y1_view,y2_view,A_view,time_step)
                J(j_view,x0,y1_view,y2_view,A_view,time_step)
                c_invert(j_augmented_view,j_view)
                c_matrix_multiply(mmul_view,j_view,f_view)
                x_res=x0-mmul
                if sqrt(np.sum((x_res-x0)**2))/sqrt(np.sum(x0**2)) <= NEWTON_THRESHOLD:
                    break
                x0 = x_res
                counter+=1

            if counter == NEWTON_THRESHOLD: logger.debug(f"{counter}: No ha convergido")
            
            slopes_sum = np.zeros((6,))
            for i in range(2):
                slopes_sum += time_step*b_view[i]*x_res[6*i:6*(1+i)]
            
            x+=slopes_sum[0]
            y+=slopes_sum[1]
            z+=slopes_sum[2]
            dx+=slopes_sum[3]
            dy+=slopes_sum[4]
            dz+=slopes_sum[5]
            solutions.append(np.array([x,y,z,dx,dy,dz]))
        return np.array(solutions)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f1(double y,double z,double dy,double dz,double rad,double term) noexcept:
    return term*(dy*z - dz*y)-dy/rad**3.0

@cython.cdivision(True)
@cython.cpow(True)
cdef double f2(double x,double z,double dx,double dz,double rad,double term) noexcept:
    return -term*(dx*z - dz*x)+dx/rad**3.0

cdef double f3(double x,double y,double dx,double dy,double term) noexcept:
    return term*(dx*y - dy*x)

@cython.cdivision(True)
@cython.cpow(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void F(double[:,:] f_view,double [:] k,double[:] y1,double[:] y2,double[:,:] A_view,double h) noexcept:
    cdef double rad_1,term_1
    cdef double rad_2,term_2

    rad_1 = sqrt(y1[0]**2.0+y1[1]**2.0+y1[2]**2.0)
    rad_2 = sqrt(y2[0]**2.0+y2[1]**2.0+y2[2]**2.0)
    term_1 = (3.0*y1[2]/rad_1**5.0)
    term_2 = (3.0*y2[2]/rad_2**5.0)

    f_view[0][0] = k[0]-y1[3]
    f_view[1][0] = k[1]-y1[4]
    f_view[2][0] = k[2]-y1[5]
    f_view[3][0] = k[3]-f1(y1[1],y1[2],y1[4],y1[5],rad_1,term_1)
    f_view[4][0] = k[4]-f2(y1[0],y1[2],y1[3],y1[5],rad_1,term_1)
    f_view[5][0] = k[5]-f3(y1[0],y1[1],y1[3],y1[4],term_1)
    f_view[6][0] = k[6]-y2[3]
    f_view[7][0] = k[7]-y2[4]
    f_view[8][0] = k[8]-y2[5]
    f_view[9][0] = k[9]-f1(y2[1],y2[2],y2[4],y2[5],rad_2,term_2)
    f_view[10][0] = k[10]-f2(y2[0],y2[2],y2[3],y2[5],rad_2,term_2)
    f_view[11][0] = k[11]-f3(y2[0],y2[1],y2[3],y2[4],term_2)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f11(double x,double y,double z,double dy,double dz) noexcept:
    return 3.0*dy*x*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*x*z*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f12(double x,double y,double z,double dy,double dz) noexcept:
    return 3.0*dy*y*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 3.0*dz*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*y*z*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f13(double x,double y,double z,double dy,double dz) noexcept:
    return 6.0*dy*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*z**2*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-3.5) + 3.0*(dy*z - dz*y)*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f15(double x,double y,double z) noexcept:
    return 3.0*z**2*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - (x**2.0 + y**2.0 + z**2.0)**(-1.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f16(double x,double y,double z) noexcept:
    return -3.0*y*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f21(double x,double y,double z,double dx,double dz) noexcept:
    return -3.0*dx*x*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 3.0*dz*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 15.0*x*z*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f22(double x,double y,double z,double dx,double dz) noexcept:
    return -3.0*dx*y*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 15.0*y*z*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f23(double x,double y,double z,double dx,double dz) noexcept:
    return -6.0*dx*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + 15.0*z**2*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5) - 3.0*(dx*z - dz*x)*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f24(double x,double y,double z) noexcept:
    return -3.0*z**2*(x**2.0 + y**2.0 + z**2.0)**(-2.5) + (x**2.0 + y**2.0 + z**2.0)**(-1.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f26(double x,double y,double z) noexcept:
    return 3.0*x*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f31(double x,double y,double z,double dx,double dy) noexcept:
    return -3.0*dy*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*x*z*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f32(double x,double y,double z,double dx,double dy) noexcept:
    return 3.0*dx*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5) - 15.0*y*z*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f33(double x,double y,double z,double dx,double dy) noexcept:
    return -15.0*z**2*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-3.5) + 3.0*(dx*y - dy*x)*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f34(double x,double y,double z) noexcept:
    return 3.0*y*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

@cython.cdivision(True)
@cython.cpow(True)
cdef double f35(double x,double y,double z) noexcept:
    return -3.0*x*z*(x**2.0 + y**2.0 + z**2.0)**(-2.5)

@cython.cdivision(True)
@cython.cpow(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void J(double[:,:] j_view,double[:] k,double[:] y1,double[:] y2,double[:,:] A_view,double h):
    cdef double ha11,ha12,ha21,ha22
    cdef double f11_1,f12_1,f13_1,f15_1,f16_1,f21_1,f22_1,f23_1,f24_1,f26_1,f31_1,f32_1,f33_1,f34_1,f35_1
    cdef double f11_2,f12_2,f13_2,f15_2,f16_2,f21_2,f22_2,f23_2,f24_2,f26_2,f31_2,f32_2,f33_2,f34_2,f35_2

    ha11 = -h*A_view[0,0]
    ha12 = -h*A_view[0,1]
    ha21 = -h*A_view[1,0]
    ha22 = -h*A_view[1,1]

    f11_1 = f11(y1[0],y1[1],y1[2],y1[4],y1[5])
    f12_1 = f12(y1[0],y1[1],y1[2],y1[4],y1[5])
    f13_1 = f13(y1[0],y1[1],y1[2],y1[4],y1[5])
    f15_1 = f15(y1[0],y1[1],y1[2])
    f16_1 = f16(y1[0],y1[1],y1[2])
    f21_1 = f21(y1[0],y1[1],y1[2],y1[3],y1[5])
    f22_1 = f22(y1[0],y1[1],y1[2],y1[3],y1[5])
    f23_1 = f23(y1[0],y1[1],y1[2],y1[3],y1[5])
    f24_1 = f24(y1[0],y1[1],y1[2])
    f26_1 = f26(y1[0],y1[1],y1[2])
    f31_1 = f31(y1[0],y1[1],y1[2],y1[3],y1[4])
    f32_1 = f32(y1[0],y1[1],y1[2],y1[3],y1[4])
    f33_1 = f33(y1[0],y1[1],y1[2],y1[3],y1[4])
    f34_1 = f34(y1[0],y1[1],y1[2])
    f35_1 = f35(y1[0],y1[1],y1[2])

    f11_2 = f11(y2[0],y2[1],y2[2],y2[4],y2[5])
    f12_2 = f12(y2[0],y2[1],y2[2],y2[4],y2[5])
    f13_2 = f13(y2[0],y2[1],y2[2],y2[4],y2[5])
    f15_2 = f15(y2[0],y2[1],y2[2])
    f16_2 = f16(y2[0],y2[1],y2[2])
    f21_2 = f21(y2[0],y2[1],y2[2],y2[3],y2[5])
    f22_2 = f22(y2[0],y2[1],y2[2],y2[3],y2[5])
    f23_2 = f23(y2[0],y2[1],y2[2],y2[3],y2[5])
    f24_2 = f24(y2[0],y2[1],y2[2])
    f26_2 = f26(y2[0],y2[1],y2[2])
    f31_2 = f31(y2[0],y2[1],y2[2],y2[3],y2[4])
    f32_2 = f32(y2[0],y2[1],y2[2],y2[3],y2[4])
    f33_2 = f33(y2[0],y2[1],y2[2],y2[3],y2[4])
    f34_2 = f34(y2[0],y2[1],y2[2])
    f35_2 = f35(y2[0],y2[1],y2[2])

    j_view[0][0] = 1.0
    j_view[0][1] = 0.0
    j_view[0][2] = 0.0
    j_view[0][3] = ha11
    j_view[0][4] = 0.0
    j_view[0][5] = 0.0
    j_view[0][6] = 0.0
    j_view[0][7] = 0.0
    j_view[0][8] = 0.0
    j_view[0][9] = ha12
    j_view[0][10] = 0.0
    j_view[0][11] = 0.0

    j_view[1][0] = 0.0
    j_view[1][1] = 1.0
    j_view[1][2] = 0.0
    j_view[1][3] = 0.0
    j_view[1][4] = ha11
    j_view[1][5] = 0.0
    j_view[1][6] = 0.0
    j_view[1][7] = 0.0
    j_view[1][8] = 0.0
    j_view[1][9] = 0.0
    j_view[1][10] = ha12
    j_view[1][11] = 0.0

    j_view[2][0] = 0.0
    j_view[2][1] = 0.0
    j_view[2][2] = 1.0
    j_view[2][3] = 0.0
    j_view[2][4] = 0.0
    j_view[2][5] = ha11
    j_view[2][6] = 0.0
    j_view[2][7] = 0.0
    j_view[2][8] = 0.0
    j_view[2][9] = 0.0
    j_view[2][10] = 0.0
    j_view[2][11] = ha12

    j_view[3][0] = f11_1*ha11
    j_view[3][1] = f12_1*ha11
    j_view[3][2] = f13_1*ha11
    j_view[3][3] = 1.0
    j_view[3][4] = f15_1*ha11
    j_view[3][5] = f16_1*ha11
    j_view[3][6] = f11_1*ha12
    j_view[3][7] = f12_1*ha12
    j_view[3][8] = f13_1*ha12
    j_view[3][9] = 0.0
    j_view[3][10] = f15_1*ha12
    j_view[3][11] = f16_1*ha12

    j_view[4][0] = f21_1*ha11
    j_view[4][1] = f22_1*ha11
    j_view[4][2] = f23_1*ha11
    j_view[4][3] = f24_1*ha11
    j_view[4][4] = 1.0
    j_view[4][5] = f26_1*ha11
    j_view[4][6] = f21_1*ha12
    j_view[4][7] = f22_1*ha12
    j_view[4][8] = f23_1*ha12
    j_view[4][9] = f24_1*ha12
    j_view[4][10] = 0.0
    j_view[4][11] = f26_1*ha12

    j_view[5][0] = f31_1*ha11
    j_view[5][1] = f32_1*ha11
    j_view[5][2] = f33_1*ha11
    j_view[5][3] = f34_1*ha11
    j_view[5][4] = f35_1*ha11
    j_view[5][5] = 1.0
    j_view[5][6] = f31_1*ha12
    j_view[5][7] = f32_1*ha12
    j_view[5][8] = f33_1*ha12
    j_view[5][9] = f34_1*ha12
    j_view[5][10] = f35_1*ha12
    j_view[5][11] = 0.0

    j_view[6][0] = 0.0
    j_view[6][1] = 0.0
    j_view[6][2] = 0.0
    j_view[6][3] = ha21
    j_view[6][4] = 0.0
    j_view[6][5] = 0.0
    j_view[6][6] = 1.0
    j_view[6][7] = 0.0
    j_view[6][8] = 0.0
    j_view[6][9] = ha22
    j_view[6][10] = 0.0
    j_view[6][11] = 0.0

    j_view[7][0] = 0.0
    j_view[7][1] = 0.0
    j_view[7][2] = 0.0
    j_view[7][3] = 0.0
    j_view[7][4] = ha21
    j_view[7][5] = 0.0
    j_view[7][6] = 0.0
    j_view[7][7] = 1.0
    j_view[7][8] = 0.0
    j_view[7][9] = 0.0
    j_view[7][10] = ha22
    j_view[7][11] = 0.0

    j_view[8][0] = 0.0
    j_view[8][1] = 0.0
    j_view[8][2] = 0.0
    j_view[8][3] = 0.0
    j_view[8][4] = 0.0
    j_view[8][5] = ha21
    j_view[8][6] = 0.0
    j_view[8][7] = 0.0
    j_view[8][8] = 1.0
    j_view[8][9] = 0.0
    j_view[8][10] = 0.0
    j_view[8][11] = ha22

    j_view[9][0] = f11_2*ha21
    j_view[9][1] = f12_2*ha21
    j_view[9][2] = f13_2*ha21
    j_view[9][3] = 0.0
    j_view[9][4] = f15_2*ha21
    j_view[9][5] = f16_2*ha21
    j_view[9][6] = f11_2*ha22
    j_view[9][7] = f12_2*ha22
    j_view[9][8] = f13_2*ha22
    j_view[9][9] = 1.0
    j_view[9][10] = f15_2*ha22
    j_view[9][11] = f16_2*ha22

    j_view[10][0] = f21_2*ha21
    j_view[10][1] = f22_2*ha21
    j_view[10][2] = f23_2*ha21
    j_view[10][3] = f24_2*ha21
    j_view[10][4] = 0.0
    j_view[10][5] = f26_2*ha21
    j_view[10][6] = f21_2*ha22
    j_view[10][7] = f22_2*ha22
    j_view[10][8] = f23_2*ha22
    j_view[10][9] = f24_2*ha22
    j_view[10][10] = 1.0
    j_view[10][11] = f26_2*ha22

    j_view[11][0] = f31_2*ha21
    j_view[11][1] = f32_2*ha21
    j_view[11][2] = f33_2*ha21
    j_view[11][3] = f34_2*ha21
    j_view[11][4] = f35_2*ha21
    j_view[11][5] = 0.0
    j_view[11][6] = f31_2*ha22
    j_view[11][7] = f32_2*ha22
    j_view[11][8] = f33_2*ha22
    j_view[11][9] = f34_2*ha22
    j_view[11][10] = f35_2*ha22
    j_view[11][11] = 1.0

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