import cython
import numpy as np
cimport numpy as np
from sympy import symbols

cdef extern from "math.h":
    double sqrt(double x)

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double polinomio_Legendre(double x,Py_ssize_t k,double[:] puntos_iniciales):
    cdef double res = 1
    cdef Py_ssize_t i,x_max
    x_max = puntos_iniciales.shape[0]
    for i in range(x_max):
        if i != k:
            res *= (x-puntos_iniciales[i])/(puntos_iniciales[k]-puntos_iniciales[i])
    return res

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)    
def interpolador_Legendre(double[:] points,double[:] puntos_iniciales,double[:] f_puntos_iniciales):
    cdef list f_points = []
    cdef Py_ssize_t i,j,points_max,x_max
    points_max = points.shape[0]
    x_max = puntos_iniciales.shape[0]
    for j in range(points_max):
        res = 0
        for i in range(x_max):
            res += polinomio_Legendre(points[j],i,puntos_iniciales)*f_puntos_iniciales[i]
        f_points.append(res)
    return f_points

##### Same but symbolicaly (VERY SLOW!) #####
@cython.boundscheck(False) 
@cython.wraparound(False)
def SYM_polinomio_Legendre(Py_ssize_t k,double[:] puntos_iniciales):
    cdef Py_ssize_t x_max = puntos_iniciales.shape[0]  
    cdef Py_ssize_t i
    x = symbols('x')
    L = 1
    for i in range(x_max):
        if i != k:
            L *= (x-puntos_iniciales[i])/(puntos_iniciales[k]-puntos_iniciales[i])
    return L

@cython.boundscheck(False) 
@cython.wraparound(False)
def SYM_interpolador_Legendre(double[:] puntos_iniciales,double[:] f_puntos_iniciales):
    cdef Py_ssize_t x_max = puntos_iniciales.shape[0]    
    cdef Py_ssize_t i
    P = 0
    for i in range(x_max):
        P += SYM_polinomio_Legendre(i,puntos_iniciales)*f_puntos_iniciales[i]
    return P
##############################################

def get_zero(points,f_points,fact):
    """
    Asumo que corta entre points[0] y points[1]
    """
    #P = SYM_interpolador_Legendre(points,f_points).simplify().as_poly()
    #a,b,c = [np.float64(x) for x in P.all_coeffs()]
    return _get_zero_aux(points,f_points,fact)

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double _get_zero_aux(double[:] points,double[:] f,double fact):
    """
    Asumo que corta entre points[0] y points[1]
    """
    cdef double s1,s2,a,b,c
    cdef double t01,t02,t12,t10,t20,t21
    cdef double term0,term1,term2

    t01 = points[0]-points[1]
    t02 = points[0]-points[2]
    term0 = t01*t02
    t10 = -t01
    t12 = points[1]-points[2]
    term1 = t10*t12
    t20 = -t02
    t21 = -t12
    term2 = t20*t21

    a = f[0]/term0 + f[1]/term1 + f[2]/term2
    b = -f[0]/term0*(points[1]+points[2]) - f[1]/term1*(points[0]+points[2]) - f[2]/term2*(points[0]+points[1])
    c = (f[0]*points[1]*points[2])/term0 + (f[1]*points[0]*points[2])/term1 + (f[2]*points[0]*points[1])/term2 - fact

    s1 = (-b - sqrt(b**2 - 4*a*c))/(2*a)
    s2 = (-b + sqrt(b**2 - 4*a*c))/(2*a)

    if s1 >= points[0] and s1 <= points[1]: return s1
    return s2