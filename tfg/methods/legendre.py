from sympy import symbols,sqrt
from sympy.solvers import solve

def polinomio_Legendre(x,k,puntos_iniciales):
    res = 1
    for i in range(len(puntos_iniciales)):
        if i != k:
            res *= (x-puntos_iniciales[i])/(puntos_iniciales[k]-puntos_iniciales[i])
    return res
    
def interpolador_Legendre(points,puntos_iniciales,f_puntos_iniciales):
    f_points = []
    for point in points:
        res = 0
        for i in range(len(puntos_iniciales)):
            res += polinomio_Legendre(point,i,puntos_iniciales)*f_puntos_iniciales[i]
        f_points.append(res)
    return f_points

##### Same but symbolicaly (VERY SLOW!) #####
def SYM_polinomio_Legendre(k,puntos_iniciales):
    x = symbols('x')
    L = 1
    for i in range(len(puntos_iniciales)):
        if i != k:
            L *= (x-puntos_iniciales[i])/(puntos_iniciales[k]-puntos_iniciales[i])
    return L
    
def SYM_interpolador_Legendre(puntos_iniciales,f_puntos_iniciales):
    P = 0
    for i in range(len(puntos_iniciales)):
        P += SYM_polinomio_Legendre(i,puntos_iniciales)*f_puntos_iniciales[i]
    return P
##############################################

def get_zero(points,f_points,fact):
    """
    Asumo que corta entre points[0] y points[1]
    """
    P = SYM_interpolador_Legendre(points,f_points).simplify()
    sols = solve(P-fact)
    return sols[0] if points[0] <= sols[0] <= points[1] else sols[1]