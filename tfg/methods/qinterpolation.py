from typing import Iterable,List,Callable

# ----- Functions ----- #
def H(t: float,ts: Iterable,fts: Iterable,dfts: Iterable) -> float:
    """
    Get the H polynomial obtained using Hermite interpolation
    evaluated at t.

    (*) For the moment the polynomial is constructed only with 
    two points [t0,t1]

    Useful for Poincaré maps, where only one point is needed
    and a new polynomial will be constructed each time. For more general
    use case please consider get_H().

    Parameters:
        - t: point where to evaluate H(t)
        - ts: [t0,t1]
        - fts: [f(t0),f(t1)]
        - dfts: [f'(t0),f'(t1)]
    
    Returns:
        H(t) for t in x
    """

    h = (ts[1]-ts[0])
    Zs1s2 = (fts[1]-fts[0])/h
    Zs0s1s2 = (Zs1s2-dfts[0])/h
    Zs1s2s3 = (dfts[1]-Zs1s2)/h
    Q0 = fts[0]
    Q1 = dfts[0]
    Q2 = Zs0s1s2
    Q3 = (Zs1s2s3-Zs0s1s2)/h

    return Q0 + Q1*(t-ts[0]) + Q2*((t-ts[0])**2) + Q3*((t-ts[0])**2)*(t-ts[1])

def get_H(ts: Iterable,fts: Iterable,dfts: Iterable) -> Callable:
    """
    Get the H(t) polynomial using Hermite interpolation.

    (*) For the moment only with two points [t0,t1]

    Parameters:
        - ts: [t0,t1]
        - fts: [f(t0),f(t1)]
        - dfts: [f'(t0),f'(t1)]
    
    Returns:
        H(t) as a function

    Example:
        > ts = np.array([1/3,1/2])
        > fts = [ts[0]**2,ts[1]**2]
        > dfts = [2*ts[0],2*ts[1]]
        > f_H = get_H(ts,fts,dfts)
        > f_H(4)
        16.00000000000015
    """

    h = (ts[1]-ts[0])
    Zs1s2 = (fts[1]-fts[0])/h
    Zs0s1s2 = (Zs1s2-dfts[0])/h
    Zs1s2s3 = (dfts[1]-Zs1s2)/h
    Q0 = fts[0]
    Q1 = dfts[0]
    Q2 = Zs0s1s2
    Q3 = (Zs1s2s3-Zs0s1s2)/h

    def H_fun(x):
        return Q0 + Q1*(x-ts[0]) + Q2*((x-ts[0])**2) + Q3*((x-ts[0])**2)*(x-ts[1])

    return H_fun

def get_dH(ts: Iterable,fts: Iterable,dfts: Iterable) -> Callable:
    """
    Get the dH/dt.

    Parameters:
        - xs: points where to evaluate H(t)
        - ts: [t0,t1]
        - fts: [f(t0),f(t1)]
        - dfts: [f'(t0),f'(t1)]
    
    Returns:
        dH/dt as a function
    """

    h = (ts[1]-ts[0])
    Zs1s2 = (fts[1]-fts[0])/h
    Zs0s1s2 = (Zs1s2-dfts[0])/h
    Zs1s2s3 = (dfts[1]-Zs1s2)/h
    Q0 = fts[0]
    Q1 = dfts[0]
    Q2 = Zs0s1s2
    Q3 = (Zs1s2s3-Zs0s1s2)/h

    def dH_fun(x):
        return Q1*x + 2*Q2*(x-ts[0]) + Q3*(x-ts[0])*(3*x-2*ts[1]-ts[0])

    return dH_fun