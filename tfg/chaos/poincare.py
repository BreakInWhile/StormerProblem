import numpy as np

from typing import Tuple

from tfg.helpers import hamiltonian
from tfg.structures import ReducedDimension,StormerSolvingMethod,Dimension
from tfg.utils.utils import create_logger
from tfg.methods.optim import legendre,qinterpolation

from tfg.methods.optim.SV import StormerVerletMethod

logger = create_logger("Poincaré")

DTYPE = np.float64
NEWTON_ITERS = 100

def application_reduced(
        initial_values: np.ndarray,
        fixed_var: ReducedDimension,
        fixed_value: float,
        time_step: int = .002,
        iterations: int = 10000,
        method: StormerSolvingMethod = StormerVerletMethod(),
        interpolation: str = 'legendre'
) -> Tuple[dict,np.ndarray]:
    """
    [DOESN'T CHECK FOR HAMILTONIAN NEQ NAN!!!]

    Computes Poincaré map for one solution of the reduced Stormer problem.

    Parameters:
        - initial_values: initial values of the problem
        - fixed_var: the fixed dimension. Can be rho,z,d_rho,d_z
        - fixed_value: the value of fixed dimension variable
        - time_step: the time step for simulation of the trajectory
        - iterations: number of iterations for simulation of the trajectory
    
    Returns:
        Dictionary mapping dimension to cut
    """

    assert initial_values.ndim == 1 and initial_values.size == 4, \
        f"Initial values array must be of shape (4,) not {initial_values.shape}"
    assert isinstance(fixed_var,ReducedDimension), \
        f"fixed_var must be one of ReducedDimension values not {type(fixed_var)}"
    
    logger.info(f"Computing Poincaré map for reduced Stormer problem. Method: {method.name}")
    logger.info(f"Interpolation method: {interpolation}")
    logger.info(f"H = {hamiltonian(initial_values)}")

    trajectory = method.simulate(initial_values,time_step,iterations,True)

    if interpolation == 'legendre':
        fixed_values_arr = trajectory[:,fixed_var.value]
        inds = dict() # ind: solution
        # get the indices where trajectory cuts fixed_var = fixed_value
        # if i in inds -> trajectory cuts between [i,i+1]
        for i in range(len(fixed_values_arr)-2):
            if fixed_values_arr[i]<=fixed_value and fixed_values_arr[i+1]>=fixed_value:
                points = np.array([1,2,3],dtype=DTYPE)
                f_points = np.array([fixed_values_arr[i],fixed_values_arr[i+1],fixed_values_arr[i+2]],dtype=DTYPE)
                inds[i] = legendre.get_zero(points,f_points,0)
    else:
        fixed_values_arr = trajectory[:,fixed_var.value]
        if fixed_var == ReducedDimension.RHO or fixed_var == ReducedDimension.Z:
            fixed_values_v_arr = trajectory[:,fixed_var.value+2]
        elif fixed_var == ReducedDimension.P_RHO:
            fixed_values_v_arr = f1(trajectory[:,:2])
        elif fixed_var == ReducedDimension.P_Z:
            fixed_values_v_arr = f2(trajectory[:,:2])
        inds = dict() # ind: solution
        ts = np.array([1,2],dtype=DTYPE)
        fts = np.zeros((2,),dtype=DTYPE)
        dfts = np.zeros((2,),dtype=DTYPE)
        for i in range(len(fixed_values_arr)-2):
            if fixed_values_arr[i]<=fixed_value and fixed_values_arr[i+1]>=fixed_value:
                points = np.array([1,2,3],dtype=np.float64)
                f_points = np.array([fixed_values_arr[i],fixed_values_arr[i+1],fixed_values_arr[i+2]],dtype=np.float64)
                x0 = legendre.get_zero(points,f_points,0)
                fts[0] = fixed_values_arr[i]
                fts[1] = fixed_values_arr[i+1]
                dfts[0] = fixed_values_v_arr[i]
                dfts[1] = fixed_values_v_arr[i+1]
                f_H = qinterpolation.get_H(ts,fts,dfts)
                f_dH = qinterpolation.get_dH(ts,fts,dfts)
                for _ in range(NEWTON_ITERS):
                    x=x0-f_H(x0)/f_dH(x0)
                    if np.abs(x-x0)/np.abs(x0) <= 1e-10:
                        break
                    x0 = x
                inds[i] = x
    
    result = dict()
    ts = np.array([1,2],dtype=DTYPE)
    for var in ReducedDimension:
        if var != fixed_var:
            t_values = trajectory[:,var.value]
            if interpolation == 'hermite':
                if var == ReducedDimension.RHO or var == ReducedDimension.Z:
                    dt_values = trajectory[:,var.value+2]
                elif var == ReducedDimension.P_RHO:
                    dt_values = f1(trajectory[:,:2])
                elif var == ReducedDimension.P_Z:
                    dt_values = f2(trajectory[:,:2])
            sols = np.zeros((len(inds),))
            c = 0
            for i in inds.keys():
                if interpolation == 'legendre':
                    points = np.array([1,2,3],dtype=np.float64)
                    f_points = np.array([t_values[i],t_values[i+1],t_values[i+2]],dtype=np.float64)
                    sol = legendre.interpolador_Legendre(np.array([inds[i]]),points,f_points)
                    sols[c] = sol[0]
                else:
                    fts[0] = t_values[i]
                    fts[1] = t_values[i+1]
                    dfts[0] = dt_values[i]
                    dfts[1] = dt_values[i+1]
                    sols[c] = qinterpolation.H(inds[i],ts,fts,dfts)
                c+=1
            result[var] = sols

    return result,initial_values

    

def application_general(
        initial_values: np.ndarray,
        fixed_var: Dimension,
        fixed_value: float,
        time_step: int = .002,
        iterations: int = 10000,
        method: StormerSolvingMethod = StormerVerletMethod(),
        interpolation: str = 'legendre'
) -> Tuple[dict,np.ndarray]:
    """
    [DOESN'T CHECK FOR HAMILTONIAN NEQ NAN!!!]

    Computes Poincaré map for one solution of the Stormer problem.

    Parameters:
        - initial_values: initial values of the problem
        - fixed_var: the fixed dimension. Can be rho,z,d_rho,d_z
        - fixed_value: the value of fixed dimension variable
        - time_step: the time step for simulation of the trajectory
        - iterations: number of iterations for simulation of the trajectory
    
    Returns:
        Dictionary mapping dimension to cut
    """

    assert initial_values.ndim == 1 and initial_values.size == 6, \
        f"Initial values array must be of shape (6,) not {initial_values.shape}"
    assert isinstance(fixed_var,Dimension), \
        f"fixed_var must be one of Dimension values not {type(fixed_var)}"
    
    logger.info(f"Computing Poincaré map for reduced Stormer problem. Method: {method.name}")
    logger.info(f"Interpolation method: {interpolation}")
    logger.info(f"H = {hamiltonian(initial_values)}")

    trajectory = method.simulate(initial_values,time_step,iterations,True)
    np.savez("sols",np.array(trajectory))

    if trajectory is None:
        return None
    
    if interpolation == 'legendre':
        fixed_values_arr = trajectory[:,fixed_var.value]
        inds = dict() # ind: solution
        # get the indices where trajectory cuts fixed_var = fixed_value
        # if i in inds -> trajectory cuts between [i,i+1]
        for i in range(len(fixed_values_arr)-2):
            if fixed_values_arr[i]<=fixed_value and fixed_values_arr[i+1]>=fixed_value:
                points = np.array([1,2,3],dtype=DTYPE)
                f_points = np.array([fixed_values_arr[i],fixed_values_arr[i+1],fixed_values_arr[i+2]],dtype=DTYPE)
                inds[i] = legendre.get_zero(points,f_points,0)
    else:
        fixed_values_arr = trajectory[:,fixed_var.value]
        if fixed_var == Dimension.X or fixed_var == Dimension.Y or fixed_var == Dimension.Z:
            fixed_values_v_arr = trajectory[:,fixed_var.value+3]
        elif fixed_var == Dimension.DX:
            fixed_values_v_arr = general_f1(trajectory)
        elif fixed_var == Dimension.DY:
            fixed_values_v_arr = general_f2(trajectory)
        elif fixed_var == Dimension.DZ:
            fixed_values_v_arr = general_f3(trajectory)
        inds = dict() # ind: solution
        ts = np.array([1,2],dtype=DTYPE)
        fts = np.zeros((2,),dtype=DTYPE)
        dfts = np.zeros((2,),dtype=DTYPE)
        for i in range(len(fixed_values_arr)-2):
            if fixed_values_arr[i]<=fixed_value and fixed_values_arr[i+1]>=fixed_value:
                points = np.array([1,2,3],dtype=np.float64)
                f_points = np.array([fixed_values_arr[i],fixed_values_arr[i+1],fixed_values_arr[i+2]],dtype=np.float64)
                x0 = legendre.get_zero(points,f_points,0)
                fts[0] = fixed_values_arr[i]
                fts[1] = fixed_values_arr[i+1]
                dfts[0] = fixed_values_v_arr[i]
                dfts[1] = fixed_values_v_arr[i+1]
                f_H = qinterpolation.get_H(ts,fts,dfts)
                f_dH = qinterpolation.get_dH(ts,fts,dfts)
                for _ in range(NEWTON_ITERS):
                    x=x0-f_H(x0)/f_dH(x0)
                    if np.abs(x-x0)/np.abs(x0) <= 1e-10:
                        break
                    x0 = x
                inds[i] = x
    
    result = dict()
    ts = np.array([1,2],dtype=DTYPE)
    for var in Dimension:
        if var != fixed_var:
            t_values = trajectory[:,var.value]
            if interpolation == 'hermite':
                if var == Dimension.X or var == Dimension.Y or var == Dimension.Z:
                    dt_values = trajectory[:,var.value+3]
                elif var == Dimension.DX:
                    dt_values = general_f1(trajectory)
                elif var == Dimension.DY:
                    dt_values = general_f2(trajectory)
                elif var == Dimension.DZ:
                    dt_values = general_f3(trajectory)
            sols = np.zeros((len(inds),))
            c = 0
            for i in inds.keys():
                if interpolation == 'legendre':
                    points = np.array([1,2,3],dtype=np.float64)
                    f_points = np.array([t_values[i],t_values[i+1],t_values[i+2]],dtype=np.float64)
                    sol = legendre.interpolador_Legendre(np.array([inds[i]]),points,f_points)
                    sols[c] = sol[0]
                else:
                    fts[0] = t_values[i]
                    fts[1] = t_values[i+1]
                    dfts[0] = dt_values[i]
                    dfts[1] = dt_values[i+1]
                    sols[c] = qinterpolation.H(inds[i],ts,fts,dfts)
                c+=1
            result[var] = sols

    return result,initial_values


def f1(rz_arr):
    res = np.zeros((rz_arr.shape[0],))
    for i in range(rz_arr.shape[0]):
        rho,z = rz_arr[i]
        rad = np.sqrt(rho**2 + z**2)
        term = (rho/rad**3 - 1/rho)
        res[i] = term*(-1.0/rho**2.0 - 1.0/rad**3.0 + (3.0*rho**2.0)/rad**5.0)
    return res

def f2(rz_arr):
    res = np.zeros((rz_arr.shape[0],))
    for i in range(rz_arr.shape[0]):
        rho,z = rz_arr[i]
        rad = np.sqrt(rho**2 + z**2)
        term = (rho/rad**3 - 1/rho)
        res[i] = term*((3.0*rho*z)/rad**5.0)
    return res

def general_f1(point) -> float:
    res = np.zeros((point.shape[0],))
    for i in range(point.shape[0]):
        x,y,z,dx,dy,dz = point[i]
        rad = np.sqrt(x**2.0+y**2.0+z**2.0)
        term = (3.0*z/rad**5.0)
        res[i] = term*(dy*z - dz*y)-dy/rad**3.0
    return res

def general_f2(point) -> float:
    res = np.zeros((point.shape[0],))
    for i in range(point.shape[0]):
        x,y,z,dx,dy,dz = point[i]
        rad = np.sqrt(x**2.0+y**2.0+z**2.0)
        term = (3.0*z/rad**5.0)
        res[i] = -term*(dx*z - dz*x)+dx/rad**3.0
    return res

def general_f3(point) -> float:
    res = np.zeros((point.shape[0],))
    for i in range(point.shape[0]):
        x,y,z,dx,dy,dz = point[i]
        rad = np.sqrt(x**2.0+y**2.0+z**2.0)
        term = (3.0*z/rad**5.0)
        res[i] = term*(dx*y - dy*x)
    return res