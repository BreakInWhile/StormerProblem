"""
Implements Newton-Raphson method
"""

import numpy as np
from typing import Callable, Tuple

from tfg.utils.utils import create_logger

logger = create_logger("Newton-Raphson")

def computeOld(
        p: np.ndarray,
        f: Callable[[np.ndarray],np.ndarray],
        J: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, int]:
    
    """
    Computes Newton-Raphson method.

    Given the initial point p, searches for c such that
    f(c) = 0.

    Parameters:
        - p: initial point [p1,p2...pn]
        - f: function f(p)
        - J: Jacobian matriz of f
        - threshold: when to stop the method
    """

    assert p.ndim == 1, f"p must be of shape (N,) not {p.shape}"
    #tmp
    assert p.size == 2, f"Falta por implementar para cualquier N!=2"
    assert J.ndim ==2 and J.shape[0] == J.shape[1],\
            f"J must be square matriz of shape (N,N) not {J.shape}"
    assert p.size == J.shape[0],\
            f"p and J have incompatible shapes {p.shape}{J.shape}"

    logger.info(f"p={p}")

    p_prev = p.copy()

    if p.size == 2:
        logger.info("N = 2. Can compute inverse of J directly.")
    counter = 0
    while True:
        # Calculate Jacobian matrix for next step
        eval_J = np.empty(J.shape)
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                eval_J[i][j] = J[i][j](p_prev)
        logger.debug(f"J(p) = {eval_J}")

        # can compute inverse of J
        if p.size == 2:
            fx = eval_J[0][0]
            fy = eval_J[0][1]
            gx = eval_J[1][0]
            gy = eval_J[1][1]

            tmp_mat = np.array([
                [gy,-fy],
                [-gx,fx]
            ])

            inverse = 1/(fx*gy - fy*gx) * tmp_mat
            logger.debug(f"inverse of J = {inverse}")
            logger.debug(f"f(p) = {f(p)}")

            p_prev = p_prev - inverse@f(p_prev)

        #TODO: N > 2

        logger.debug(f"p_next = {p_prev}")
        logger.debug(f"mod(p_next) = {np.sqrt(np.sum(p_prev**2))} ")
        counter += 1
        if np.sqrt(np.sum(f(p_prev)**2)) < threshold:
            break

    return p_prev,counter

def compute(x0,threshold,iterations,F,J):
    """
    Newton-Raphson method

    Parameters:
        - x0: the seed
        - threshold: when to stop
        - iterations: maximum number of iterations
        - F: the function
        - J: the Jacobian of the function 
    """
    x=x0
     
    for i in range(iterations):
         
        x=x-F(x)/J(x)
     
        if np.abs(F(x)) <= threshold:
            break
    return x
