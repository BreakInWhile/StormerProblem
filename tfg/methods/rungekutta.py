import numpy as np
import tfg.utils.utils as utils
from typing import Callable,Tuple,List
from tqdm import tqdm

from tfg.methods import newton

logger = utils.create_logger("Runge-Kutta")

def not_thread_safe(func):
    """
    Decorator to indicate that a function is not thread-safe.
    """
    def wrapper(*args, **kwargs):
        logger.warning("Not thread safe function!")
        return func(*args, **kwargs)
    return wrapper

class RungeKuttaParams():
    def __init__(
            self,
            t0 = 0.0,
            y0: np.ndarray = np.array([.0,.0]),
            sz = .001
        ) -> None:
        self.t0 = t0
        self.y0 = y0
        self.sz = sz

class PartitionedRungeKuttaParams():
    """
    Parámetros para el método Runge-Kutta partido
    """
    def __init__(
            self,
            q: np.ndarray,
            p: np.ndarray,
            f: Callable[[np.ndarray,np.ndarray],np.ndarray],
            g: Callable[[np.ndarray,np.ndarray],np.ndarray]
        ) -> None:

        """
        Params:
            - q: [q1,q2...qn]
            - p: [p1,p2...pn]
        """

        self.q = q
        self.p = p
        self.f = f
        self.g = g

@not_thread_safe
def compute(
        f: Callable[[float,np.ndarray],np.ndarray],
        A: np.ndarray,
        b: np.ndarray,
        params: RungeKuttaParams,
        n_steps: int = 1,
    ) -> np.ndarray:

    shape_A = A.shape
    shape_b = b.shape
    sz = params.sz
    assert shape_A[0] == shape_A[1], "La matriz debe ser cuadrada"
    assert shape_b == (shape_A[0],), f"A.shape = {shape_A} vs. b.shape = {shape_b}"
    assert sz > 0, f"h debe ser > 0, pero = {sz}"


    t0 = params.t0
    y0 = params.y0

    c = np.array([np.sum(line) for line in A])
    solutions = []

    for _ in tqdm(range(n_steps)):
        # Calculo las pendientes k_i
        slopes: List[np.ndarray] = []
        for i in range(shape_A[0]):
            term1 = f(
                    t0 + c[i]*sz,
                    y0 + sz*np.sum(
                            [A[i,j]*slopes[j] \
                                    for j in range(len(slopes)) if j!=i],
                            axis=0
                        )
                    )
            term2 = 0 if A[i,j]==0 else newton.compute()
            #slopes.append(slope)

        # Calculos siguiente valor y_{n+1}
        assert len(slopes) == len(b)
        y_n = y0 + sz*np.sum([b[k]*slopes[k] for k in range(len(slopes))])
        y0 = y_n

        solutions.append(y_n)


    return np.array(solutions)

@not_thread_safe
def compute_partitioned(
        params: PartitionedRungeKuttaParams,
        Aq: np.ndarray,
        bq: np.ndarray,
        Ap: np.ndarray,
        bp: np.ndarray,
        sz: float,
        n_steps: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Calcula n_steps pasos del método Runge-Kutta partido para
    valores iniciales params y coeficientes Aq,bq,Ap,bp.

    Argumentos:
        - params: valores iniciales
        - Aq,bq: coeficientes Ronge-Kutta para la variable q
        - Ap,bp: coeficientes Ronge-Kutta para la variable p
        - sz: step size
        - n_steps: número de pasos
    """

    shape_Aq = Aq.shape
    shape_bq = bq.shape
    shape_Ap = Ap.shape
    shape_bp = bp.shape
    assert shape_Aq[0] == shape_Aq[1], "La matriz debe ser cuadrada"
    assert shape_bq == (shape_Aq[0],), f"Aq.shape = {shape_Aq} vs. bq.shape = {shape_bq}"
    assert shape_Ap[0] == shape_Ap[1], "La matriz debe ser cuadrada"
    assert shape_bp == (shape_Ap[0],), f"Ap.shape = {shape_Ap} vs. bp.shape = {shape_bp}"
    assert sz > 0, f"h debe ser > 0, pero = {sz}"



    # Soluciones [q0,q1...qn_step] y [p0,p1...pn]
    qs = [params.q]
    ps = [params.p]

    for _ in tqdm(range(n_steps)):
        # Calculo las pendientes k_i y l_i
        k = []
        l = []
        for i in range(shape_Aq[0]): # s-step Ronge-Kutta
            q_term = qs[-1] + sz*np.sum([Aq[i,j]*k[j] for j in range(len(k))],axis=0)
            p_term = ps[-1] + sz*np.sum([Ap[i,j]*l[j] for j in range(len(l))],axis=0)
            k_i = params.f(q_term,p_term)
            l_i = params.g(q_term,p_term)
            k.append(k_i)
            l.append(l_i)

        # Calculos siguiente valor q_{n+1} y p_{n+1}
        assert len(k) == len(bq)
        assert len(l) == len(bp)

        q_n = qs[-1] + sz*np.sum([bq[j]*k[j] for j in range(len(k))],axis=0)
        p_n = ps[-1] + sz*np.sum([bp[j]*l[j] for j in range(len(l))],axis=0)

        qs.append(q_n)
        ps.append(p_n)

    return np.array(qs), np.array(ps)
