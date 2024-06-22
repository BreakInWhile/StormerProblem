import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils

from typing import Optional

logger = utils.create_logger("Stormer")

def Q(gamma:float, ro:float, z:float, r:Optional[float]) -> float:
    rad = np.sqrt(ro**2 + z*+2) if r is None else r
    return 1-(2*gamma/ro + ro/(rad**3))**2

def U(ro:float, z:float, r:Optional[float]) -> float:
    rad = np.sqrt(ro**2 + z*+2) if r is None else r
    return -(-1/ro + ro/(rad**3))**2

if __name__ == '__main__':

    GAMMA = -1.0
    ro_range = np.linspace(.3,3,500)
    q_vals = [U(ro,0,None) for ro in ro_range]

    plt.plot(ro_range,q_vals,'k-')

    plt.ylim([-0.2,0.0])
    plt.show()

    pass
