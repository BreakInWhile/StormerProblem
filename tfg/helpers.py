import numpy as np

def U(u,v):
    rad = np.sqrt(u**2.0 + v**2.0)
    return 1.0/2.0 * (1.0/u - u/(rad**3.0))**2.0

def hamiltonian(y):
    return 1.0/2.0*(y[2]**2.0 + y[3]**2.0) + U(y[0],y[1])

def discrete_hamiltionian(sols,time_step,cons=False):
    hams = []
    for i in range(len(sols)-1):
        cin = 1.0/2.0 * (((sols[i+1][0]-sols[i][0])/time_step)**2 + ((sols[i+1][1]-sols[i][1])/time_step)**2 )
        pot = 1.0/2.0 * (U(sols[i+1][0],sols[i+1][1]) + U(sols[i][0],sols[i][1]))
        hams.append(cin+pot)
    return hams
