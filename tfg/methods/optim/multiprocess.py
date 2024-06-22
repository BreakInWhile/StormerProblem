import logging,sys
import numpy as np
from tfg.methods.optim.SV import StormerVerletMethod
from tfg.methods.optim.conservative import ConservativeMethod
from tfg.methods.optim.gauss import GaussMethod
from tfg.methods.optim.RN_symmetrized import RNMethod 
from tfg.helpers import hamiltonian
from tfg.chaos import poincare

logging.disable(sys.maxsize)

def worker_function(initial_values,ts,its,method = StormerVerletMethod()):
    if np.isnan(hamiltonian(initial_values)):
            print("Prueba con otro")
            return None
    result = method.simulate(initial_values,ts,its,True)
    return result

def poincare_worker(initial_values,fixed_var,fixed_val,ts,its,method = StormerVerletMethod(),interpolation='legendre'):
    if np.isnan(hamiltonian(initial_values)):
            print("Prueba con otro")
            return None
    result =poincare.application_reduced(initial_values,fixed_var,fixed_val,time_step=ts,iterations=its,method=method,interpolation=interpolation)
    return result

def poincare3D_worker(initial_values,fixed_var,fixed_val,ts,its,method = StormerVerletMethod(),interpolation='legendre'):
    #if np.isnan(hamiltonian(initial_values)):
    #        print("Prueba con otro")
    #        return None
    result =poincare.application_general(initial_values,fixed_var,fixed_val,time_step=ts,iterations=its,method=method,interpolation=interpolation)
    return result

      