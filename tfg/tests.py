import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a custom logger
logger = logging.getLogger('Stormer')
logger.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(levelname)s - %(message)s')

# Create a handler and set the formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

#C = 1.3
MASA_ELECTRON = 9.1e-28
ALPHA_ELECTRON = -5.588e6
C_ELECTRON = MASA_ELECTRON * ALPHA_ELECTRON

MASA_PROTON = 1.7e-24
ALPHA_PROTON = 3.037e3
#ALPHA_PROTON = 7.88e23
C_PROTON = MASA_PROTON * ALPHA_PROTON

x = np.arange(.01,3,.01)

def fun_aux(x: float,c, m: float, a: float) -> float:
    return  c/(2*math.pow(m,2)) * math.pow((c/x - (m*a*x)/math.pow(x,3)),2)

#def fun(x_list):
#    return np.array([fun_aux(xi,MASA_PROTON,ALPHA_PROTON) for xi in x_list])

def V_Z0(x,c, m, a):
    return (math.pow(c,2)/(2*math.pow(m,2)*math.pow(x,2))) \
            - (a*c/(m*math.pow(x,3))) \
            + (math.pow(a,2)/(2*math.pow(x,4)))

def prueba(x, c, m, a):
    return c**2/(2*m**2*x**2) \
            - (a*c)/(m*x**3) \
            + a**2/(2*x**4)

def fun_V_Z0(x_list,c,m,a):
    return np.array([prueba(xi,c,m,a) for xi in x_list])

def adimensional(x):
    return 1/2*(1/x - 1/x**2)**2

def pozo_adimensional(x_list):
    return np.array([adimensional(xi) for xi in x_list])


def f_potencial(ro,z,lim):
    return np.min(1/2 * (1/ro - ro/(np.sqrt(ro**2 + z**2))**3)**2,lim)

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def custom_function(x, y):
    mask = (x != 0) & (y != 0)

    inner_expr = np.abs(1/x - x/(np.sqrt(x**2 + y**2)**3))

    z = np.empty_like(x)
    z[mask] = 0.5 * inner_expr[mask]**2
    z[~mask] = np.nan  # Handle division by zero
    #z[z > 1/32] = 1/32
    return z

def potencial_3d():
    logger.debug("Dibujando potencial 3D")

    ro = np.arange(0, 3, 0.001)
    z = np.arange(-1,1,0.001)

    logger.debug(f"---> Dimensión ro: {ro.shape}")
    logger.debug(f"---> Dimensión z: {z.shape}")

    x,y = np.meshgrid(ro,z)
    logger.debug(f"---> Forma X: {x.shape}")
    logger.debug(f"---> Forma Y: {y.shape}")

    logger.debug("---> Calculando potencial...")
    potencial = custom_function(x,y)
    potencial[potencial > 2/32] = 2/32
    #potencial = f_potencial(X,Y)

    logger.debug("---> Dibujando...")
    
    fig = plt.figure(figsize=(10,6))
    ax1: Axes3D = fig.add_subplot(121, projection='3d') #type: ignore

    # Plot surface
    ax1.plot_surface(x,y, potencial,cmap='binary')

    # Set labels and title
    ax1.grid(False)
    ax1.set_xlabel('ro')
    ax1.xaxis.set_ticks([0])
    ax1.set_ylabel('Z')
    ax1.yaxis.set_ticks([0])
    #ax1.set_zlabel('V')
    ax1.zaxis.set_ticks([])
    ax1.view_init(30, -35)
    ax1.set_title('Potencial ')

    ax2: Axes3D = fig.add_subplot(122, projection='3d') #type: ignore

    # Plot surface
    ax2.plot_surface(x,y, potencial,cmap='binary')

    # Set labels and title
    ax2.grid(False)
    ax2.set_xlabel('ro')
    ax2.xaxis.set_ticks([0])
    ax2.set_ylabel('Z')
    ax2.yaxis.set_ticks([0])
    #ax2.set_zlabel('V')
    ax2.zaxis.set_ticks([])
    ax2.view_init(90,-90)
    ax2.set_title('Potencial visto desde arriba')
    # Show plot
    plt.show()


class ParticleConfig():

    def __init__(self, m, c, a):
        self.m = m
        self.c = c
        self.a = a

class StormerVerlet():

    """
    Clase que implementa el método Stormer-Verlet
    """

    def __init__(
            self,
            num_steps: int,
            step_len: float,
            ro_0: float,
            dro_0: float,
            phi_0: float,
            config: ParticleConfig
            ) -> None:

        self._step_len = step_len
        self._num_steps = num_steps
        self.ro_0 = ro_0
        self.dro_0 = dro_0
        self.phi_0 = phi_0
        self.m = config.m
        self.c = config.c
        self.a = config.a

    def get_trajectory(self):

        ro_reg = [self.ro_0]
        dro_reg = [self.dro_0]
        phi_reg = [self.phi_0]

        for _ in range(self._num_steps):
            #TODO: Mirar complejidad acceso al ultimo elemento de una lista
            ro_n = self._next_ro(ro_reg[-1],dro_reg[-1]) 
            dro_n = self._next_dro(ro_reg[-1],ro_n,dro_reg[-1])
            phi_n = self._next_phi(phi_reg[-1],ro_reg[-1])

            ro_reg.append(ro_n)
            dro_reg.append(dro_n)
            phi_reg.append(phi_n)

        return np.array(ro_reg), np.array(phi_reg)
    
    def _f(self,ro):
        return self.c**2/(self.m**2 * ro**3) \
                - (3*self.c*self.a)/(self.m * ro**4) \
                + (2 * self.a**2)/ro**5
    
    def _next_ro(self, ro_n, dro_n):
        return ro_n + self._step_len*dro_n \
                + (self._step_len**2 / 2) * self._f(ro_n)

    def _next_dro(self, ro_n, ro_nn, dro_n):
        return dro_n + (self._step_len / 2)*self._f(ro_n) \
                + (self._step_len / 2)*self._f(ro_nn)

    def _next_phi(self, phi_n, ro_n):
        return phi_n + \
                self._step_len*(self.c /(self.m * ro_n**2) - self.a/ro_n**3)

    


if __name__=='__main__':

    fig, axs = plt.subplots(1,2,figsize=(10,6))
    pa = pozo_adimensional(x)
    axs[0].plot(x, pa)
    axs[0].plot(x, [1/32]*len(x),'r--')
    axs[0].set_ylim([0,1/16])
    axs[0].set_title("Electrón")

    axs[1].plot(x,pa/1838.02)
    axs[1].plot(x, [1.7e-5]*len(x),'r--')
    axs[1].set_ylim([0,1/(16*1838.02)])
    axs[1].set_title("Protón")

    fig.suptitle("Pozos de potencial")
    plt.show()

    #r1 = 2.282
    #r2 = 2.755
    #r3 = 5.510
    # Define angles for generating points on the circle
    #theta = np.linspace(0, 2*np.pi, 100)

    # Calculate (x, y) coordinates for each circle
    #x1 = r1 * np.cos(theta)
    #y1 = r1 * np.sin(theta)
#
    #x2 = r2 * np.cos(theta)
    #y2 = r2 * np.sin(theta)

    #x3 = r3 * np.cos(theta)
    #y3 = r3 * np.sin(theta)

    #c = MASA_PROTON* (3.0**2) * 10.0 + (MASA_PROTON*ALPHA_PROTON)/3
#
    #conf_proton = ParticleConfig(MASA_PROTON,c,ALPHA_PROTON)
    #sv_proton = StormerVerlet(10000,0.0001,3.0,10.0,0,conf_proton)
    #ro, phi = sv_proton.get_trajectory()

    #x = ro * np.cos(phi)
    #y = ro * np.sin(phi)

    #logger.debug(f"{x[0]},{y[0]}")
    #plt.plot(np.arange(0,len(ro)), ro)

    #plt.figure(figsize=(6,6))

    """
    plt.plot(x1,y1,'k')
    plt.plot(x2,y2,'k')
    plt.plot(x3,y3,'k')
    """

    """
    conf_proton = ParticleConfig(MASA_PROTON,c,ALPHA_PROTON)
    sv_proton = StormerVerlet(10000,0.0001,3.0,10.0,0,conf_proton)
    ro, phi = sv_proton.get_trajectory()

    x = ro * np.cos(phi)
    y = ro * np.sin(phi)
    
    plt.plot(x,y)


    sv_proton = StormerVerlet(10000,0.0001,3.0,80.0,3*np.pi/2,conf_proton)
    ro, phi = sv_proton.get_trajectory()

    x = ro * np.cos(phi)
    y = ro * np.sin(phi)
    
    plt.plot(x,y)
    """

    #s = 0
    #jar = np.linspace(0,(r3-r2)/2,50)
    #bar = np.linspace(0,40,50)
    #for i in range(len(jar)):
        #sv_proton = StormerVerlet(10000,0.0001,r2 + jar[i] + (r3-r2)/2,10,0,conf_proton)
        #ro, phi = sv_proton.get_trajectory()
#
        #x = ro * np.cos(phi)
        #y = ro * np.sin(phi)
        
        #sp_j = plt.scatter(x,y,s=2,c=np.sqrt(x**2 + y**2),cmap='plasma')
#
        #sv_proton = StormerVerlet(10000,0.0001,3.0,0+bar[i],0,conf_proton)
        #ro, phi = sv_proton.get_trajectory()

        #x = ro * np.cos(phi)
        #y = ro * np.sin(phi)
        
        #sp_b = plt.scatter(x,y,s=2,c=np.sqrt(x**2 + y**2),cmap='plasma')
        #plt.pcolormesh(x,y,np.sqrt(x**2+y**2),cmap="plasma")
        #plt.plot(ro,phi)
        #plt.plot(phi)
        #plt.plot(ro)

        #plt.show()

        #plt.xlim([-r3-.1,r3+.1])
        #plt.ylim([-r3-.1,r3+.1])
        #plt.axis("off")
        #plt.show()

        #REMOVE
        #plt.savefig(f"./gif/{s}.png", bbox_inches='tight')
        #jaja.remove()
        #sp_j.remove()
        #sp_b.remove()
        #s+=1






