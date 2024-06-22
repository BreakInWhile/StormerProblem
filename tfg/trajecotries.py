import numpy as np
import matplotlib.pyplot as plt

STEP_PHI = 0.01 # for q_r on Rz-plane

def plot_regions(
        gamma,
        k_vals,
    ):

    plt.figure(figsize=(10,6))

    for k in k_vals:
        c1,c2 = regions(gamma,k)

        plt.plot(c1[0],c1[1],'k-')
        plt.plot(c1[0],-c1[1],'k-')

        plt.plot(c2[0],c2[1],'r-')
        plt.plot(c2[0],-c2[1],'r-')



def regions(gamma,k):

    r1_l = []
    z1_l = []

    r2_l = []
    z2_l = []

    for phi in np.arange(0,np.pi/2,STEP_PHI):
        r1 = (gamma + np.sqrt(gamma**2 + k*np.cos(phi)**3))/(k*np.cos(phi))
        r2 = (gamma - np.sqrt(gamma**2 + k*np.cos(phi)**3))/(k*np.cos(phi))

        ro1 = r1*np.cos(phi)
        z1 = ro1*np.tan(phi)

        ro2 = r2*np.cos(phi)
        z2 = ro2*np.tan(phi)

        r1_l.append(ro1)
        z1_l.append(z1)

        r2_l.append(ro2)
        z2_l.append(z2)

    return [np.array(r1_l),np.array(z1_l)], \
            [np.array(r2_l),np.array(z2_l)]

class StormerVerlet():

    def __init__(self):
        self.name = "Stormer-Verlet"

    def _V_ro(self,ro,z):
        r = np.sqrt(ro**2 + z**2)
        return (-1/ro + ro/(r**3))*(-1/(ro**2) + 3*(ro**2)/(r**5)) \
                - 1/(r**3)

    def _V_z(self,ro,z):
        r = np.sqrt(ro**2 + z**2)
        # TODO: Reciclar
        return (-1/ro + ro/(r**3))*(3*ro*z/(r**5))

    def _next_ro(self,h,ro,z,dro):
        return ro + h*dro + self._V_ro(ro,z)*(h**2)/2

    def _next_dro(self,h,ro,ro_n,z,z_n,dro):
        return dro + self._V_ro(ro,z)*h/2 + self._V_ro(ro_n,z_n)*h/2

    def _next_z(self,h,ro,z,dz):
        return z + h*dz + self._V_z(ro,z)*(h**2)/2

    def _next_dz(self,h,ro,ro_n,z,z_n,dz):
        return dz + self._V_z(ro,z)*h/2 + self._V_z(ro_n,z_n)*h/2

    def get_orbit(self,ro,z,dro,dz,n_steps,sz):
        ro_l = [ro]
        z_l = [z]
        dro_l = [dro]
        dz_l = [dz]

        for _ in range(n_steps):
            ro_n = self._next_ro(sz,ro_l[-1],z_l[-1],dro_l[-1])
            z_n = self._next_z(sz,ro_l[-1],z_l[-1],dz_l[-1])

            dro_n = self._next_dro(sz,ro_l[-1],ro_n,z_l[-1],z_n,dro_l[-1])
            dz_n = self._next_dro(sz,ro_l[-1],ro_n,z_l[-1],z_n,dz_l[-1])

            ro_l.append(ro_n)
            z_l.append(z_n)
            dro_l.append(dro_n)
            dz_l.append(dz_n)

        return ro_l,z_l




if __name__=="__main__":

    GAMMA = -.9

    #k_vals = np.linspace(-1,1,50)
    k_vals = [-1,0.00001,1]
    plot_regions(GAMMA,k_vals)

    plt.xlim([0,3])
    plt.ylim([-1.5,1.5])

    
    """
    sv = StormerVerlet()
    ro,z = sv.get_orbit(.6,.1,.1,.0,10000,.001)
    plt.plot(ro,z)
    """
    plt.show()

