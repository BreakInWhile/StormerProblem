import numpy as np
import matplotlib.pyplot as plt
import utils.utils_plot as uplot
import utils.utils as utils
from methods import newton,rungekutta as rk
from tqdm import tqdm

logger = utils.create_logger("main")

def f(u,v):
    return np.array([u*(v-2),v*(1-u)])

def I(u,v):
    return np.log(u) - u + 2*np.log(v) - v

def lotka_volterra():
    _,axs = plt.subplots(1,2,figsize=(10,6))

    uplot.plot_vector_field_2D(axs[0],f,np.linspace(.1,3,20),np.linspace(.1,5,35))

    uplot.plot_level_lines_2D(axs[0],I,10,np.linspace(.1,3,100),np.linspace(.1,5,100),color='r')
    uplot.plot_level_lines_2D(axs[1],I,10,np.linspace(.1,3,100),np.linspace(.1,5,100))

    #plt.gca().set_aspect(5/5)
    axs[0].set_ylim([0,5])
    axs[1].set_ylim([0,5])
    plt.show()

class StormerProblem():

    def __init__(self):
        pass

    def U(self,u,v):
        rad = np.sqrt(u**2 + v**2)
        return 1/2 * (1/u - u/(rad**3))**2

    def plot_level_lines(self,n_levels,u_range = None,v_range = None):
        _,axs = plt.subplots(1,1,figsize=(10,6))

        uplot.plot_level_lines_2D(axs,self.U,n_levels,u_range,v_range)

        plt.show()

    def f(self,q):
        rho,z = q
        rad = np.sqrt(rho**2 + z**2)
        first_term = (-1/rho + rho/(rad**3))
        d_ro = first_term \
                * (-1/(rho**2) + 3*(rho**2)/(rad**5) - 1/(rad**3))
        d_z = first_term * 3*rho*z/(rad**5)

        return np.array([d_ro,d_z])

    def simulate(self,q_0,p_0,n_steps,h,method = "Stormer-Verlet"):

        cin = 1/2 * (p_0[0]**2 + p_0[1]**2)
        pot = self.U(q_0[0],q_0[1])

        logger.debug(f"K: {cin}")
        logger.debug(f"V: {pot}")
        logger.debug(f"Hamiltoniano: {cin+pot}\n")

        q = [q_0]
        p = [p_0]
        ham = [cin+pot]

        logger.debug(f"Method: {method}")

        if method == "Runge-Kutta":
            q,p = runge_kutta(q_0,p_0,h,n_steps)
        else:
            #for _ in tqdm(range(n_steps)):
            for _ in range(n_steps):
                if method == "Stormer-Verlet":
                    ff = self.f(q[-1])
                    q_n = q[-1] + h*p[-1] + ((h**2)/2)*ff
                    p_n = p[-1] + (h/2)*ff + (h/2)*self.f(q_n)
                else:
                    q_n = q[-1] + h*p[-1]
                    p_n = p[-1] + h*self.f(q[-1])

                q.append(q_n)
                p.append(p_n)

                cin = 1/2 * (p_n[0]**2 + p_n[1]**2)
                pot = self.U(q_n[0],q_n[1])

                ham.append(cin+pot)


        cin = [1/2*(p[i][0]**2 + p[i][1]**2) for i in range(len(p))] 
        pot = [self.U(q[i][0],q[i][1]) for i in range(len(q))]

        ham = [cin[i] + pot[i] for i in range(len(cin))]
        return np.array(q), ham

    def plot_stormer_reduced(self,axs,q_0,p_0,n_steps,h,method = "Stormer-Verlet",color = "k"):
        qs, ham = self.simulate(q_0,p_0,n_steps,h,method)

        axs[0].plot(qs[:,0], qs[:,1],color = color,label=method) # type : ignore
        axs[0].scatter(q_0[0],q_0[1],color='r',s=5)

        axs[1].plot(np.linspace(0,len(ham),len(ham)), ham, color = color, label = method)
        axs[1].plot(np.linspace(0,len(ham),len(ham)), [ham[0]]*len(ham), color = "r", label = "Real")
        
    def _get_p(self,H_0,p_otro,q_otro):
        return np.sqrt(2*H_0 - p_otro**2 - (1/q_otro - 1/(q_otro**2))**2)

    def poincare_cuts(self, H_0, rho_0, p_rho_0,  n_points, h):
        """
        1. Fijo z = 0
        
        Luego estoy en el plano rho-p_rho y P_i = (rho,p_rho)

        2. Elijo P_0 = (rho_0,p_rho_0) cualquiera
        3. Saco p_z_0 del Hamiltoniano
        4. Simulo hasta z = 0 otra vez con p_z > 0
        5. Obtengo P_1
        6. Repito

        """

        z = 0
        rho = rho_0
        p_rho = p_rho_0
        P = np.array([rho,p_rho])
        p_z = self._get_p(H_0,p_rho,rho)

        logger.debug(f"""
                     - rho = {rho}
                     - z = {z}
                     - p_rho = {p_rho}
                     - p_z = {p_z}

                     - P = {P}
                     """)

        ppoints = [P]
        for _ in tqdm(range(n_points)):
            q = [np.array([rho,z])]
            p = [np.array([p_rho,p_z])]
            while True:
                ff = self.f(q[-1])
                q_n = q[-1] + h*p[-1] + ((h**2)/2)*ff
                p_n = p[-1] + (h/2)*ff + (h/2)*self.f(q_n)

                q.append(q_n)
                p.append(p_n)

                #logger.debug(f"Next q = {q_n}; Next p = {p_n}")

                if q_n[1]*q[-2][1] < 0 and p_n[1] > 0:
                    rho = q_n[0]
                    p_rho = p_n[0]
                    ppoints.append(np.array([rho,p_rho]))
                    p_z = self._get_p(H_0,p_rho,rho)
                    break

        return np.array(ppoints)

    
def f1(t,y):
    return t**2 + y**2


def F(q,p):
    return p

def G(q,p):
    rho,z = q
    rad = np.sqrt(rho**2 + z**2)
    first_term = (-1/rho + rho/(rad**3))
    d_ro = first_term \
            * (-1/(rho**2) + 3*(rho**2)/(rad**5) - 1/(rad**3))
    d_z = first_term * 3*rho*z/(rad**5)

    return np.array([d_ro,d_z])

def runge_kutta(q0,p0,sz,n_steps):
    #params = rk.RungeKuttaParams(0,.46,.001) #default
    #A = np.array([
    #    [0,0,0,0],
    #    [1/2,0,0,0],
    #    [0,1/2,0,0],
    #    [0,0,1,0]
    #    ])
    #b = np.array([1/6,1/3,1/3,1/6])

    params = rk.PartitionedRungeKuttaParams(q0,p0,F,G)
    #Aq = np.array([
    #    [0,0],
    #    [1/2,1/2]
    #    ])
    #bq = np.array([1/2,1/2])
    #Ap = np.array([
    #    [1/2,0],
    #    [1/2,0]
    #    ])
    #bp = np.array([1/2,1/2])

    #Aq = np.array([
        #[0,0,0],
        #[5/24,1/3,-1/24],
        #[1/6,2/3,1/6]
        #])
    #bq = np.array([1/6,2/3,1/6])
    #Ap = np.array([
        #[1/6,-1/6,0],
        #[1/6,1/3,0],
        #[1/6,5/6,0]
        #])
    #bp = np.array([1/6,2/3,1/6])

    Aq = np.array([
        [5/36,2/9 - np.sqrt(15)/15,5/36 - np.sqrt(15)/30],
        [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
        [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]
        ])
    bq = np.array([5/18,4/9,5/18])
    Ap = np.copy(Aq)
    bp = np.copy(bq)

    sols, pps = rk.compute_partitioned(params,Aq,bq,Ap,bp,sz,n_steps)
    return sols, pps

def newton_test():
    def f(p):
        return np.array([(p[0]-1)**2,p[1]**2])
    J = np.array([
        [lambda p: 2*p[0], lambda _: 0],
        [lambda _: 0, lambda p: 2*p[1]]
        ])
    p0 = np.array([2,2])
    THRESHOLD = .0001

    print(newton.compute(p0,f,J,THRESHOLD))

def stormer_test():
    ro_range = np.linspace(.001,2,1000)
    z_range = np.linspace(-.5,.5,1000)

    #q_0 = np.array([1,.4])
    # 1,.25,-.0004,0
    #q_0 = np.array([1.6,.1])
    #p_0 = np.array([-1/200,-.00008])
    q_0 = np.array([1,.25])
    p_0 = np.array([-.0004,0])
    h = .002
    N = 12000

    
    sp = StormerProblem()
    
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot2grid((2,5), (0,0), rowspan=2,colspan=3)
    ax2 = plt.subplot2grid((2,5), (0,3), colspan=2)
    #ax3 = plt.subplot2grid((2,5), (1,0), colspan=3)
    ax4 = plt.subplot2grid((2,5), (1,3), colspan=2)
    uplot.plot_level_lines_2D(ax1,sp.U,3,[0,1/32],ro_range,z_range)
    #uplot.plot_level_lines_2D(ax3,sp.U,3,[0,1/32],ro_range,z_range)
    #sp.plot_stormer_reduced([ax1,ax2],q_0,p_0,N,h,"Euler",color="k")
    #sp.plot_stormer_reduced([ax1,ax4],q_0,p_0,N,h,"Stormer-Verlet",color="b")
    sp.plot_stormer_reduced([ax1,ax4],q_0,p_0,N,h,"Runge-Kutta",color="b")
    
    #ppoints = sp.poincare_cuts(1/64,1,-1/12,100,.001)
    #logger.debug(f"Puntos shape: {ppoints}")

    #axs[1].scatter(ppoints[:,0],ppoints[:,1],color='r',s=5)



    ax1.legend()
    ax1.title.set_text('Trayectorias')
    ax2.legend()
    ax2.title.set_text('Hamiltoniano real vs. Hamiltoniano en cada iteración')
    #ax3.legend()
    ax4.legend()
    title = fr"{N} iteraciones. $h={h},\, q(0)=[{q_0[0]},{q_0[1]}], \, p(0)=[{p_0[0]},{p_0[1]}]$"
    plt.suptitle(title)
    fig.tight_layout()
    plt.show()
    #sp.plot_level_lines(3,ro_range,z_range)



if __name__ == "__main__":

    #lotka_volterra()
    #runge_kutta()

    #newton_test()

    stormer_test()

