import numpy as np
import tfg.utils.utils as utils
import matplotlib.pyplot as plt

logger = utils.create_logger("TestFuncionesPlot") # type: ignore

def plot_vector_field_2D(axs,f,u_range = None, v_range = None, color = None):

    u = np.linspace(-2,2,20) if u_range is None else u_range
    v = np.linspace(-2,2,20) if v_range is None else v_range

    u_g, v_g = np.meshgrid(u,v)

    x,y = f(u_g,v_g)
    
    c = 'k' if color is None else color

    axs.quiver(u_g,v_g,x,y,scale=30,color=c)

def plot_level_lines_2D(axs,f,n_levels,levels=None,u_range = None, v_range = None, color = None):

    u = np.linspace(-2,2,20) if u_range is None else u_range
    v = np.linspace(-2,2,20) if v_range is None else v_range

    u_g, v_g = np.meshgrid(u, v)

    z = f(u_g,v_g)

    ls = np.linspace(z.min(),z.max(),n_levels) if levels is None else levels

    c = 'k' if color is None else color

    axs.contour(u_g, v_g, z, levels=ls, colors=c) 
    # axs.clabel(conts, conts.levels, manual=False, fmt=fmt, fontsize=10)


def f(u,v):
    return np.log(u) - u + 2*np.log(v) - v

if __name__ == "__main__":

    u = np.linspace(.1,4,100)
    v = np.linspace(.1,6,100)

    u_g, v_g = np.meshgrid(u, v)

    z = f(u_g,v_g)

    logger.debug(f"Forma de Z : {z.shape}")
    logger.debug(f"Media de Z : {z.mean()}")
    logger.debug(f"Mímina de Z : {z.min()}")
    logger.debug(f"Max de Z : {z.max()}")

    plt.contour(u_g,v_g,z,levels=[-2.92,-2.12,-1.64],colors='k')
    plt.show()
