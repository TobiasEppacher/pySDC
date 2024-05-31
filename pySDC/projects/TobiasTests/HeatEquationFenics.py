from pySDC.helpers.stats_helper import get_sorted

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.projects.TobiasTests.Fenics_Heat_2D import fenics_heat_2d
from pySDC.projects.TobiasTests.Fenics_Heat_2D_2 import fenics_heat_2d_2
from pySDC.projects.TobiasTests.Fenics_Heat_2D_3 import fenics_heat_2d_3
from pySDC.projects.TobiasTests.Fenics_Heat_2D_4 import fenics_heat_2d_4
from pySDC.projects.TobiasTests.Fenics_Heat_2D_custom import fenics_heat_2d_custom, MeshType, Equation
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat

import matplotlib.pyplot as plt

from dolfin import *

def problem_setup(t0, nspace, mesh_type, equation):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5e-8
    level_params['dt'] = 0.1
    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 30

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = dict()
    problem_params['mesh_type'] = mesh_type
    problem_params['equation'] = equation
    problem_params['t0'] = t0
    problem_params['ny'] = nspace  # number of degrees of freedom for each level
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    
    base_transfer_params = dict()
    base_transfer_params['finter'] = True

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = fenics_heat_2d_custom
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    #description['space_transfer_class'] = mesh_to_mesh_fenics
    #description['base_transfer_params'] = base_transfer_params

    return description, controller_params


def main():
    
    # Basic parameters for the simulation
    # For more detailed parameters, adjust the problem_setup function
    t0 = 0.0
    Tend = 1.0
    nspace = 8
    mesh_type = MeshType.UNIT_SQUARE
    equation = Equation.MIXED
    
    # Plotting parameters
    
    vmin = 0.0
    vmax = 5.0
    
    description, controller_params = problem_setup(t0=t0, nspace=nspace, mesh_type=mesh_type, equation=equation)
    
    print(description['level_params']['restol'])
    
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    
    fig = plt.figure(figsize=(5, 10))
    fig.add_subplot(3,1,1)
    p = plot(uinit.values, title='Initial condition', vmin=vmin, vmax=vmax, cmap='coolwarm')
    fig.colorbar(p)
    
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    
    uex = P.u_exact(Tend)
    err = abs(uex - uend) / abs(uex)
    out = f' -- error at time {Tend}: {err}'
    print(out)
    
    iter_counts = get_sorted(stats, type='niter', sortby='time')
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % (int(np.argmax(niters)), int(np.argmin(niters)))
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    print(out)

    timing = get_sorted(stats, type='timing_run', sortby='time')
    out = f'Time to solution: {timing[0][1]:6.4f} sec.'
    print(out)
    
    fig.add_subplot(3,1,2)
    p = plot(uend.values, title='Computed solution', vmin=vmin, vmax=vmax, cmap='coolwarm')
    fig.colorbar(p)
    fig.add_subplot(3,1,3)
    p = plot(uex.values, title='Exact solution', vmin=vmin, vmax=vmax, cmap='coolwarm')
    fig.colorbar(p)
    plt.show()
    
    
    return 0

if __name__ == '__main__':
    main()