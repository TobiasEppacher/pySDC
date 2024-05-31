from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.visualization_tools import show_residual_across_simulation

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.projects.TobiasTests.Fenics_Heat_2D_custom import fenics_heat_2d_custom, MeshType, Equation

import matplotlib
import matplotlib.pyplot as plt

from dolfin import *

import logging
from pathlib import Path

def get_residuals(stats):
    residuals = []
    for key, value in stats.items():
        if key.type == 'residual_post_sweep':
            residuals.append((key.time, key.iter, value))
    
    residuals = sorted(residuals, key=lambda x: (x[0], x[1]))
    return residuals
    

def problem_setup(t0, dt, nspace, restol, mesh_type, equation):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = restol
    level_params['dt'] = dt
    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

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
    problem_params['order'] = [2]

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
    # Set up matplotlib to use Qt5Agg backend (for interactive plots)
    matplotlib.use('Qt5Agg')
    
    # Suppress missing font warning
    logging.getLogger('matplotlib.font_manager').disabled = True
    
    plotting = False
    print_residuals = True
    
    # Basic parameters for the simulation
    # For more detailed parameters, adjust the problem_setup function
    t0 = 0.0
    Tend = 1.0
    dt_arr = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    nspace = 8
    restol = 1e-15
    mesh_type = MeshType.UNIT_SQUARE
    equation = Equation.POLY_N
    
    # Plotting parameters
    vmin = 1.0
    vmax = 3.2
    
    description, controller_params = problem_setup(t0=t0, dt=dt_arr[0], nspace=nspace, restol=restol, mesh_type=mesh_type, equation=equation)
    
    print(f'\nRunning with mesh type {mesh_type.name} and equation {equation.name}')
    print(f'Mesh size parameter: {nspace}')
    print(f'Time interval: [{t0}, {Tend}]')
    print(f'Residual tolerance: {description["level_params"]["restol"]}')
    for dt in dt_arr:
        description['level_params']['dt'] = dt
        print(f'\nTime step size dt={dt}')
        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)
        
        if plotting:
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
        
        if plotting:
            fig.add_subplot(3,1,2)
            p = plot(uend.values, title='Computed solution', vmin=vmin, vmax=vmax, cmap='coolwarm')
            fig.colorbar(p)
            fig.add_subplot(3,1,3)
            p = plot(uex.values, title='Exact solution', vmin=vmin, vmax=vmax, cmap='coolwarm')
            fig.colorbar(p)
            plt.show()
           
        if print_residuals:
            # Save residuals to file
            Path(f'data/convergence/{mesh_type.name}/{equation.name}').mkdir(exist_ok=True, parents=True)
            fname = f'data/convergence/{mesh_type.name}/{equation.name}/heat_2d_{dt}step_size.txt'
            f = open(fname, 'w')
            residuals = get_residuals(stats)
            for item in residuals:
                f.write('t: %.8f | iteration: %3i | residual: %.13f\n' % item)
                
            # Save residual plot
            fname = f'data/convergence/{mesh_type.name}/{equation.name}/heat_2d_{dt}step_size.png'
            plt.figure()
            show_residual_across_simulation(stats, fname)
    
    return 0

if __name__ == '__main__':
    main()