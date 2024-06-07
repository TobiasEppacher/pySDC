from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.visualization_tools import show_residual_across_simulation

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKuttaIMEX
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.projects.TobiasTests.Fenics_Heat_2D_custom import fenics_heat_2d_custom, MeshType, Equation

import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from dolfin import *

def get_residuals(stats):
    residuals = []
    for key, value in stats.items():
        if key.type == 'residual_post_sweep' or key.type == 'residual_post_step':
            residuals.append((key.time, key.iter, value))
    
    residuals = sorted(residuals, key=lambda x: (x[0], x[1]))
    return residuals
    

def problem_setup(t0, dt, nspace, maxiter, restol, num_nodes, mesh_type, equation):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = restol
    level_params['dt'] = dt
    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = num_nodes

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
    description['sweeper_class'] = imex_1st_order_mass
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
    
    # Plotting parameters
    plotting = False
    vmin = 0.0
    vmax = 1.0
    
    # Saves statistics about the execution to a file
    save_statistics = True
    
    # Basic parameters for the simulation
    # For more detailed parameters, adjust the problem_setup function
    t0 = 0.0
    Tend = 1.0
    timesteps = [2**i for i in range(2, 10)]
    nspace = 8
    maxiter = 5
    restol = 1e-15
    num_nodes = 1
    mesh_type = MeshType.RECTANGLE_2x1
    equation = Equation.POLY_N
    
    
    description, controller_params = problem_setup(
        t0=t0, 
        dt=(Tend - t0) / timesteps[0], 
        nspace=nspace, 
        maxiter=maxiter, 
        restol=restol, 
        num_nodes=num_nodes, 
        mesh_type=mesh_type, 
        equation=equation)
    
    print(f'\nRunning with mesh type {mesh_type.name} and equation {equation.name}')
    print(f'Mesh size parameter: {nspace}')
    print(f'Time interval: [{t0}, {Tend}]')
    print(f'Residual tolerance: {description["level_params"]["restol"]}')
    
    file_dir = f'pySDC/projects/TobiasTests/data/{mesh_type.name}/{equation.name}/order{num_nodes}'
    
    convergence_data = {}
    convergence_data['timesteps'] = []
    convergence_data['final_residual'] = []
    convergence_data['final_error'] = []
    
    for timestep_count in timesteps:
        dt = (Tend - t0) / timestep_count
        description['level_params']['dt'] = dt
        print(f'\nNumber of time steps={timestep_count}')
        print(f'Time step size dt={dt}')

        full_final_statistics = {}
        full_final_statistics['t_start'] = []
        full_final_statistics['t_end'] = []
        full_final_statistics['residual'] = []
        full_final_statistics['err'] = []
        full_final_statistics['iter_mean'] = []
        full_final_statistics['iter_range'] = []
        full_final_statistics['iter_max_index'] = []
        full_final_statistics['iter_min_index'] = []
        full_final_statistics['iter_std'] = []
        full_final_statistics['iter_var'] = []
        full_final_statistics['time_to_solution'] = []
        
        
        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
        P = controller.MS[0].levels[0].prob
        print(f'Fenics DoFs: {P.getDofCount()}')
        
        uinit = P.u_exact(t0)
        
        ex_solution_arr = [uinit]
        solution_arr = [uinit]
        
        if plotting:
            fig = plt.figure(figsize=(5, 10))
            fig.add_subplot(3,1,1)
            p = plot(uinit.values, title='Initial condition', vmin=vmin, vmax=vmax, cmap='coolwarm')
            fig.colorbar(p)
            
        curr_t = t0
        while curr_t < Tend:           
            uend, stats = controller.run(u0=solution_arr[-1], t0=curr_t, Tend=curr_t+dt)
            solution_arr.append(uend)
            
            uex = P.u_exact(curr_t+dt)
            ex_solution_arr.append(uex)
            
            err = abs(uex - uend) / abs(uex)
            
            iter_counts = get_sorted(stats, type='niter', sortby='time')
            niters = np.array([item[1] for item in iter_counts])
            
            full_final_statistics['t_start'].append(curr_t)
            full_final_statistics['t_end'].append(curr_t + dt)
            try:
                full_final_statistics['residual'].append(get_residuals(stats)[-1][2])
            except:
                print(get_residuals(stats))
            full_final_statistics['err'].append(err)
            full_final_statistics['iter_mean'].append(np.mean(niters))
            full_final_statistics['iter_range'].append(np.ptp(niters))
            full_final_statistics['iter_max_index'].append(int(np.argmax(niters)))
            full_final_statistics['iter_min_index'].append(int(np.argmin(niters)))
            full_final_statistics['iter_std'].append(float(np.std(niters)))
            full_final_statistics['iter_var'].append(float(np.var(niters)))
            full_final_statistics['time_to_solution'].append(get_sorted(stats, type='timing_run', sortby='time')[0][1])
            
            curr_t += dt
            
        convergence_data['timesteps'].append(timestep_count)
        convergence_data['final_residual'].append(full_final_statistics['residual'][-1])
        convergence_data['final_error'].append(full_final_statistics['err'][-1])          
            
        if plotting:
            fig.add_subplot(3,1,2)
            p = plot(uend.values, title='Computed solution', vmin=vmin, vmax=vmax, cmap='coolwarm')
            fig.colorbar(p)
            fig.add_subplot(3,1,3)
            p = plot(uex.values, title='Exact solution', vmin=vmin, vmax=vmax, cmap='coolwarm')
            fig.colorbar(p)
            plt.show()
            
        if save_statistics:
            # Save full execution statistics to seperate file for each time step count / time step size
            # This includes data like residual, error, time to solution etc. for each time step
            Path(f'{file_dir}/statistics').mkdir(exist_ok=True, parents=True)
            fname = f'{file_dir}/statistics/heat_2d_{dt}step_size.csv'
            f = open(fname, 'w')
            
            df = pd.DataFrame.from_dict(full_final_statistics)
            df.to_csv(f, index=False)
        
        
    # Save convergence data (final residual and error for each time step count)
    # to file for further analysis
    if save_statistics:       
        Path(file_dir).mkdir(exist_ok=True, parents=True)
        fname = f'{file_dir}/convergence_data.csv'
        f = open(fname, 'w')
        
        df = pd.DataFrame.from_dict(convergence_data)
        df.to_csv(f, index=False)
    
    return 0




if __name__ == '__main__':
    main()