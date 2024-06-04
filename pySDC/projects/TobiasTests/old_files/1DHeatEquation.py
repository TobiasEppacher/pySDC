from pathlib import Path
import numpy as np

from collections import namedtuple

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

ID = namedtuple('ID', 'timestep')

def problem_setup(dt, nspace):
    # initialize level parameters
    level_params = {}
    level_params['restol'] = 5e-10
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['nvars'] = nspace  # number of degrees of freedom for each level
    problem_params['bc'] = 'dirichlet'  # boundary conditions

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 50
    step_params['errtol'] = 1e-05

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['all_to_done'] = True  # can ask the controller to keep iterating all steps until the end
    controller_params['predict_type'] = 'pfasst_burnin'  # activate iteration estimator

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heatNd_unforced
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh
    description['space_transfer_params'] = space_transfer_params

    return description, controller_params


def main():
    
    t0 = 0.0
    tend = 1.0
    ntime = 128
    nspace = [(65,65), (33,33)]
    
    dt = (tend - t0) / ntime
    
    
    
    description, controller_params = problem_setup(dt, nspace)
    
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    
    
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    
    result = dict()
    result[ID(0)] = uinit
    
    timestep = 1
    t = t0
    while t <= tend:
        print("Execute time %f" % t)
        ucurr, stats = controller.run(u0=result[ID(timestep-1)], t0=t, Tend=t+dt)
        result[ID(timestep)] = ucurr
        timestep += 1
        t += dt
    
    for entry in result:
        Path("data").mkdir(parents=True, exist_ok=True)
        f = open('data/tobias_test_output/tobias_test_%04i.csv' % entry, 'w')
        f.write('x,y,z,value\n')
        
        data = result[entry]
        for i in range(len(data)):
            for j in range(len(data[i])):
                out = "%f,%f,%f,%f\n" % (i, j, 0, data[i][j])
                f.write(out)
        
        f.close()
        print("File for time %04i written." % entry)
        
    
    return 0

if __name__ == '__main__':
    main()