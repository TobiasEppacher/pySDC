from pySDC.Step import step

from implementations.problem_classes.HeatEquation_1D_FD import heat1d
from implementations.datatype_classes.mesh import mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.generic_LU import generic_LU
from implementations.transfer_classes.TransferMesh_1D import mesh_to_mesh_1d_dirichlet

def main():
    """
    A simple test program to setup a full step hierarchy
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = [0.1, 0.2]

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5, 3]

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [31, 15, 7]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heat1d                           # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = mesh                                   # pass data type for f
    description['sweeper_class'] = generic_LU                       # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params                  # pass sweeper parameters
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_1d_dirichlet # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # now the description contains more or less everything we need to create a step with multiple levels
    S = step(description=description)

    # print out and check
    for l in range(len(S.levels)):
        L = S.levels[l]
        print('Level %2i: nvars = %4i -- nnodes = %2i -- dt = %4.2f' %(l, L.prob.params.nvars, L.sweep.coll.num_nodes, L.dt))
        assert L.prob.params.nvars == problem_params['nvars'][min(l,len(problem_params['nvars'])-1)], \
            "ERROR: number of DOFs is not correct on this level, got %s" %L.prob.params.nvars
        assert L.sweep.coll.num_nodes == sweeper_params['num_nodes'][min(l,len(sweeper_params['num_nodes'])-1)], \
            "ERROR: number of nodes is not correct on this level, got %s" %L.sweep.coll.num_nodes
        assert L.dt == level_params['dt'][min(l,len(level_params['dt'])-1)], \
            "ERROR: dt is not correct on this level, got %s" %L.dt


if __name__ == "__main__":
    main()
