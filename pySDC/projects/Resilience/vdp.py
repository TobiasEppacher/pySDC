# script to run a van der Pol problem
import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.core.Errors import ProblemError
from pySDC.projects.Resilience.hook import log_error_estimates


def plot_step_sizes(stats, ax):

    # convert filtered statistics to list of iterations count, sorted by process
    u = np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
    p = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
    t = np.array(get_sorted(stats, type='u', recomputed=False, sortby='time'))[:, 0]

    e_em = np.array(get_sorted(stats, type='e_embedded', recomputed=False, sortby='time'))[:, 1]
    dt = np.array(get_sorted(stats, type='dt', recomputed=False, sortby='time'))
    restart = np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))

    ax.plot(t, u, label=r'$u$')
    ax.plot(t, p, label=r'$p$')

    dt_ax = ax.twinx()
    dt_ax.plot(dt[:, 0], dt[:, 1], color='black')
    dt_ax.plot(t, e_em, color='magenta')
    dt_ax.set_yscale('log')
    dt_ax.set_ylim((5e-10, 3e-1))

    ax.plot([None], [None], label=r'$\Delta t$', color='black')
    ax.plot([None], [None], label=r'$\epsilon_\mathrm{embedded}$', color='magenta')
    ax.plot([None], [None], label='restart', color='grey', ls='-.')

    for i in range(len(restart)):
        if restart[i, 1] > 0:
            ax.axvline(restart[i, 0], color='grey', ls='-.')
    ax.legend(frameon=False)

    ax.set_xlabel('time')


def plot_wiggleroom(stats, ax, wiggle):
    sweeps = get_sorted(stats, type='sweeps', recomputed=None)
    restarts = get_sorted(stats, type='restart', recomputed=None)

    color = 'blue' if wiggle else 'red'
    ls = ':' if not wiggle else '-.'
    label = 'with' if wiggle else 'without'
     
    ax.plot([me[0] for me in sweeps], np.cumsum([me[1] for me in sweeps]), color=color, label=f'{label} wiggleroom')
    [ax.axvline(me[0], color=color, ls=ls) for me in restarts if me[1]]

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$k$')
    ax.legend(frameon=False)



def run_vdp(
    custom_description=None,
    num_procs=1,
    Tend=10.0,
    hook_class=log_error_estimates,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    use_MPI=False,
    **kwargs,
):

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {
        'mu': 5.0,
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
        'u0': np.array([2.0, 0.0]),
    }

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = vanderpol  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if k == 'sweeper_class':
                description[k] = custom_description[k]
                continue
            description[k] = {**description.get(k, {}), **custom_description.get(k, {})}

    # set time parameters
    t0 = 0.0

    # instantiate controller
    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

        # get initial values on finest level
        P = controller.S.levels[0].prob
        uinit = P.u_exact(t0)
    else:
        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

    # insert faults
    if fault_stuff is not None:
        controller.hooks.random_generator = fault_stuff['rng']
        controller.hooks.add_fault(
            rnd_args={'iteration': 3, **fault_stuff.get('rnd_params', {})},
            args={'time': 1.0, 'target': 0, **fault_stuff.get('args', {})},
        )

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ProblemError:
        stats = controller.hooks.return_stats()

    return stats, controller, Tend


def fetch_test_data(stats, comm=None, use_MPI=False):
    """
    Get data to perform tests on from stats

    Args:
        stats (pySDC.stats): The stats object of the run

    Returns:
        dict: Key values to perform tests on
    """
    types = ['e_embedded', 'restart', 'dt', 'sweeps', 'residual_post_step']
    data = {}
    for type in types:
        if type not in get_list_of_types(stats):
            raise ValueError(f"Can't read type \"{type}\" from stats, only got", get_list_of_types(stats))

        if comm is None or use_MPI == False:
            data[type] = [me[1] for me in get_sorted(stats, type=type, recomputed=None, sortby='time')]
        else:
            data[type] = [me[1] for me in get_sorted(stats, type=type, recomputed=None, sortby='time', comm=comm)]
    return data


def check_if_tests_match(data_nonMPI, data_MPI):
    """
    Check if the data matches between MPI and nonMPI versions

    Args:
        data_nonMPI (dict): Key values to perform tests on
        data_MPI (dict): Key values to perform tests on

    Returns:
        None
    """
    ops = [np.mean, np.min, np.max, len, sum]
    for type in data_nonMPI.keys():
        for op in ops:
            val_nonMPI = op(data_nonMPI[type])
            val_MPI = op(data_MPI[type])
            assert np.isclose(val_nonMPI, val_MPI), (
                f"Mismatch in operation {op.__name__} on type \"{type}\": with {data_MPI['size'][0]} ranks: "
                f"nonMPI: {val_nonMPI}, MPI: {val_MPI}"
            )
    print(f'Passed with {data_MPI["size"][0]} ranks')


def mpi_vs_nonMPI(MPI_ready, comm):
    if MPI_ready:
        size = comm.size
        rank = comm.rank
        use_MPI = [True, False]
    else:
        size = 1
        rank = 0
        use_MPI = [False, False]

    if rank == 0:
        print(f"Running with {size} ranks")

    custom_description = {'convergence_controllers': {}}
    custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7, 'wiggleroom': False}

    custom_controller_params = {'logger_level': 30}

    data = [{}, {}]

    for i in range(2):
        if use_MPI[i] or rank == 0:
            stats, controller, Tend = run_vdp(
                custom_description=custom_description,
                num_procs=size,
                use_MPI=use_MPI[i],
                custom_controller_params=custom_controller_params,
                Tend=1.0,
                comm=comm,
            )
            data[i] = fetch_test_data(stats, comm, use_MPI=use_MPI[i])
            data[i]['size'] = [size]

    if rank == 0:
        check_if_tests_match(data[1], data[0])


def check_adaptivity_with_wiggleroom(comm=None, size=1):
    fig, ax = plt.subplots()
    custom_description = {'convergence_controllers': {}, 'level_params': {'dt': 1.e-2}}
    custom_controller_params = {'logger_level': 15, 'all_to_done': True}
    results = {'e': {}, 'sweeps': {}, 'restarts': {}}
    size = comm.size if comm is not None else size
     
    for wiggle in [True, False]:
        custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7, 'wiggleroom': wiggle}
        stats, controller, Tend = run_vdp(
            custom_description=custom_description,
            num_procs=size,
            use_MPI=comm is not None,
            custom_controller_params=custom_controller_params,
            Tend=10.0e0,
            comm=comm,
        )
        plot_wiggleroom(stats, ax, wiggle)

        # check error
        u = get_sorted(stats, type='u', recomputed=False)[-1]
        if comm is None:
            u_exact = controller.MS[0].levels[0].prob.u_exact(t=u[0])
        else:
            u_exact = controller.S.levels[0].prob.u_exact(t=u[0])
        results['e'][wiggle] = abs(u[1] - u_exact)

        # check iteration counts
        results['sweeps'][wiggle] = sum([me[1] for me in get_sorted(stats, type='sweeps', recomputed=None, comm=comm)])
        results['restarts'][wiggle] = sum([me[1] for me in get_sorted(stats, type='restart', comm=comm)])

    fig.tight_layout()
    fig.savefig(f'data/vdp-{size}procs{"-use_MPI" if comm is not None else ""}-wiggleroom.png')

    assert np.isclose(results['e'][True], results['e'][False], rtol=5.), 'Errors don\'t match with wiggleroom and without, got '\
     f'{results["e"][True]:.2e} and {results["e"][False]:.2e}'
    if size == 1:
        assert results['sweeps'][True] - results['sweeps'][False] == 1301 - 1344, '{Expected to save 43 iterations '\
                f"with wiggleroom, got {results['sweeps'][False] - results['sweeps'][True]}"
        assert results['restarts'][True] - results['restarts'][False] == 0 - 10, '{Expected to save 10 restarts '\
                f"with wiggleroom, got {results['restarts'][False] - results['restarts'][True]}"
        print('Passed wiggleroom tests with 1 process')
    if size == 4:
        assert results['sweeps'][True] - results['sweeps'][False] == 2916 - 3008, '{Expected to save 92 iterations '\
                f"with wiggleroom, got {results['sweeps'][False] - results['sweeps'][True]}"
        assert results['restarts'][True] - results['restarts'][False] == 0 - 18, '{Expected to save 18 restarts '\
                f"with wiggleroom, got {results['restarts'][False] - results['restarts'][True]}"
        print('Passed wiggleroom tests with 4 processes')

 
if __name__ in "__main__":
    try:
        from mpi4py import MPI

        MPI_ready = True
        comm = MPI.COMM_WORLD
    except ModuleNotFoundError:
        MPI_ready = False
        comm = None
    mpi_vs_nonMPI(MPI_ready, comm)
    comm = None
    check_adaptivity_with_wiggleroom(comm=comm, size=4)
