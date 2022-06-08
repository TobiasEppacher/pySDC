import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.error_estimation.accuracy_check import setup_mpl

from pySDC.core.Hooks import hooks


class log_data(hooks):

    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v1', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v2', value=L.uend[1])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='p3', value=L.uend[2])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_embedded', value=L.status.error_embedded_estimate)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_extrapolated', value=L.status.error_extrapolation_estimate)
        self.increment_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='restart', value=1, initialize=0)
        self.increment_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='sweeps', value=step.status.iter)


def run(use_adaptivity, num_procs):
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 5e-2
    level_params['e_tol'] = 1e-7

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'Vs': 100.,
        'Rs': 1.,
        'C1': 1.,
        'Rpi': 0.2,
        'C2': 1.,
        'Lpi': 1.,
        'Rl': 5.,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['use_HotRod'] = True
    controller_params['use_adaptivity'] = use_adaptivity
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 2e+1

    # instantiate controller
    controller_class = controller_nonMPI
    controller = controller_class(num_procs=num_procs, controller_params=controller_params,
                                  description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats


def plot(stats, use_adaptivity, num_procs, generate_reference=False):
    setup_mpl()
    recomputed = False

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = np.array(sort_stats(filter_stats(stats, type='v1', recomputed=recomputed), sortby='time'))[:, 1]
    v2 = np.array(sort_stats(filter_stats(stats, type='v2', recomputed=recomputed), sortby='time'))[:, 1]
    p3 = np.array(sort_stats(filter_stats(stats, type='p3', recomputed=recomputed), sortby='time'))[:, 1]
    t = np.array(sort_stats(filter_stats(stats, type='p3', recomputed=recomputed), sortby='time'))[:, 0]
    dt = np.array(sort_stats(filter_stats(stats, type='dt', recomputed=recomputed), sortby='time'))[:, 1]
    e_em = np.array(sort_stats(filter_stats(stats, type='e_embedded', recomputed=recomputed), sortby='time'))[:, 1]
    e_ex = np.array(sort_stats(filter_stats(stats, type='e_extrapolated', recomputed=recomputed), sortby='time'))[:, 1]
    restarts = np.array(sort_stats(filter_stats(stats, type='restart', recomputed=recomputed), sortby='time'))[:, 1]
    sweeps = np.array(sort_stats(filter_stats(stats, type='sweeps', recomputed=recomputed), sortby='time'))[:, 1]
    ready = np.logical_and(e_ex != np.array(None), e_em != np.array(None))
    restarts = np.array(sort_stats(filter_stats(stats, type='restart', recomputed=recomputed), sortby='time'))[:, 1]

    if use_adaptivity and num_procs == 1:
        error_msg = 'Error when using adaptivity in serial:'
        expected = {
            'v1': 83.88240785649039,
            'v2': 80.62744370831926,
            'p3': 16.137248067563256,
            'e_em': 6.236525251779312e-08,
            'e_ex': 6.33675177461887e-08,
            'dt': 0.09607389421085785,
            'restarts': 1.0,
            'sweeps': 2416.0,
        }
    elif use_adaptivity and num_procs == 4:
        error_msg = 'Error when using adaptivity in parallel:'
        expected = {
            'v1': 83.88317481237193,
            'v2': 80.62700122995363,
            'p3': 16.136136147445036,
            'e_em': 2.9095385656319195e-08,
            'e_ex': 5.666752074991754e-08,
            'dt': 0.09347348421862582,
            'restarts': 8.0,
            'sweeps': 2400.0,
        }
    elif not use_adaptivity and num_procs == 4:
        error_msg = 'Error with fixed step size in parallel:'
        expected = {
            'v1': 83.88400149770143,
            'v2': 80.62656173487008,
            'p3': 16.134849851184736,
            'e_em': 4.977994905175365e-09,
            'e_ex': 5.048084913047097e-09,
            'dt': 0.05,
            'restarts': 0.0,
            'sweeps': 1600.0,
        }
    elif not use_adaptivity and num_procs == 1:
        error_msg = 'Error with fixed step size in serial:'
        expected = {
            'v1': 83.88400149770143,
            'v2': 80.62656173487008,
            'p3': 16.134849851184736,
            'e_em': 4.977994905175365e-09,
            'e_ex': 5.048084913047097e-09,
            'dt': 0.05,
            'restarts': 0.0,
            'sweeps': 1600.0,
        }

    got = {
        'v1': v1[-1],
        'v2': v2[-1],
        'p3': p3[-1],
        'e_em': e_em[-1],
        'e_ex': e_ex[e_ex != [None]][-1],
        'dt': dt[-1],
        'restarts': restarts.sum(),
        'sweeps': sweeps.sum(),
    }

    if generate_reference:
        print(f'Adaptivity: {use_adaptivity}, num_procs={num_procs}')
        print('expected = {')
        for k in got.keys():
            v = got[k]
            if type(v) in [list, np.ndarray]:
                print(f'\t\'{k}\': {v[v!=[None]][-1]},')
            else:
                print(f'\t\'{k}\': {v},')
        print('}')

    if use_adaptivity and num_procs == 4:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        ax.plot(t, v1, label='v1', ls='-')
        ax.plot(t, v2, label='v2', ls='--')
        ax.plot(t, p3, label='p3', ls='-.')
        ax.legend(frameon=False)
        ax.set_xlabel('Time')
        fig.savefig('data/piline_solution_adaptive.png', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    ax.plot(t, dt, color='black')
    e_ax = ax.twinx()
    e_ax.plot(t, e_em, label=r'$\epsilon_\mathrm{embedded}$')
    e_ax.plot(t, e_ex, label=r'$\epsilon_\mathrm{extrapolated}$', ls='--')
    e_ax.plot(t[ready], abs(e_em[ready] - e_ex[ready]), label='difference', ls='-.')
    e_ax.plot([None, None], label=r'$\Delta t$', color='black')
    e_ax.set_yscale('log')
    if use_adaptivity:
        e_ax.legend(frameon=False, loc='upper left')
    else:
        e_ax.legend(frameon=False, loc='upper right')
    e_ax.set_ylim((7.367539795147197e-12, 1.109667868425781e-05))
    ax.set_ylim((0.012574322653781072, 0.10050387672423527))

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\Delta t$')

    if use_adaptivity:
        fig.savefig(f'data/piline_hotrod_adaptive_{num_procs}procs.png', bbox_inches='tight', dpi=300)
    else:
        fig.savefig(f'data/piline_hotrod_{num_procs}procs.png', bbox_inches='tight', dpi=300)

    for k in expected.keys():
        assert np.isclose(expected[k], got[k], rtol=1e-3),\
               f'{error_msg} Expected {k}={expected[k]:.2e}, got {k}={got[k]:.2e}'


def main():
    generate_reference = False
    for use_adaptivity in [False, True]:
        for num_procs in [1, 4]:
            stats = run(use_adaptivity, num_procs=num_procs)
            plot(stats, use_adaptivity, num_procs=num_procs, generate_reference=generate_reference)


if __name__ == "__main__":
    main()