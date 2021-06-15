from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame

from romtime.parameters import get_uniform_dist
from romtime.problems.mfp1 import HyperReducedOrderModelMoving, define_mfp1_problem
from romtime.conventions import Domain, OperatorType, RomParameters
from romtime.rom.base import Reductor
from pprint import pprint

from copy import deepcopy

domain = dict(
    L0=2.0,
    nx=500,
    nt=500,
    T=10.0,
)


# Compute omega variables
n_min = 0.5
n_max = 0.8

tf = domain["T"]
omegas = [(1.0 / tf) * np.arcsin(1.0 - n) for n in (n_min, n_max)]
omega_max = max(omegas)
omega_min = min(omegas)

grid = {
    "delta": get_uniform_dist(min=0.01, max=5.0),
    "beta": get_uniform_dist(min=0.05, max=0.1),
    "alpha_0": get_uniform_dist(min=0.01, max=2.0),
    "omega": get_uniform_dist(min=omega_min, max=omega_max),
}

_, boundary_conditions, forcing_term, u0, ue, Lt, dLt_dt = define_mfp1_problem()

fom_params = dict(
    domain=domain,
    dirichlet=boundary_conditions,
    forcing_term=forcing_term,
    u0=u0,
    exact_solution=ue,
    Lt=Lt,
    dLt_dt=dLt_dt,
    degrees=1,
)


reductor = Reductor(grid=grid)
rnd = np.random.RandomState(0)
mu_space = reductor.build_sampling_space(num=1, rnd=rnd)
mu_space = list(mu_space)

# -----------------------------------------------------------------------------
# ROM
tols = [0.21, 0.41, 0.61, 0.81, 1.0]

tols = [0.81]

cartesian_product = product([True], [True], [True, False], [True])

_errors = []
cols = []

EVALUATE = True
for mass, stiffness, convection, rhs in cartesian_product:

    models = {
        OperatorType.MASS: mass,
        OperatorType.STIFFNESS: stiffness,
        OperatorType.CONVECTION: convection,
        OperatorType.RHS: rhs,
    }

    name = "".join([str(value)[0] for value in models.values()])
    cols.append(name)

    # Reduced basis parameters
    rom_params = {
        RomParameters.NUM_SNAPSHOTS: None,
        RomParameters.TOL_TIME: None,
        RomParameters.TOL_MU: None,
    }

    # (M)DEIM parametrization
    tf, nt = domain[Domain.T], domain[Domain.NT]
    ts = np.linspace(tf / nt, tf, nt)

    deim_params = {
        RomParameters.TS: ts,
        RomParameters.NUM_SNAPSHOTS: None,
    }
    mdeim_params = {
        RomParameters.TS: ts,
        RomParameters.NUM_SNAPSHOTS: None,
        # RomParameters.NUM_MU: 2,
        # RomParameters.NUM_TIME: 50,
    }

    hrom = HyperReducedOrderModelMoving(
        grid=grid,
        fom_params=fom_params,
        rom_params=rom_params,
        deim_params=deim_params,
        mdeim_params=mdeim_params,
        models=models,
        rnd=rnd,
    )

    hrom.setup()
    hrom.setup_hyperreduction()
    hrom.run_offline_hyperreduction(mu_space=mu_space, evaluate=EVALUATE)

    # hrom.mdeim_mass.plot_errors()
    # hrom.mdeim_stiffness.plot_errors()
    # hrom.mdeim_convection.plot_errors()
    # hrom.deim_rhs.plot_errors()

    _models = [
        hrom.mdeim_mass,
        hrom.mdeim_stiffness,
        hrom.mdeim_convection,
        hrom.deim_rhs,
    ]

    hrom.run_offline_rom(mu_space=mu_space)

    hrom.generate_summary()
    # hrom.plot_spectrums()
    hrom.evaluate_online(mu_space=mu_space)

    # timesteps = hrom.rom.timesteps[1:]
    # for _, errors in hrom.rom.errors.items():
    #     _errors.append(deepcopy(errors))

    pprint(hrom.summary_basis)

    for model in _models:

        errors_rom = model.errors_rom
        name = model.name

        for error in errors_rom.values():
            plt.plot(model.tree_walk_params["ts"], np.log10(error), label=name)

    hrom.plot_errors(new=False, label="Reduced Basis", show=False)

    title = "Online Errors"
    if convection:
        title += " - Convection: MDEIM"
    else:
        title += " - Convection: FOM"

    plt.title(title)
    plt.legend(loc="center right")

    # plt.grid(True)
    plt.show()

    del hrom

# errors = np.array(np.log10(_errors)).T

# DataFrame(errors, columns=cols).to_csv("errors.csv")

# plt.plot(timesteps, errors)
# plt.legend(cols)
# plt.grid(True)
# plt.show()

# labels = ["{:.2f}".format(tol) for tol in tols]
# plt.legend(labels, title="Energy ratio", ncol=2, loc="best")
# plt.grid(True)
# plt.show()

#

# -----------------------------------------------------------------------------
# HROM
# tols = np.linspace(0.1, 1.0, 4)

# for idx, tol in enumerate(tols):

#     rom_params = {
#         RomParameters.NUM_SNAPSHOTS: None,
#         RomParameters.TOL_TIME: None,
#         RomParameters.TOL_MU: tol,
#     }

#     tf, nt = domain[Domain.T], domain[Domain.NT]
#     ts = np.linspace(tf / nt, tf, nt)

#     deim_params = {"ts": ts, RomParameters.NUM_SNAPSHOTS: 1}

#     hrom = HyperReducedOrderModelMoving(
#         grid=grid,
#         fom_params=fom_params,
#         rom_params=rom_params,
#         deim_params=deim_params,
#         mdeim_params=deim_params,
#         rnd=rnd,
#     )

#     hrom.setup()

#     hrom.setup_hyperreduction()
#     hrom.run_offline_hyperreduction()

#     hrom.run_offline_rom(mu_space=mu_space)

#     # online_params = dict(num=1, rnd=np.random.RandomState(2))

#     hrom.evaluate_online(mu_space=mu_space)

#     hrom.generate_summary()

#     label = "{:.2f}".format(tol)

#     if idx == 0:
#         hrom.plot_errors(new=True)
#     else:
#         hrom.plot_errors(new=False)

#     del hrom


# labels = ["{:.2f}".format(tol) for tol in tols]
# plt.legend(labels, title="Energy ratio", ncol=2, loc="best")
# plt.grid(True)
# plt.show()
