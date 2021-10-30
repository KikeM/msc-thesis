import pickle
from pathlib import Path

import fenics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from romtime.conventions import Domain, OperatorType, RomParameters, Stage
from romtime.fom import OneDimensionalBurgers
from romtime.parameters import get_uniform_dist
from romtime.problems.piston import define_piston_problem
from romtime.rom.hrom import HyperReducedPiston

fenics.set_log_level(50)


def create_solver(L, nx, nt, tf, grid_base):
    """Solve burgers equation problem.

    Parameters
    ----------
    L : fenics.Constant
    nx : int
    nt : int
    ft : float
    parameters : tuple

    Returns
    -------
    solver : romtime.OneDimensionalHeatEquationSolver
    """

    (
        domain,
        boundary_conditions,
        forcing_term,
        u0,
        Lt,
        dLt_dt,
    ) = define_piston_problem(L, nx, tf, nt)

    solver = OneDimensionalBurgers(
        domain=domain,
        dirichlet=boundary_conditions,
        parameters=grid_base,
        forcing_term=forcing_term,
        degrees=1,
        u0=u0,
        exact_solution=None,
        Lt=Lt,
        dLt_dt=dLt_dt,
    )

    solver.setup()

    return solver


# -----------------------------------------------------------------------------
# Parametrization
grid_params = {
    "a0": {
        "min": 9.0,
        "max": 15.0,
    },
    "omega": {
        "min": 15.0,
        "max": 25.0,
    },
    "delta": {
        "min": 0.1,
        "max": 0.15,
    },
    # Constants
    "alpha": {
        "min": 1e-6,
        "max": 1e-6,
    },
    "gamma": {
        "min": 1.4,
        "max": 1.4,
    },
}

# Testing phase
VALIDATE_ROM = True
EVALUATE_DEIM = False

# Snapshots size
NUM_OFFLINE = 20
NUM_ONLINE = 20
NUM_OFFLINE_DEIM = 5
NUM_ONLINE_DEIM = 10
NUM_ONLINE_NDEIM = 5

# Tolerances
TOL_TIME = None
TOL_MU = 0.5

TOL_TIME_DEIM = None
TOL_MU_DEIM = None

TOL_TIME_NDEIM = None
TOL_MU_NDEIM = None

# Seeds
RND_OFFLINE = 0
RND_ONLINE = 442
RND_DEIM = 440

# Space-Time Domain
NX = 1e3
NT = 5e2

models = {
    OperatorType.MASS: True,
    OperatorType.STIFFNESS: True,
    OperatorType.CONVECTION: True,
    OperatorType.TRILINEAR: True,
    OperatorType.NONLINEAR_LIFTING: True,
    OperatorType.RHS: True,
}

# -----------------------------------------------------------------------------
# Create data structures
domain = {
    Domain.L0: 1.0,
    Domain.NX: int(NX),
    Domain.NT: int(NT),
    Domain.T: 1.0,
}

grid = {
    "a0": get_uniform_dist(**grid_params["a0"]),
    "omega": get_uniform_dist(**grid_params["omega"]),
    "delta": get_uniform_dist(**grid_params["delta"]),
    # Constants
    "alpha": get_uniform_dist(**grid_params["alpha"]),
    "gamma": get_uniform_dist(**grid_params["gamma"]),
}

_, boundary_conditions, _, u0, Lt, dLt_dt = define_piston_problem(
    L=domain[Domain.L0],
    nt=domain[Domain.NT],
    nx=domain[Domain.NX],
    tf=domain[Domain.T],
)

fom_params = dict(
    grid_params=grid_params,
    domain=domain,
    dirichlet=boundary_conditions,
    u0=u0,
    Lt=Lt,
    dLt_dt=dLt_dt,
)

rom_params = {
    RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE,
    RomParameters.TOL_MU: TOL_MU,
    RomParameters.TOL_TIME: TOL_TIME,
}
rnd = np.random.RandomState(RND_OFFLINE)
tf, nt = domain[Domain.T], domain[Domain.NT]
ts = np.linspace(tf / nt, tf, nt // 4)
# ts = np.linspace(tf / nt, tf, nt)

deim_params = {
    "rnd_num": RND_DEIM,
    "ts": ts.tolist(),
    RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE_DEIM,
    RomParameters.NUM_ONLINE: NUM_ONLINE_DEIM,
    RomParameters.TOL_MU: TOL_MU_DEIM,
    RomParameters.TOL_TIME: TOL_TIME_DEIM,
}
deim_nonlinear_params = {
    "rnd_num": RND_DEIM,
    "ts": ts.tolist(),
    RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE_DEIM,
    RomParameters.NUM_ONLINE: NUM_ONLINE_NDEIM,
    RomParameters.TOL_MU: TOL_MU_NDEIM,
    RomParameters.TOL_TIME: TOL_TIME_NDEIM,
}

hrom = HyperReducedPiston(
    grid=grid,
    fom_params=fom_params,
    rom_params=rom_params,
    deim_params=deim_params,
    mdeim_params=deim_params,
    mdeim_nonlinear_params=deim_nonlinear_params,
    models=models,
    rnd=rnd,
)

#  -----------------------------------------------------------------------------
hrom.setup()
hrom.setup_hyperreduction()

hrom.dump_setup("setup.json")

#  -----------------------------------------------------------------------------
hrom.run_offline_rom()
hrom.run_offline_hyperreduction(evaluate=EVALUATE_DEIM)
hrom.dump_reduced_basis()
# hrom.dump_validation_fom()
hrom.rom.project_reductors()

hrom.generate_summary()
hrom.dump_errors_deim()
hrom.dump_mu_space_deim()
hrom.summary_basis.to_csv("summary_basis.csv")

if VALIDATE_ROM:
    hrom.evaluate_validation()
    hrom.dump_mu_space("mu_space.json")
    hrom.plot_errors(which=Stage.VALIDATION, save="validation_errors", show=False)
    hrom.dump_errors(which=Stage.VALIDATION)

# -----------------------------------------------------------------------------

online_params = dict(
    num=NUM_ONLINE,
    rnd_num=RND_ONLINE,
)
hrom.evaluate_online(
    params=online_params,
    rnd=np.random.RandomState(RND_ONLINE),
)
hrom.dump_errors(which=Stage.ONLINE)
hrom.plot_errors(which=Stage.ONLINE, save="online_errors", show=False)
hrom.dump_mu_space("mu_space.json")

hrom.generate_summary()

with open("summary_energy.pkl", mode="wb") as fp:
    pickle.dump(hrom.summary_energy, fp)

hrom.plot_spectrums(save="spectrum_decay", show=False)
hrom.plot_energy(save="energy", show=False)
hrom.mdeim_mass.plot_errors(show=False, save=True)
hrom.mdeim_stiffness.plot_errors(show=False, save=True)
hrom.mdeim_convection.plot_errors(show=False, save=True)
hrom.mdeim_trilinear.plot_errors(show=False, save=True)
hrom.mdeim_trilinear_lifting.plot_errors(show=False, save=True)
hrom.deim_rhs.plot_errors(show=False, save=True)
