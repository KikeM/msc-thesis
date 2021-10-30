import pickle

import fenics
import numpy as np
from romtime.conventions import (
    Domain,
    OperatorType,
    PistonParameters,
    RomParameters,
    Stage,
)

from itertools import product
from romtime.parameters import get_uniform_dist
from romtime.problems.piston import define_piston_problem
from romtime.rom.hrom import HyperReducedPiston
from tqdm import tqdm


fenics.set_log_level(100)


# -----------------------------------------------------------------------------
# Parametrization
grid_params = {
    PistonParameters.A0: {
        "min": 18.0,
        "max": 25.0,
    },
    PistonParameters.OMEGA: {
        "min": 15.0,
        "max": 30.0,
    },
    PistonParameters.DELTA: {
        "min": 0.15,
        "max": 0.3,
    },
}


# -----------------------------------------------------------------------------
# Testing phase
LOAD_BASIS = True
VALIDATE_ROM = False
EVALUATE_DEIM = False


# -----------------------------------------------------------------------------
# Snapshots size
NUM_OFFLINE = None
NUM_ONLINE = 5
ROM_KEEP = 10
SROM_KEEP = 15
SROM_TRUNCATE = SROM_KEEP - ROM_KEEP

NUM_OFFLINE_DEIM = None
NUM_ONLINE_DEIM = None
NUM_ONLINE_NDEIM = None

# -----------------------------------------------------------------------------
# Tolerances
TOL_TIME = None
TOL_MU = None

TOL_TIME_DEIM = None
TOL_MU_DEIM = None

TOL_TIME_NDEIM = None
TOL_MU_NDEIM = None

# -----------------------------------------------------------------------------
# Seeds
RND_OFFLINE = None
RND_ONLINE = 42
RND_DEIM = None

# -----------------------------------------------------------------------------
# Space-Time Domain
NX = 1e3
NT = 5e2

# -----------------------------------------------------------------------------
# HROM Models
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
}

_, boundary_conditions, _, u0, Lt, dLt_dt = define_piston_problem(
    L=domain[Domain.L0],
    nt=domain[Domain.NT],
    nx=domain[Domain.NX],
    tf=domain[Domain.T],
    which="rest",
)

fom_params = dict(
    grid_params=grid_params,
    domain=domain,
    dirichlet=boundary_conditions,
    u0=u0,
    Lt=Lt,
    dLt_dt=dLt_dt,
)

rnd = np.random.RandomState(RND_OFFLINE)
tf, nt = domain[Domain.T], domain[Domain.NT]
# We use less timesteps for the operators
ts = np.linspace(tf / nt, tf, nt // 4)

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

# N_ROM = [5, 10, 15, 20]
# N_SROM = [1, 5, 10]

N_ROM = [10]
N_SROM = [15]

combinations = list(product(N_ROM, N_SROM))
combinations = sorted(combinations, key=lambda x: 2 * x[0] + x[1], reverse=True)
print(combinations)

for n_rom, n_truncation in tqdm(combinations, desc="ROM-SROM Combinations"):

    n_srom = n_rom + n_truncation

    rom_params = {
        RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE,
        RomParameters.TOL_MU: TOL_MU,
        RomParameters.TOL_TIME: TOL_TIME,
        RomParameters.SROM_TRUNCATE: n_truncation,
        RomParameters.SROM_KEEP: n_srom,
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

    # -----------------------------------------------------------------------------
    hrom.setup()
    hrom.setup_hyperreduction()

    hrom.dump_setup("setup.json")

    # -----------------------------------------------------------------------------
    # Offline stage
    hrom.start_from_existing_basis()
    hrom.project_reductors()

    # -----------------------------------------------------------------------------
    # Online stage
    online_params = dict(
        num=NUM_ONLINE,
        rnd_num=RND_ONLINE,
    )
    hrom.evaluate_online(
        params=online_params,
        rnd=np.random.RandomState(RND_ONLINE),
    )
    hrom.dump_mu_space("mu_space.json")

    name_error_estimator = f"errors_estimator_rom_{n_rom}_srom_{n_srom}.pkl"
    with open(name_error_estimator, mode="wb") as fp:
        pickle.dump(hrom.errors, fp)