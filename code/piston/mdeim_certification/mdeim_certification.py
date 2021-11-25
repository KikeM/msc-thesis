import pickle

import fenics
import numpy as np
import pandas as pd
from romtime.conventions import (
    Domain,
    OperatorType,
    PistonParameters,
    RomParameters,
    Stage,
)
from romtime.parameters import get_uniform_dist
from romtime.problems.piston import define_piston_problem
from romtime.rom.hrom import HyperReducedPiston
from tqdm import tqdm

from romtime.utils import dump_json

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
LOAD_BASIS = False
VALIDATE_ROM = False
EVALUATE_DEIM = False


# -----------------------------------------------------------------------------
# Snapshots size
NUM_OFFLINE = 10
NUM_ONLINE = 5
ROM_KEEP = 7
SROM_KEEP = None  # Keep all basis elements
if SROM_KEEP is None:
    SROM_TRUNCATE = 1
else:
    SROM_TRUNCATE = SROM_KEEP - ROM_KEEP

NUM_OFFLINE_DEIM = 10
NUM_ONLINE_DEIM = 1
NUM_ONLINE_NDEIM = 1

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
RND_OFFLINE = 0
RND_ONLINE = 5656
RND_DEIM = 440

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

rom_params = {
    RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE,
    RomParameters.TOL_MU: TOL_MU,
    RomParameters.TOL_TIME: TOL_TIME,
    RomParameters.SROM_TRUNCATE: SROM_TRUNCATE,
    RomParameters.SROM_KEEP: SROM_KEEP,
}
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
if EVALUATE_DEIM:
    hrom.evaluate_deim()

rom = hrom.rom
mdeim = hrom.mdeim_trilinear

num_mu = 10
num_psi = 10

mu_space = rom.build_sampling_space(num=num_mu, rnd=RND_ONLINE)
dump_json("mu_space_mdeim_certification.json", mu_space)


mdeim.u_n = rom.basis[:, :num_psi]
mdeim.evaluate(ts=ts.tolist(), mu_space=mu_space)

pd.DataFrame(mdeim.errors_rom, index=ts).to_csv(f"psi_{num_psi}_full_errors.csv")

modes_to_remove = [
    65,
    64,
    63,
    62,
    61,
    60,
    59,
    58,
    57,
    56,
    55,
    40,
    45,
    30,
    35,
    25,
    15,
    5,
    1,
    2,
    3,
]
modes_to_remove = sorted(modes_to_remove, reverse=False)
for n in tqdm(modes_to_remove, desc="Number of modes"):

    tmdeim = mdeim.truncate(n=n)
    tmdeim.u_n = rom.basis[:, :num_psi]
    tmdeim.evaluate(ts=ts.tolist(), mu_space=mu_space)

    name_errors = f"psi_{num_psi}_num_{n}_trucanted_errors.csv"
    pd.DataFrame(tmdeim.errors_rom, index=ts).to_csv(name_errors)

    del tmdeim
