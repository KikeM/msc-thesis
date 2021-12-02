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
from romtime.utils import dump_pickle

from tqdm import tqdm

fenics.set_log_level(100)


# -----------------------------------------------------------------------------
# Parametrization
grid_params = {
    # -------------------------------------------------------------------------
    # Physical parameters
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
    # -------------------------------------------------------------------------
    # Mesh parameters
    PistonParameters.LOC: {
        "min": 0.2,
        "max": 0.75,
    },
    PistonParameters.SIGMA: {
        "min": 0.1,
        "max": 0.2,
    },
    PistonParameters.SCALE: {
        "min": 0.25,
        "max": 1.75,
    },
}


# -----------------------------------------------------------------------------
# Testing phase
LOAD_BASIS = False
VALIDATE_ROM = False
EVALUATE_DEIM = False


# -----------------------------------------------------------------------------
# Snapshots size
NUM_OFFLINE = 20
NUM_ONLINE = 1
ROM_KEEP = 30
SROM_KEEP = 35
SROM_TRUNCATE = SROM_KEEP - ROM_KEEP
# SROM_TRUNCATE = 1

NUM_OFFLINE_DEIM = 2
NUM_ONLINE_DEIM = 2
NUM_ONLINE_NDEIM = 2
NUM_PSI_NMDEIM = 5

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
RND_DEIM_ONLINE = 7854

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
    PistonParameters.A0: get_uniform_dist(**grid_params[PistonParameters.A0]),
    PistonParameters.OMEGA: get_uniform_dist(**grid_params[PistonParameters.OMEGA]),
    PistonParameters.DELTA: get_uniform_dist(**grid_params[PistonParameters.DELTA]),
    PistonParameters.LOC: get_uniform_dist(**grid_params[PistonParameters.LOC]),
    PistonParameters.SIGMA: get_uniform_dist(**grid_params[PistonParameters.SIGMA]),
    PistonParameters.SCALE: get_uniform_dist(**grid_params[PistonParameters.SCALE]),
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
    RomParameters.NMDEIM_SIZE: None,
}
rnd = np.random.RandomState(RND_OFFLINE)
tf, nt = domain[Domain.T], domain[Domain.NT]
# We use less timesteps for the operators
ts = np.linspace(tf / nt, tf, nt // 4)

deim_params = {
    RomParameters.RND: RND_DEIM,
    RomParameters.RND_ONLINE: RND_DEIM_ONLINE,
    RomParameters.TS: ts.tolist(),
    RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE_DEIM,
    RomParameters.NUM_ONLINE: NUM_ONLINE_DEIM,
    RomParameters.TOL_MU: TOL_MU_DEIM,
    RomParameters.TOL_TIME: TOL_TIME_DEIM,
}
deim_nonlinear_params = {
    RomParameters.RND: RND_DEIM,
    RomParameters.RND_ONLINE: RND_DEIM_ONLINE,
    RomParameters.TS: ts.tolist(),
    RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE_DEIM,
    RomParameters.NUM_ONLINE: NUM_ONLINE_NDEIM,
    RomParameters.TOL_MU: TOL_MU_NDEIM,
    RomParameters.TOL_TIME: TOL_TIME_NDEIM,
    RomParameters.NUM_PSI_NMDEIM: NUM_PSI_NMDEIM,
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
# Assembly trilinear snapshots from RB

# for num in [1, 3, 5]:

print()
print("--------------------------------------------------------")
# print(f"NUM_PSI: {num}")
hrom.setup()
hrom.setup_hyperreduction()
hrom.start_from_existing_basis(deim=False)  # To have ROM basis
rom = hrom.srom


mdeim = hrom.mdeim_trilinear


# -----------------------------------------------------------------------------
# Create f(x)
basis = rom.basis[:, :3]

# Create linear interpolation from RB basis
function = 2.0 * basis[:, 0] + 1.0 * basis[:, 1] + 3.0 * basis[:, 2]
#  function = 2.0 * basis[:, 0] + 1.0 * basis[:, 1]

l2_norms = np.linalg.norm(function, axis=0)
function = np.divide(function, l2_norms)
function = np.reshape(function, (function.shape[0], 1))

# -----------------------------------------------------------------------------

un = rom.basis[:, :15]
print("Number of phi elements")
print(un.shape)
mu_space_off = rom.build_sampling_space(num=NUM_OFFLINE_DEIM, rnd=RND_DEIM)
mdeim.run(u_n=un, mu_space=mu_space_off)

mu_space_on = rom.build_sampling_space(num=NUM_ONLINE_DEIM, rnd=RND_DEIM_ONLINE)

mdeim.evaluate(ts=ts, funcs=function, mu_space=mu_space_on)

mdeim.dump_fom_basis()
name_pickle = f"report.pkl"
print(mdeim.report)
dump_pickle(path=name_pickle, obj=mdeim.report)

print(mdeim.errors_rom)