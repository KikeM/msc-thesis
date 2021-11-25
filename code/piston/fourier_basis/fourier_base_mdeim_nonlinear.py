import pickle

from romtime.rom.pod import orth
from scipy.sparse.linalg.eigen import eigs

import fenics
import numpy as np
import pandas as pd
from romtime.conventions import (
    FIG_KWARGS,
    Domain,
    OperatorType,
    PistonParameters,
    RomParameters,
    Stage,
    Treewalk,
)
from romtime.parameters import get_uniform_dist
from romtime.problems.piston import define_piston_problem
from romtime.rom.hrom import HyperReducedPiston

import matplotlib.pyplot as plt

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
DEIM_TS = np.linspace(tf / nt, tf, nt // 4)

deim_params = {
    "rnd_num": RND_DEIM,
    "ts": DEIM_TS.tolist(),
    RomParameters.NUM_SNAPSHOTS: NUM_OFFLINE_DEIM,
    RomParameters.NUM_ONLINE: NUM_ONLINE_DEIM,
    RomParameters.TOL_MU: TOL_MU_DEIM,
    RomParameters.TOL_TIME: TOL_TIME_DEIM,
}
deim_nonlinear_params = {
    "rnd_num": RND_DEIM,
    "ts": DEIM_TS.tolist(),
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

rom = hrom.rom
mdeim = hrom.mdeim_nonlinear

num_mu = 3
num_psi = 5

mu_space = rom.build_sampling_space(num=num_mu, rnd=RND_ONLINE)
dump_json("mu_space_mdeim_certification.json", mu_space)

# Create Fourier basis
twopi = 2 * np.pi
epsilon = 1e-6
twopi_epsilon = twopi * (1 + epsilon)
NX = int(NX)
fourier = np.empty(shape=(NX + 1, 1))
x = hrom.fom.x
for omega in range(1, num_psi + 1):
    print(omega)
    element = np.cos(twopi_epsilon * omega * x)
    element = np.reshape(element, (NX + 1, 1))
    fourier = np.hstack([fourier, element])

# Remove the first element, it contains numerical trash
fourier = np.matrix(fourier[:, 1:])

l2_norms = np.linalg.norm(fourier, axis=0)
fourier_unit = np.divide(fourier, l2_norms)

fourier_unit = np.abs(fourier_unit)

# plt.plot(x, fourier_unit)
# plt.grid(True)
# plt.show()
# plt.close()

mdeim.u_n = fourier_unit
mdeim.run(u_n=fourier_unit, mu_space=mu_space)

sigmas = pd.Series(mdeim.report[Stage.OFFLINE][Treewalk.SPECTRUM_MU])
print("Sigmas")
sigmas.to_csv("sigmas.csv")
print(sigmas.round(4).to_latex())

# mdeim.evaluate(ts=DEIM_TS.tolist(), mu_space=mu_space)
# errors_full_basis = pd.DataFrame(mdeim.errors_rom)

# truncate = mdeim.truncate(1)
# truncate.u_n = fourier_unit
# truncate.evaluate(ts=DEIM_TS.tolist(), mu_space=mu_space)
# errors_truncate = pd.DataFrame(truncate.errors_rom)

# fig, ax = plt.subplots()
# ax.semilogy(DEIM_TS, errors_full_basis)
# ax.semilogy(DEIM_TS, errors_truncate, linestyle="--")
# ax.grid(True)
# ax.set_xlabel("$t$")
# ax.set_ylabel("$L_2$ Error")
# ax.set_title("N-MDEIM Reconstruction Error (Fourier Basis)")

# plt.savefig("fourier_basis_mdeim_truncation_errors_comparison.png", **FIG_KWARGS)