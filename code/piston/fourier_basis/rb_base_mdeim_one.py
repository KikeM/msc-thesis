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

from romtime.utils import dump_json, read_pickle

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
mdeim = hrom.mdeim_nonlinear

num_mu = 10
num_psi = 2

rom = hrom.rom
mu_space = rom.build_sampling_space(num=num_mu, rnd=RND_ONLINE)
dump_json("mu_space_mdeim_certification.json", mu_space)

# -----------------------------------------------------------------------------
# Load RB basis
basis = read_pickle("basis_srom.pkl")
basis = basis[:, :3]

# Create linear interpolation from RB basis
function = 2.0 * basis[:, 0] + 1.0 * basis[:, 1] + 3.0 * basis[:, 2]

l2_norms = np.linalg.norm(function, axis=0)
function = np.divide(function, l2_norms)

function = np.reshape(function, (function.shape[0], 1))

x = hrom.fom.x
plt.plot(x, function)
plt.plot(x, basis, alpha=0.5, linestyle="--", linewidth=0.75)
plt.grid(True)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Linear comb. of RB basis functions")
plt.savefig("linear_combination.png", **FIG_KWARGS)
plt.close()

quit()

NUM_BASIS = 3
FORM = "Trilinear"

mdeim.u_n = function

if NUM_BASIS == 1:
    _basis = basis[:, 0]
elif NUM_BASIS == 3:
    _basis = basis

mdeim.run(u_n=_basis, mu_space=mu_space)

sigmas = pd.Series(mdeim.report[Stage.OFFLINE][Treewalk.SPECTRUM_MU])
print("Sigmas")
sigmas.to_csv("sigmas.csv")
sigmas.to_latex(buf="sigmas.tex", float_format="%.1e")

fig, ax = plt.subplots()
ax.semilogy(sigmas)
ax.grid(True)
ax.set_ylabel("$\\sigma_i$")
ax.set_xlabel("$i$-th component")
ax.set_title(f"Singular Values ({FORM}, {NUM_BASIS}-RB Basis)")
plt.savefig(f"sigmas_{FORM}_num_{NUM_BASIS}.png", **FIG_KWARGS)
plt.close()

mdeim.evaluate(ts=DEIM_TS.tolist(), mu_space=mu_space)
errors_full_basis = pd.DataFrame(mdeim.errors_rom)
errors_full_basis.to_csv("errors_full_basis.csv")

print("MDEIM size:", mdeim.N)

truncate = mdeim.truncate(1)
truncate.u_n = function
truncate.evaluate(ts=DEIM_TS.tolist(), mu_space=mu_space)
errors_truncate = pd.DataFrame(truncate.errors_rom)
errors_truncate.to_csv("errors_truncate_basis.csv")

fig, ax = plt.subplots()
ax.semilogy(DEIM_TS, errors_full_basis)
ax.semilogy(DEIM_TS, errors_truncate, linestyle="--")
ax.grid(True)
ax.set_xlabel("$t$")
ax.set_ylabel("$L_2$ Error")
ax.set_title(f"N-MDEIM Reconstruction Error ({FORM}, {NUM_BASIS}-RB Basis)")

plt.savefig(f"rb_basis_mdeim_errors_{FORM.lower()}_num_{NUM_BASIS}.png", **FIG_KWARGS)