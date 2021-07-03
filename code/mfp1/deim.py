from itertools import product
from matplotlib import pyplot as plt
import numpy as np

from pprint import pprint
from pandas.core.frame import DataFrame

from romtime.parameters import get_uniform_dist
from romtime.problems.mfp1 import (
    HyperReducedOrderModelFixed,
    HyperReducedOrderModelMoving,
    define_mfp1_problem,
)
from romtime.conventions import Domain, RomParameters
from romtime.rom.base import Reductor

domain = dict(
    L0=2.0,
    nx=500,
    nt=100,
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
mu_space = reductor.build_sampling_space(num=10, rnd=rnd)
mu_space = list(mu_space)

# -----------------------------------------------------------------------------
# ROM
tols = [0.21, 0.41, 0.61, 0.81, 1.0]
tols = [0.0]


rom_params = {
    RomParameters.NUM_SNAPSHOTS: None,
    RomParameters.TOL_TIME: None,
    RomParameters.TOL_MU: 1e-10,
}

tf, nt = domain[Domain.T], domain[Domain.NT]
ts = np.linspace(tf / nt, tf, nt)

ints_mu = [1, 2, 3, 5, 10]
ints_t = [1, 2, 5, 10, 20]
integers = product(ints_mu, ints_t)

results = []
for num_mu, num_t in integers:

    deim_params = {
        RomParameters.TS: ts,
        RomParameters.NUM_SNAPSHOTS: None,
        RomParameters.NUM_MU: num_mu,
        RomParameters.NUM_TIME: num_t,
    }

    hrom = HyperReducedOrderModelMoving(
        grid=grid,
        fom_params=fom_params,
        rom_params=rom_params,
        deim_params=deim_params,
        mdeim_params=deim_params,
        rnd=rnd,
    )

    hrom.setup()
    hrom.setup_hyperreduction()
    hrom.run_offline_hyperreduction(mu_space=mu_space)

    hrom.mdeim_stiffness.create_errors_summary()
    maximum = hrom.mdeim_stiffness.summary_errors["max"].max()

    results.append((num_mu, num_t, np.log10(maximum)))

    del hrom

results = DataFrame(results, columns=["num_mu", "num_time", "max"])

print(results)

for idx in ints_mu:

    mask = results["num_mu"] == idx
    data = results.loc[mask, ["num_time", "max"]].values

    plt.plot(data[:, 0], data[:, 1], label=str(idx))

plt.xlabel("# time")
plt.legend(title="# mu")
plt.grid(True)
plt.show()
