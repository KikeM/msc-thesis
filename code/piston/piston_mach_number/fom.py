import fenics
import matplotlib.pyplot as plt
import numpy as np
from romtime.conventions import FIG_KWARGS, Domain, PistonParameters
from romtime.fom import OneDimensionalBurgers
from romtime.parameters import get_uniform_dist
from romtime.problems.piston import define_piston_problem
from sklearn.model_selection import ParameterSampler
import pandas as pd
from copy import deepcopy

import ujson

from tqdm import tqdm

fenics.set_log_level(100)

plt.set_cmap("viridis")


def plot_probes(probes, save=None):

    locations = probes.columns

    fig, axes = plt.subplots(
        nrows=len(locations),
        ncols=1,
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.20},
    )

    axes = axes.flatten()
    ts = probes.index

    for idx_probe, loc in enumerate(locations):
        values = probes[loc]

        ax = axes[idx_probe]
        ax.plot(ts, values)
        ax.grid(True)
        # ax.set_title(label)
        ax.set_ylabel(f"$u({loc}, t)$")

    plt.xlabel("u (m/s)")
    plt.xlabel("t (s)")

    if save is None:
        plt.show()
    else:
        plt.savefig(save + ".png", **FIG_KWARGS)

    plt.close()


def build_sampling_space(grid, num, rnd=None):
    """Build sampling space according to filling linearity slope.

    Parameters
    ----------
    num : int
    rnd : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """

    print("Building linearity sampling space ...")

    piston_mach_space = compute_piston_mach_number_space(grid=grid, num=num)

    sampler = ParameterSampler(
        param_distributions=grid, n_iter=int(1e4), random_state=rnd
    )

    samples = []
    domains = [
        (start, end) for start, end in zip(piston_mach_space, piston_mach_space[1:])
    ]
    for sample in tqdm(sampler):

        piston_mach = compute_piston_mach(sample)

        remove = None
        for domain in domains:
            start, end = domain

            is_ge = piston_mach >= start
            is_le = piston_mach <= end
            inside = is_ge & is_le

            if inside:

                sample[PistonParameters.MACH_PISTON] = piston_mach
                samples.append(sample)

                remove = domain
                break

        if remove is not None:
            domains.remove(remove)
            print(len(domains))

        if len(domains) == 0:
            break

    # Add sorting so the idx makes sense
    samples = sorted(samples, key=lambda x: x[PistonParameters.MACH_PISTON])

    return samples


def compute_piston_mach(sample):

    A0 = PistonParameters.A0
    OMEGA = PistonParameters.OMEGA
    DELTA = PistonParameters.DELTA

    mach = sample[DELTA] * sample[OMEGA] / sample[A0]

    return mach


def compute_piston_mach_number_space(grid, num):

    A0 = PistonParameters.A0
    OMEGA = PistonParameters.OMEGA
    DELTA = PistonParameters.DELTA

    params = [A0, OMEGA, DELTA]
    support = {}
    for var in params:
        _support = grid[var].support()
        support[var] = {"min": min(_support), "max": max(_support)}

    # Less input into the system, maximum linearity
    mach_min = support[DELTA]["min"] * support[OMEGA]["min"] / support[A0]["max"]

    # Maximum input into the system, minimum linearity
    # forcing_max = support[DELTA]["max"] * support[OMEGA]["max"] / support[A0]["min"]
    mach_max = 0.4

    print(f"forcing : (min, max) = {mach_min}, {mach_max}")

    space = np.linspace(start=mach_min, stop=mach_max, num=num + 1)

    return space


def create_solver(L, nx, nt, tf, grid_base, which):
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
    ) = define_piston_problem(L, nx, tf, nt, which=which)

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
    # Constants
    PistonParameters.ALPHA: {
        "min": 1e-6,
        "max": 1e-6,
    },
    PistonParameters.GAMMA: {
        "min": 1.4,
        "max": 1.4,
    },
}

with open("grid_params.json", mode="w") as fp:
    ujson.dump(grid_params, fp)

WHICH = "rest"
# -----------------------------------------------------------------------------
# Space-Time Domain
NX = 1e3
NT = 1e2

# Create data structures
domain = {
    Domain.L0: 1.0,
    Domain.NX: int(NX),
    Domain.NT: int(NT),
    Domain.T: 0.75,
}

solver = create_solver(
    L=domain[Domain.L0],
    nt=domain[Domain.NT],
    nx=domain[Domain.NX],
    tf=domain[Domain.T],
    grid_base=grid_params,
    which=WHICH,
)

grid = {
    "a0": get_uniform_dist(**grid_params["a0"]),
    "omega": get_uniform_dist(**grid_params["omega"]),
    "delta": get_uniform_dist(**grid_params["delta"]),
    # Constants
    "alpha": get_uniform_dist(**grid_params["alpha"]),
    "gamma": get_uniform_dist(**grid_params["gamma"]),
}


N_SAMPLES = 20

samples = build_sampling_space(grid=grid, num=N_SAMPLES)
results = []
for idx, mu in tqdm(enumerate(samples), total=len(samples)):

    try:
        solver.setup()
        solver.update_parametrization(mu)
        solver.solve()

        name_probes = f"probes_FOM_{idx}"
        probes = solver.save_probes(name=name_probes + ".csv")
        plot_probes(probes, save=name_probes)

        u_p, eta = solver.nonlinearity
    except:
        eta = None

    mu[PistonParameters.NONLINEARITY] = eta

    print(f"{idx} : (u_p, eta) = ({u_p}, {eta})")
    results.append(deepcopy(mu))

df = pd.DataFrame(results)

# assume none
df["wiggles"] = "no"

df = df.round(2)
print(df)

df.to_csv("wiggles.csv")
