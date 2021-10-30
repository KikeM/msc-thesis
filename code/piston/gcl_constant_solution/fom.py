from pprint import pprint

import fenics
import matplotlib.pyplot as plt
import numpy as np
import ujson
from romtime.conventions import (
    BDF,
    FIG_KWARGS,
    Domain,
    MassConservation,
    PistonParameters,
)
from romtime.fom import OneDimensionalBurgers
from romtime.parameters import get_uniform_dist
from romtime.problems.gcl import define_constant_solution
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

fenics.set_log_level(100)

plt.set_cmap("viridis")


def plot_mass_conservation(ts, mass_change, outflow, title, save):

    fig, (ax_mass, ax_error) = plt.subplots(
        nrows=2, ncols=1, sharex=True, gridspec_kw={"hspace": 0.35}
    )

    ax_mass.plot(
        ts,
        mass_change,
        label="$\\frac{d}{dt} \\int \\rho dx$",
    )
    ax_mass.plot(
        ts,
        outflow,
        linestyle="--",
        label="Outflow $(\\rho(0,t)u(0,t))$",
    )
    ax_mass.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, -0.175))
    ax_mass.grid(True)
    ax_mass.set_title(title)
    ax_mass.set_ylabel("Flow")

    #  -------------------------------------------------------------------------
    #  Mass error
    mc = mass_change - outflow
    mc = np.log10(np.abs(mc))

    ax_error.plot(
        ts,
        mc,
        color="black",
    )

    mc_mean = np.mean(mc)
    ax_error.axhline(mc_mean, 0.1, 0.9, linestyle="dashdot", color="red", alpha=0.5)

    ax_error.grid(True)
    ax_error.set_xlabel("t (s)")
    ax_error.set_ylabel("Error (log10)")

    plt.savefig(save + ".png", **FIG_KWARGS)
    plt.close()


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
    # mach_min = support[DELTA]["min"] * support[OMEGA]["min"] / support[A0]["max"]
    mach_min = 0.35

    # Maximum input into the system, minimum linearity
    # forcing_max = support[DELTA]["max"] * support[OMEGA]["max"] / support[A0]["min"]
    mach_max = 0.4

    print(f"forcing : (min, max) = {mach_min}, {mach_max}")

    space = np.linspace(start=mach_min, stop=mach_max, num=num + 1)

    return space


def create_solver(L, nx, nt, tf, grid_base, which=None):
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
    ) = define_constant_solution(L, nx, tf, nt, which)

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
}

with open("grid_params.json", mode="w") as fp:
    ujson.dump(grid_params, fp)

# -----------------------------------------------------------------------------
# Space-Time Domain
NX = 5e1
NT = 5e2

# Create data structures
domain = {
    Domain.L0: 1.0,
    Domain.NX: int(NX),
    Domain.NT: int(NT),
    Domain.T: 0.75,
}

which = "moving"

solver = create_solver(
    L=domain[Domain.L0],
    nt=domain[Domain.NT],
    nx=domain[Domain.NX],
    tf=domain[Domain.T],
    grid_base=grid_params,
    which=which,
)

grid = {
    "a0": get_uniform_dist(**grid_params["a0"]),
    "omega": get_uniform_dist(**grid_params["omega"]),
    "delta": get_uniform_dist(**grid_params["delta"]),
}


N_SAMPLES = 1

mu = build_sampling_space(grid=grid, num=N_SAMPLES, rnd=42)
mu = mu[0]

pprint(mu)

with open(f"mu.json", mode="w") as fp:
    ujson.dump(mu, fp)


NXs = [5e1, 1e2, 2e2, 3e2, 5e2, 1e3]
for nx in NXs:

    nx = int(nx)

    solver.domain[Domain.NX] = nx
    solver.setup()
    solver.update_parametrization(mu)
    solver.BDF_SCHEME = BDF.ONE
    solver.solve()

    name_probes = f"probes_FOM_nx_{nx}_{which}"
    name_solutions = f"solutions_FOM_nx_{nx}_{which}"

    solver.dump_solutions(name_solutions)
    probes = solver.save_probes(name=name_probes + ".csv")

    plot_probes(probes, save=name_probes)