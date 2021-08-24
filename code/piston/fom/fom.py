import fenics
import numpy as np
import matplotlib.pyplot as plt
from romtime.conventions import FIG_KWARGS, Domain, MassConservation, PistonParameters
from romtime.fom import OneDimensionalBurgers
from romtime.problems.piston import define_piston_problem

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


def plot_probes(probes, save):

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

    plt.xlabel("t (s)")
    plt.savefig(save + ".png", **FIG_KWARGS)
    plt.close()


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
        "min": 9.0,
        "max": 15.0,
    },
    PistonParameters.OMEGA: {
        "min": 15.0,
        "max": 25.0,
    },
    PistonParameters.DELTA: {
        "min": 0.1,
        "max": 0.15,
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

WHICH = "rest"
# -----------------------------------------------------------------------------
# Space-Time Domain
NX = 1e3
NT = 5e2

# Create data structures
domain = {
    Domain.L0: 1.0,
    Domain.NX: int(NX),
    Domain.NT: int(NT),
    Domain.T: 1.0,
}

solver = create_solver(
    L=domain[Domain.L0],
    nt=domain[Domain.NT],
    nx=domain[Domain.NX],
    tf=domain[Domain.T],
    grid_base=grid_params,
    which=WHICH,
)

mu = {
    "a0": 14.095488100907177,
    "alpha": 1e-6,
    "delta": 0.12211862909242338,
    "gamma": 1.4,
    "omega": 23.31467709771428,
    "forcing": 0.2019906217168987,
}

solver.update_parametrization(mu)
solver.solve()

# -----------------------------------------------------------------------------
# Check solution
print("Postprocessing ...")
name_mass = f"mass_conservation_FOM_{WHICH}"
name_probes = f"probes_FOM_{WHICH}"

mass_data = solver.save_mass_conservation(name=name_mass + ".csv")
probes = solver.save_probes(name=name_probes + ".csv")

print("Plot mass ...")
plot_mass_conservation(
    ts=mass_data[MassConservation.TIMESTEPS],
    mass_change=mass_data[MassConservation.MASS_CHANGE],
    outflow=mass_data[MassConservation.OUTFLOW],
    title=f"Mass Conservation ({WHICH})",
    save=name_mass,
)

print("Plot probes ...")
plot_probes(probes, name_probes)