from os import name
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from romtime.conventions import (
    FIG_KWARGS,
    Errors,
    MassConservation,
    OperatorType,
    ProbeLocations,
    Stage,
    Treewalk,
)
from romtime.utils import singular_to_energy
from tqdm import tqdm

sns.set_theme(context="paper", palette="viridis")


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

    fig, (outflow_ax, middle_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.20},
    )

    ts = probes.index
    piston = probes["L"].squeeze()

    piston_fig_options = dict(linestyle="--", alpha=0.5, color="black")

    # ------------------------------------------------------------------------
    # Outflow
    outflow_ax.plot(ts, probes["0.0"])
    outflow_ax.plot(ts, piston.values, **piston_fig_options)
    outflow_ax.grid(True)
    outflow_ax.set_ylabel(f"$u(0, t)$")

    # ------------------------------------------------------------------------
    # Middle
    middle_ax.plot(ts, probes["0.5"])
    middle_ax.plot(ts, piston.values, **piston_fig_options)
    middle_ax.grid(True)
    middle_ax.set_ylabel(f"$u(L_0 / 2, t)$")

    middle_ax.set_xlabel("t (s)")
    plt.savefig(save + ".png", **FIG_KWARGS)
    plt.close()


def plot_probes_comparison(comparison, save):

    PISTON = ProbeLocations.PISTON

    piston = comparison[PISTON]

    comparison = comparison.drop([PISTON], axis=1)

    ax = comparison.plot()
    ax.plot(piston, label=PISTON, linestyle="--", color="black", alpha=0.5)
    ax.grid(True)
    ax.set_ylabel("$u$ (m/s)")
    ax.set_xlabel("$t$ (s)")

    plt.savefig(save + ".png", **FIG_KWARGS)
    plt.close()


# # -----------------------------------------------------------------------------
# # Energy / Sigmas
# with open("summary_energy.pkl", mode="rb") as fp:
#     energy = pickle.load(fp)

# energy_rb = energy[OperatorType.REDUCED_BASIS][Treewalk.ENERGY_MU]

# plt.plot(energy_rb)
# plt.grid(which="both", axis="both")
# plt.savefig("energy_rb.png", **FIG_KWARGS)
# plt.close()

# -----------------------------------------------------------------------------
# Sigmas
with open("summary_sigmas.pkl", mode="rb") as fp:
    summary_sigmas = pickle.load(fp)

sigmas_nonlinear = summary_sigmas[OperatorType.NONLINEAR][Treewalk.SPECTRUM_MU]
sigmas_rb = summary_sigmas[OperatorType.REDUCED_BASIS][Treewalk.SPECTRUM_MU]

sigmas = pd.Series(sigmas_nonlinear, name=OperatorType.NONLINEAR)
sigmas = pd.concat([sigmas, pd.Series(sigmas_rb, name="RB")], axis=1)
sigmas.to_csv("sigmas.csv")
sigmas = sigmas.apply(np.log10)
sigmas.to_csv("sigmas_log10.csv")

sigmas = sigmas.loc[:65, "RB"]
ax_sigmas = sigmas.plot()
ax_sigmas.grid(True)
ax_sigmas.set_ylabel("$\\sigma_i$")
ax_sigmas.set_xlabel("i-th basis element")
plt.savefig("sigmas_rb.png", **FIG_KWARGS)
plt.close()

# energy_nonlinear = singular_to_energy(sigmas_nonlinear)
# energy_rb = singular_to_energy(sigmas_rb)

# energy_nonlinear = energy_nonlinear[:65]
# energy_rb = energy_rb[:65]

# plt.plot(energy_nonlinear, label="Nonlinear")
# plt.plot(energy_rb, label="RB")
# plt.grid(True)
# plt.legend()
# plt.xlabel("i-th basis")
# plt.ylabel("Energy")
# plt.title("SVD Basis Reconstruction Capacity")
# plt.savefig("energy_rb.png", **FIG_KWARGS)
# plt.close()

# energy_nonlinear = pd.Series(energy_nonlinear, name="nonlinear")
# energy_rb = pd.Series(energy_rb, name="rb")

# pd.concat([energy_rb, energy_nonlinear], axis=1).to_csv("energy_rb.csv")

# # -----------------------------------------------------------------------------
# # Basis Elements
# with open("basis_srom.pkl", mode="rb") as fp:
#     basis_srom = pickle.load(fp)

# x = np.linspace(0, 1.0, num=basis_srom.shape[0])
# x = np.flip(x)

# basis_srom = pd.DataFrame(basis_srom)
# # scale = basis_srom.max(axis=0)
# # scale = basis_srom.max(axis=0)
# # scale = basis_srom.tail(1).squeeze().abs()
# # basis_srom = basis_srom.div(scale)
# # basis_srom["x"] = x

# N = 5
# plt.plot(x, basis_srom.iloc[:, :N].values)
# plt.grid(True)
# plt.legend(basis_srom.columns[:N])
# plt.xlabel("$x$")
# plt.ylabel("$\\psi_i(x)$")
# plt.show()
# plt.close()


# # -----------------------------------------------------------------------------
# # Errors
# with open("errors_estimator.pkl", mode="rb") as fp:
#     results_sacrificial = pickle.load(fp)

# # Validation
# VALIDATION = Stage.VALIDATION
# ONLINE = Stage.ONLINE

# desc = "(ROM ERRORS)"
# for stage in tqdm([VALIDATION, ONLINE], desc=desc):

#     if stage not in results_sacrificial:
#         continue

#     payload = results_sacrificial[stage]

#     for idx_mu, errors in tqdm(payload.items(), leave=False):

#         errors = pd.DataFrame(errors)

#         estimator = errors[Errors.ESTIMATOR].copy()
#         errors = errors.drop(Errors.ESTIMATOR, axis=1)

#         ax = errors.plot(grid=True, logy=True)
#         ax.plot(estimator.index, estimator, label=Errors.ESTIMATOR, linestyle="--")

#         ax.legend()
#         title = f"{idx_mu} - {stage}"
#         ax.set_title(title)

#         figname = f"{idx_mu}_{stage}.png"
#         plt.savefig(figname, **FIG_KWARGS)

#         plt.close()


# # -----------------------------------------------------------------------------
# # Mass conservation
# MASS_CONSERVATION_FILES = list(Path(".").glob("mass_conservation_*.csv"))

# desc = "(MASS CONSERVATION)"
# for file in tqdm(MASS_CONSERVATION_FILES, desc=desc):

#     results = pd.read_csv(file)

#     save = file.stem
#     title = results[MassConservation.WHICH].unique()[0].upper()

#     plot_mass_conservation(
#         ts=results[MassConservation.TIMESTEPS],
#         mass_change=results[MassConservation.MASS_CHANGE],
#         outflow=results[MassConservation.OUTFLOW],
#         title=title,
#         save=save,
#     )

# # -----------------------------------------------------------------------------
# # Probes
# FILES_PROBES = list(Path(".").glob("probes*.csv"))

# desc = "(PROBES)"
# for file in tqdm(FILES_PROBES, desc=desc):

#     probes = pd.read_csv(file, index_col=MassConservation.TIMESTEPS)
#     save = file.stem
#     plot_probes(probes, save)

# # -----------------------------------------------------------------------------
# # Probes with Model Comparison
# FILES_PROBES = list(Path(".").glob("*probes_comparison*.csv"))

# desc = "(MODEL COMPARISON)"
# for file in tqdm(FILES_PROBES, desc=desc):

#     probes = pd.read_csv(file, index_col=0)
#     save = file.stem
#     plot_probes_comparison(probes, save)
