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
    ProblemType,
    Stage,
    Treewalk,
)
from romtime.utils import singular_to_energy
from tqdm import tqdm

sns.set_theme(context="paper", palette="colorblind")


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
    fom = comparison[ProblemType.FOM]

    comparison = comparison.drop([PISTON, ProblemType.FOM], axis=1)

    fig, ax = plt.subplots()

    ax.plot(comparison[ProblemType.SROM], label="SROM")
    ax.plot(comparison[ProblemType.ROM], label="ROM")
    ax.plot(fom, label="FOM", linestyle="-.", alpha=0.75)
    ax.plot(piston, label=PISTON, color="purple", linestyle="--", alpha=0.5)
    ax.grid(True)
    ax.legend()
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

# # -----------------------------------------------------------------------------
# # Sigmas
# with open("summary_sigmas.pkl", mode="rb") as fp:
#     summary_sigmas = pickle.load(fp)

# sigmas_nonlinear = summary_sigmas[OperatorType.NONLINEAR][Treewalk.SPECTRUM_MU]
# sigmas_rb = summary_sigmas[OperatorType.REDUCED_BASIS][Treewalk.SPECTRUM_MU]

# sigmas = pd.Series(sigmas_nonlinear, name=OperatorType.NONLINEAR)
# sigmas = pd.concat([sigmas, pd.Series(sigmas_rb, name="RB")], axis=1)
# sigmas = sigmas.apply(np.log10)
# sigmas.to_csv("sigmas.csv")

# sigmas = sigmas.loc[:65, "RB"]
# ax_sigmas = sigmas.plot()
# ax_sigmas.grid(True)
# ax_sigmas.set_ylabel("$\\sigma_i$")
# ax_sigmas.set_xlabel("i-th basis element")
# plt.savefig("sigmas_rb.png", **FIG_KWARGS)
# plt.close()

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
# # Errors (aggregation)
# paths = list(Path(".").glob("errors_estimator*.pkl"))
# sigmas = pd.read_csv("sigmas.csv", index_col=0)["RB"].squeeze()

# # Normalized sigmas
# sigmas_hat = np.log10(sigmas)
# energies = singular_to_energy(sigmas=sigmas)

# ONLINE = Stage.ONLINE

# desc = "(ROM ERRORS)"

# results = []
# for path in tqdm(paths, desc=desc, total=len(paths)):

#     with open(path, mode="rb") as fp:
#         results_sacrificial = pickle.load(fp)

#     stem = path.stem.split("_")
#     n_rom = int(stem[3])
#     n_srom = int(stem[5])

#     payload = results_sacrificial[ONLINE]

#     errors = payload[0]
#     errors = pd.DataFrame(errors)

#     # -------------------------------------------------------------------------
#     # ROM
#     idx_rom = n_rom - 1
#     energy = energies[idx_rom]
#     sigma_hat = sigmas_hat[idx_rom]

#     N_error = errors.shape[0]
#     error = np.linalg.norm(errors[Errors.ROM])
#     error /= np.sqrt(N_error)

#     result = {
#         "error": error,
#         "N": n_rom,
#         "sigma_hat": sigma_hat,
#         "energy": energy,
#     }

#     results.append(result)

#     # -------------------------------------------------------------------------
#     # SROM
#     idx_rom = n_srom - 1
#     energy = energies[idx_rom]
#     sigma_hat = sigmas_hat[idx_rom]

#     error = np.linalg.norm(errors[Errors.SACRIFICIAL])
#     error /= np.sqrt(N_error)

#     result = {
#         "error": error,
#         "N": n_srom,
#         "sigma_hat": sigma_hat,
#         "energy": energy,
#     }
#     results.append(result)


# results = pd.DataFrame(results)
# results = results.drop_duplicates(subset="N")
# results = results.sort_values(by="N")

# fig, (left, right) = plt.subplots(nrows=1, ncols=2, sharey=True)

# options = dict(linestyle="--", marker=".", markersize=8)
# left.plot(results["N"], results["error"], **options)
# left.set_yscale("log")
# left.grid(True)
# left.set_xlabel("N")
# left.set_ylabel("ROM L2 Error")

# right.plot(results["energy"], results["error"], **options)
# right.set_yscale("log")
# right.grid(True)
# right.set_xlabel("Energy")

# plt.savefig("error_decay.png", **FIG_KWARGS)
# plt.close()

# # -----------------------------------------------------------------------------
# # Error Estimator (aggregation)
# paths = list(Path(".").glob("errors_estimator*.pkl"))
# sigmas = pd.read_csv("sigmas.csv", index_col=0)["RB"].squeeze()

# # Normalized sigmas
# sigmas_hat = np.log10(sigmas)
# energies = singular_to_energy(sigmas=sigmas)

# ONLINE = Stage.ONLINE

# desc = "(ERROR ESTIMATOR)"

# results = []
# for path in tqdm(paths, desc=desc, total=len(paths)):

#     with open(path, mode="rb") as fp:
#         results_sacrificial = pickle.load(fp)

#     stem = path.stem.split("_")
#     n_rom = int(stem[3])
#     n_srom = int(stem[5])

#     payload = results_sacrificial[ONLINE]

#     errors = payload[0]
#     errors = pd.DataFrame(errors)
#     # errors = errors.mean()

#     # -------------------------------------------------------------------------
#     # ROM
#     idx_rom = n_rom - 1
#     idx_srom = n_srom - 1

#     energy_rom = energies[idx_rom]
#     energy_srom = energies[idx_srom]
#     delta_energy = energy_srom - energy_rom

#     delta_n = n_srom - n_rom

#     delta_error = errors[Errors.ROM] - errors[Errors.ESTIMATOR]
#     N_error = len(delta_error)
#     delta_error = np.linalg.norm(delta_error)
#     delta_error /= np.sqrt(N_error)

#     result = {
#         "N": n_rom,
#         "N-SROM": n_srom,
#         "delta-N": delta_n,
#         "delta-energy": delta_energy,
#         "delta-error": delta_error,
#     }

#     results.append(result)


# results = pd.DataFrame(results)
# results = results.sort_values(by=["N", "N-SROM"]).reset_index(drop=True)

# groups = results.groupby(by="N").groups

# fig, (left, right) = plt.subplots(nrows=1, ncols=2, sharey=True)

# options = dict(linestyle="--", marker=".", markersize=8)

# for N, indices in groups.items():

#     data = results.loc[indices]

#     left.plot(
#         data["delta-N"],
#         data["delta-error"],
#         **options,
#         label=f"N={str(N)}",
#     )

#     right.plot(
#         data["delta-energy"],
#         data["delta-error"],
#         **options,
#         label=f"N={str(N)}",
#     )

# left.legend()
# left.set_yscale("log")
# left.grid(True)
# left.set_xlabel("$\\Delta N$")
# left.set_ylabel("Estimator Accuracy")

# right.set_yscale("log")
# right.grid(True)
# right.set_xlabel("$\\Delta \, Energy$")

# plt.savefig("estimator_accuracy.png", **FIG_KWARGS)
# plt.close()


# -----------------------------------------------------------------------------
# Errors (timewise)
paths = list(Path(".").glob("errors_estimator*.pkl"))

ONLINE = Stage.ONLINE

desc = "(ERRORS - TIMESERIES)"

ts = np.linspace(0, 1.0, 500)

for path in tqdm(paths, desc=desc, total=len(paths)):

    with open(path, mode="rb") as fp:
        results_sacrificial = pickle.load(fp)

    stem = path.stem.split("_")
    n_rom = stem[3]
    n_srom = stem[5]

    payload = results_sacrificial[ONLINE]

    for idx_mu, errors in tqdm(payload.items(), leave=False):

        errors = pd.DataFrame(errors)

        errors["index"] = ts
        errors = errors.set_index("index")

        estimator = errors[Errors.ESTIMATOR].copy()
        errors = errors.drop(Errors.ESTIMATOR, axis=1)

        ax = errors.plot(grid=True, logy=True)
        ax.plot(estimator.index, estimator, label=Errors.ESTIMATOR, linestyle="--")

        ax.legend()
        title = f"N-ROM = {n_rom}, N-SROM = {n_srom}"
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("$L_2$ Error (FOM vs. ROM)")

        figname = f"error_estimation_rom_{n_rom}_srom_{n_srom}.png"
        plt.savefig(figname, **FIG_KWARGS)

        plt.close()


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
