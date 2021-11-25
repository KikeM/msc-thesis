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
from romtime.utils import singular_to_energy, read_json
from tqdm import tqdm
from itertools import chain

sns.set_theme(context="paper")


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

sigmas_nonlinear = summary_sigmas[OperatorType.TRILINEAR][Treewalk.SPECTRUM_MU]
sigmas_rb = summary_sigmas[OperatorType.REDUCED_BASIS][Treewalk.SPECTRUM_MU]

sigmas = pd.Series(sigmas_nonlinear, name=OperatorType.TRILINEAR)
sigmas = pd.concat([sigmas, pd.Series(sigmas_rb, name="RB")], axis=1)
sigmas.to_csv("sigmas.csv")
# sigmas = sigmas.apply(np.log10)
sigmas.to_csv("sigmas_log10.csv")

_sigmas = sigmas["RB"]
ax_sigmas = _sigmas.plot(logy=True)
ax_sigmas.grid(True)
ax_sigmas.set_ylabel("$\\sigma_i$")
ax_sigmas.set_xlabel("i-th basis element")
ax_sigmas.set_title("SVD Spectrum Decay")
plt.savefig("sigmas_rb.png", **FIG_KWARGS)
plt.close()

_sigmas = sigmas[OperatorType.TRILINEAR]
ax_sigmas = _sigmas.plot(logy=True)
ax_sigmas.grid(True)
ax_sigmas.set_ylabel("$\\sigma_i$")
ax_sigmas.set_xlabel("i-th basis element")
ax_sigmas.set_title("SVD Spectrum Decay")
plt.savefig("sigmas_nonlinear.png", **FIG_KWARGS)
plt.close()


sigmas = sigmas.rename(columns={"nonlinear": "Nonlinear MDEIM", "RB": "Reduced Basis"})
ax_combined = sigmas.plot(logy=True, logx=True)
ax_combined.axvline(69, ymax=0.5, linestyle="--", alpha=0.5)
ax_combined.grid(True)
ax_combined.legend()
ax_combined.set_ylabel("$\\sigma_i$")
ax_combined.set_xlabel("i-th basis element")
ax_combined.set_title("SVD Spectrum Decay")
plt.savefig("sigmas_problem.png", **FIG_KWARGS)
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

# ---------------------------------------------------------
# Compare errors in time (average)
path_errors = Path(".").glob("psi_10*trucanted_errors.csv")
path_full = Path("psi_10_full_errors.csv")
errors = pd.DataFrame()
for path in path_errors:

    df = pd.read_csv(path, index_col=0)

    mean = df.mean(axis=1).squeeze()
    truncated_modes = int(path.stem.split("_")[-3])
    mean.name = 70 - truncated_modes

    errors = pd.concat([errors, mean], axis=1)

df = pd.read_csv(path_full, index_col=0)
mean = df.mean(axis=1).squeeze()
mean.name = 70
errors = pd.concat([errors, mean], axis=1)

errors = errors.mean(axis=0)
errors = errors.sort_index()

options = dict(linestyle="--", marker=".", markersize=8)
errors.plot(logy=True, grid=True, **options)
plt.xlabel("Number of Basis Elements")
plt.ylabel("$L_2$ Error")
plt.title("Nonlinear MDEIM (Reconstruction)")
# plt.show()
plt.savefig("nonlinear_error_decay.png", **FIG_KWARGS)
plt.close()


# ---------------------------------------------------------
# Compare errors in time
# Load mu space
mu_space = read_json("mu_space_mdeim_certification.json")
mu_space = pd.DataFrame(mu_space)

print(mu_space.round(3).to_latex())

path_errors = Path(".").glob("psi_10*trucanted_errors.csv")
path_full = Path("psi_10_full_errors.csv")
errors = pd.DataFrame()
for path in chain(path_errors, [path_full]):

    if "full" in path.stem:
        truncated_modes = 0
    else:
        truncated_modes = int(path.stem.split("_")[-3])

    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.astype(int)
    _map = mu_space["piston_mach"].round(3).squeeze()
    df = df.rename(columns=_map)

    time_average = df.mean(axis=0)
    time_average.name = 70 - truncated_modes

    errors = pd.concat([errors, time_average], axis=1)

errors = errors.sort_index(axis=1).sort_index(axis=0).T
cols = errors.columns[::-2]
errors = errors[cols]
errors = errors.add_prefix("$u_p = ").add_suffix("$")

options = dict(linestyle="--", marker=".", markersize=8)
errors.plot(logy=True, grid=True, **options)
plt.xlabel("Number of Basis Elements")
plt.ylabel("$L_2$ Error")
plt.title("Nonlinear MDEIM (Reconstruction)")
plt.savefig("nonlinear_error_decay_by_parameter.png", **FIG_KWARGS)
plt.close()
