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

sigmas_nonlinear = summary_sigmas[OperatorType.NONLINEAR][Treewalk.SPECTRUM_MU]
sigmas_rb = summary_sigmas[OperatorType.REDUCED_BASIS][Treewalk.SPECTRUM_MU]

sigmas = pd.Series(sigmas_nonlinear, name=OperatorType.NONLINEAR)
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

_sigmas = sigmas[OperatorType.NONLINEAR]
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
# Compare errors in time
path_errors = Path(".").glob("psi_10*trucanted_errors.csv")
path_full = Path("psi_10_full_errors.csv")
errors = pd.DataFrame()
for path in path_errors:

    df = pd.read_csv(path, index_col=0)

    mean = df.mean(axis=1).squeeze()
    truncated_modes = int(path.stem.split("_")[-3])
    mean.name = 69 - truncated_modes

    errors = pd.concat([errors, mean], axis=1)

# df = pd.read_csv(path_full, index_col=0)
# mean = df.mean(axis=1).squeeze()
# mean.name = 69
# errors = pd.concat([errors, mean], axis=1)

errors = errors.mean(axis=0)
errors = errors.sort_index()

errors.plot.bar(logy=True, grid=True)
plt.xlabel("Number of Basis Elements")
plt.ylabel("$L_2$ Error")
plt.title("Nonlinear MDEIM")
# plt.show()
plt.savefig("nonlinear_error_decay.png", **FIG_KWARGS)
plt.close()
