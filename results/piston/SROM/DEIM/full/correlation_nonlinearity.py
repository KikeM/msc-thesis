import json
from pathlib import Path
import numpy as np

from scipy.integrate import trapz
import pandas as pd
import seaborn as sns
from romtime.conventions import MassConservation

import matplotlib.pyplot as plt

sns.set_theme()

# -----------------------------------------------------------------------------
# Correlate boundary condition with solution nonlinearity
with open("mu_space.json", mode="r") as fp:
    mu_space = json.load(fp)

linearity = pd.read_csv("linearity.csv", index_col="sample", squeeze=True)

space_off = pd.DataFrame(mu_space["offline"])
space_on = pd.DataFrame(mu_space["online"])

space_off = space_off.drop(["alpha", "gamma"], axis=1)
space_on = space_on.drop(["alpha", "gamma"], axis=1)

space_off["delta * w / a0"] = space_off["delta"] * space_off["omega"] / space_off["a0"]
space_on["delta * w / a0"] = space_on["delta"] * space_on["omega"] / space_on["a0"]

space_off["w / a0"] = space_off["omega"] / space_off["a0"]
space_on["w / a0"] = space_on["omega"] / space_on["a0"]

for idx, value in linearity.iteritems():

    idx_mu = int(idx.split("-")[-1])

    if "offline" in idx:
        space_off.loc[idx_mu, "linearity"] = value
    else:
        if idx_mu == 6:
            space_on = space_on.drop(idx_mu)
            continue
        space_on.loc[idx_mu, "linearity"] = value


# -----------------------------------------------------------------------------
# Compute mass deficit
csv_mass = Path("mass_conservation/").glob("*.csv")
csv_mass = list(csv_mass)

results_mass = {}
for csv in csv_mass:

    stem = csv.stem

    if "rom" in stem:
        continue

    # Load CSV
    mass_data = pd.read_csv(csv, index_col=MassConservation.TIMESTEPS)
    mass_data = mass_data.drop("Unnamed: 0", axis=1)

    # Get type
    type = mass_data[MassConservation.WHICH].unique()[0]
    mass_data = mass_data.drop(MassConservation.WHICH, axis=1)

    # Compute deficit average
    deficit = (
        mass_data[MassConservation.MASS_CHANGE] - mass_data[MassConservation.OUTFLOW]
    )
    deficit = deficit.abs()
    T = mass_data.index.max()
    t = mass_data.index
    error_integral = trapz(x=t, y=deficit.values)
    error_integral /= T
    error_integral = np.log10(error_integral)

    idx_mu = int(stem.split("_")[-1])
    if "validation" in stem:
        space_off.loc[idx_mu, "mass_error"] = error_integral
    else:
        if idx_mu == 6:
            continue
        space_on.loc[idx_mu, "mass_error"] = error_integral

all = pd.concat([space_off, space_on], axis=0)

sns.pairplot(all, y_vars=["linearity", "mass_error"])
plt.show()