from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

here = Path(__file__)
path_data_bdf1 = Path(
    here.parents[3]
    / "results"
    / "piston"
    / "consistency"
    / "mass_conservation_consistency_bdf1.csv"
)
path_data_bdf2 = Path(
    here.parents[3]
    / "results"
    / "piston"
    / "consistency"
    / "mass_conservation_consistency_bdf2.csv"
)


# -----------------------------------------------------------------------------
# BDF 1
errors_bdf1 = pd.read_csv(path_data_bdf1)
x = errors_bdf1["dt"]
l2 = errors_bdf1["L2"]

plt.plot(np.log10(x), np.log10(l2), marker=".", label="BDF-1")

# -----------------------------------------------------------------------------
# BDF 2
errors_bdf2 = pd.read_csv(path_data_bdf2)
x = errors_bdf2["dt"]
l2 = errors_bdf2["L2"]

plt.plot(np.log10(x), np.log10(l2), marker=".", label="BDF-2")

# -----------------------------------------------------------------------------
# Plot arrangements
plt.legend()
plt.grid(True)
plt.xlabel("$log_{10} dt$")
plt.ylabel("Mass Conservation L2 Norm (log 10)")
plt.title("Time Consistency Check")
plt.show()
