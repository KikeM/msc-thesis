import numpy as np
from romtime.conventions import FIG_KWARGS
import matplotlib.pyplot as plt
import pickle

from pathlib import Path


here = Path(__file__)
path_data_bdf1 = Path(
    here.parents[3]
    / "results"
    / "piston"
    / "bdf_schemes"
    / "comparison"
    / "nt100"
    / "mass_conservation_bdf_1.pkl"
)
path_data_bdf2 = Path(
    here.parents[3]
    / "results"
    / "piston"
    / "bdf_schemes"
    / "comparison"
    / "nt100"
    / "mass_conservation_bdf_2.pkl"
)

# mass_conservation = {"outflow": outflow, "mass": mass, "mass_change": mass_change}

ts = np.linspace(0.0, 1.0, 100)

plt.figure()

labels = ["BDF-1", "BDF-2"]
paths = [path_data_bdf1, path_data_bdf2]
styles = ["b", "r"]
for label, path, style in zip(labels, paths, styles):

    with open(path, mode="rb") as fp:
        mass_bdf = pickle.load(fp)

    plt.plot(ts, mass_bdf["mass_change"], c=style, label=label)
    plt.plot(ts, mass_bdf["outflow"], alpha=0.75, linestyle="--", c=style)

plt.legend()
plt.grid()
plt.xlabel("t (s)")
title = "Method Comparison - dt = {:.1e}"
title = title.format(mass_bdf["dt"])
plt.title(title)
plt.ylabel("Outflow (Dashed) vs. Mass Variations (Cont.)")
plt.show()
