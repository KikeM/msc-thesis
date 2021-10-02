import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks

from pathlib import Path

NAME = "linearity"

csv_files = Path("probes/").glob("*.csv")
csv_files = list(csv_files)

results = {}
for csv in tqdm(csv_files):

    if csv.stem == NAME:
        continue

    idx = csv.stem.split("_")[-1]
    stage = csv.stem.split("_")[1]

    df = pd.read_csv(csv, index_col="timesteps")

    probe_L = df["L"]
    probe_piston = df["0.0"]

    peaks_L = find_peaks(probe_L.abs().values)[0]
    peaks_piston = find_peaks(probe_piston.abs().values)[0]

    assert probe_L[probe_L.index[peaks_L[0]]] > 0.0
    assert probe_piston[probe_piston.index[peaks_piston[1]]] > 0.0

    T = probe_L.index[peaks_L[1]] - probe_L.index[peaks_L[0]]
    T0 = probe_piston.index[peaks_piston[3]] - probe_piston.index[peaks_piston[2]]

    linearity = T / T0
    name = "-".join([stage, idx])
    results[name] = linearity

results = pd.Series(results)

results.name = "linearity"
results.index.name = "sample"
results.to_csv(NAME + ".csv")
