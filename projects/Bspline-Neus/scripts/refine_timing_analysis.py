import glob
import os
import re

import numpy as np
import pandas as pd

import sys; logdir = sys.argv[1] if len(sys.argv) > 1 else "logs/Y6_replica_office0_s2_500k"

def start_iter(path):
    m = re.search(r"loss_curve_(\d+)\.csv$", path)
    return int(m.group(1)) if m else 0

dfs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(logdir, "loss_curve*.csv")), key=start_iter)]
df = pd.concat(dfs, ignore_index=True).sort_values("iter").reset_index(drop=True)

it = df["iter"].to_numpy()
y = df["loss_render"].rolling(50, min_periods=1).median().to_numpy()

# rolling saturation metric: relative gain over a trailing 10k window
W = 10000
n = int(round(W / np.median(np.diff(it))))  # rows per 10k
gain10k = np.full_like(y, np.nan)
gain10k[n:] = (y[:-n] - y[n:]) / y[:-n] * 100  # percent per 10k

refines = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
           else [10000, 30000, 70000, 150000])
bounds = [0] + refines + [int(it[-1])]
names = ["L0", "L1", "L2", "L3", "L4"]

print("first iter where trailing-10k gain < threshold:")
print(f"{'level':<6}" + "".join(f"{t:>8}" for t in [5.0, 3.0, 2.0, 1.0]) + f"{'at_refine':>12}")
for lv in range(len(bounds) - 1):
    a, b = bounds[lv], bounds[lv + 1]
    cells = []
    for thr in [5.0, 3.0, 2.0, 1.0]:
        mask = (it >= a + W) & (it < b) & (gain10k < thr)
        cells.append(f"{int(it[np.where(mask)[0][0]])//1000}k" if mask.any() else "never")
    # actual trailing gain at interval end
    mend = (it >= a) & (it < b - 1000)
    end_g = gain10k[mend][-1] if mend.any() else float("nan")
    print(f"{names[lv]:<6}" + "".join(f"{c:>8}" for c in cells) + f"{end_g:>11.2f}%")
