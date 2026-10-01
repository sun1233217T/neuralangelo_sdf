"""Plateau / refine-timing analysis from loss_curve.csv.

Reads the CSV(s) written by the trainer (loss_curve.csv plus
loss_curve_<iter>.csv fragments created when the column set changes at a
refine), reconstructs the full loss history, and for every refine interval
reports when the smoothed render loss entered a plateau.

Usage:
    python projects/Bspline-Neus/scripts/plateau_analysis.py LOGDIR [--tol 0.01] [--window 2000]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def load_history(logdir):
    """Concatenate loss_curve*.csv fragments in iteration order."""
    files = glob.glob(os.path.join(logdir, "loss_curve*.csv"))

    def start_iter(path):
        m = re.search(r"loss_curve_(\d+)\.csv$", path)
        return int(m.group(1)) if m else 0

    dfs = []
    for f in sorted(files, key=start_iter):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True).sort_values("iter").reset_index(drop=True)
    return df


def smooth(y, window_iters, iters):
    """Boxcar smoothing over a fixed iteration window."""
    n = max(1, int(round(window_iters / np.median(np.diff(iters)))))
    kernel = np.ones(n) / n
    return np.convolve(y, kernel, mode="same")


def plateau_time(iters, y_smooth, tol, window_iters):
    """First iter where relative improvement over the trailing window < tol."""
    n = max(1, int(round(window_iters / np.median(np.diff(iters)))))
    for i in range(n, len(iters)):
        past, now = y_smooth[i - n], y_smooth[i]
        if past > 0 and (past - now) / past < tol:
            return iters[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_history(args.logdir)
    iters = df["iter"].to_numpy()
    print(f"[plateau] {len(df)} rows, iters {iters[0]}..{iters[-1]}")
    print(f"[plateau] columns: {list(df.columns)}")

    # refine events from the training log
    log_file = None
    for cand in glob.glob(os.path.join("logs", os.path.basename(args.logdir) + ".log")):
        log_file = cand
    refines = []
    if log_file:
        with open(log_file, errors="ignore") as f:
            for line in f:
                m = re.search(r"refinement at iter (\d+)", line)
                if m:
                    refines.append(int(m.group(1)))
    print(f"[plateau] refine events: {refines}")

    loss_col = "loss_render" if "loss_render" in df else "loss_total"
    y = smooth(df[loss_col].to_numpy(), args.window, iters)

    bounds = [0] + refines + [iters[-1]]
    print(f"\ninterval analysis (tol={args.tol}, window={args.window}):")
    print(f"{'level':<8}{'start':>8}{'end':>8}{'plateau@':>10}{'time_to_plateau':>18}{'budget_used':>13}")
    for lv in range(len(bounds) - 1):
        a, b = bounds[lv], bounds[lv + 1]
        mask = (iters >= a) & (iters < b)
        if mask.sum() < 5:
            continue
        pt = plateau_time(iters[mask], y[mask], args.tol, args.window)
        if pt is None:
            print(f"L{lv:<7}{a:>8}{b:>8}{'—':>10}{'never':>18}{'—':>13}")
        else:
            used = (pt - a) / (b - a) if b > a else float("nan")
            print(f"L{lv:<7}{a:>8}{b:>8}{pt:>10}{pt - a:>15} it{used:>12.0%}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        axes[0].plot(iters, y, lw=1)
        axes[0].set_yscale("log")
        axes[0].set_title(f"{loss_col} (smoothed)")
        grad_cols = [c for c in df.columns if c.startswith("grad_rms_neural_sdf")]
        for c in grad_cols:
            axes[1].plot(iters, df[c].rolling(10, min_periods=1).median(),
                         lw=1, label=c.replace("grad_rms_neural_sdf_", ""))
        axes[1].set_yscale("log")
        axes[1].set_title("per-level grad RMS (SDF)")
        axes[1].legend()
        for ax in axes:
            for r in refines:
                ax.axvline(r, color="r", ls="--", alpha=0.5)
            ax.set_xlabel("iter")
        out = args.out or os.path.join(args.logdir, "plateau_analysis.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"\n[plateau] plot saved to {out}")
    except ImportError:
        print("[plateau] matplotlib unavailable, skipping plot")


if __name__ == "__main__":
    main()
