import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from helpers.configs import METRICS_DIR

METRICS = ["flow_mm_s", "flow_rel_s", "shape_change", "area_cv"]
MIN_SIGNAL_RATIO = 5
MIN_WILCOXON_N = 6
WIND = {False: "side wind off", True: "side wind on"}


def comparisons():
    """(name, numerator, denominator), each side a (condition, side_wind) cell.
    Downwash: hover vs reference at the same side wind. Side wind: on vs off, same drone state."""
    out = []
    for wind in (False, True):
        for hover in ("hover_30", "hover_50"):
            out.append((f"{hover} / ref, {WIND[wind]}", (hover, wind), ("ref", wind)))
        out.append((f"hover_30 / hover_50, {WIND[wind]}", ("hover_30", wind), ("hover_50", wind)))
    for cond in ("ref", "hover_30", "hover_50"):
        out.append((f"side wind on / off, {cond}", (cond, True), (cond, False)))
    return out

def records(df):
    return json.loads(df.to_json(orient="records"))  # NaN -> null


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.round(2).to_string())


def qa(df):
    name = df["clip_id"] + "_" + df["cam"]
    hover = df["condition"].str.startswith("hover")
    checks = {
        "empty mask frames, re-annotate": df["empty_frames"] > 0,
        "phone clip without ArUco scale": (df["cam"] == "phone") & df["mm_per_px"].isna(),
        f"hover flow < {MIN_SIGNAL_RATIO}x background residual": hover
        & (df["signal_ratio"] < MIN_SIGNAL_RATIO),
    }
    flags = {check: name[bad].tolist() for check, bad in checks.items()}
    print("\n=== QA (flagged clips are still included) ===")
    for check, clips in flags.items():
        print(f"{check}: {', '.join(clips) or 'none'}")
    return flags


def ratio(mean, name, num, den):
    """Per (cam, plant): numerator cell / denominator cell, for every metric."""
    def cell(cond, wind):
        rows = mean[(mean["condition"] == cond) & (mean["side_wind"] == wind)]
        return rows.drop(columns=["condition", "side_wind"])

    out = cell(*num).merge(cell(*den), on=["cam", "plant"], suffixes=("", "_base"))
    for m in METRICS:
        out[m] = out[m] / out.pop(m + "_base")
    out.insert(0, "comparison", name)
    return out


def tests(ratios):
    rows = []
    for (cam, comparison), g in ratios.groupby(["cam", "comparison"], sort=False):
        for m in METRICS:
            log = np.log(g[m].dropna())
            n = len(log)
            rows.append(
                {
                    "cam": cam,
                    "comparison": comparison,
                    "metric": m,
                    "plants": n,
                    "increased": int((log > 0).sum()),
                    "median_x": float(np.exp(log.median())),
                    "wilcoxon_p": wilcoxon(log).pvalue if n >= MIN_WILCOXON_N else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main():
    df = pd.read_json(METRICS_DIR / "clips.json", orient="records")
    flags = qa(df)
    counts = df.groupby(["cam", "side_wind", "condition"]).size().unstack(fill_value=0)
    show("clips per camera, side wind and condition", counts)
    # mean over runs in the same cell (only perilla has two: wind-off references at 30 and 50 cm)
    mean = df.groupby(["cam", "plant", "condition", "side_wind"], as_index=False)[METRICS].mean()
    flow = mean.pivot_table(index=["cam", "plant"], columns=["side_wind", "condition"], values="flow_mm_s")
    show("net plant flow, mm/s (columns: side wind, condition)", flow)

    ratios = pd.concat([ratio(mean, *c) for c in comparisons()])
    show(
        "per plant: ratio (x, 1.0 = no difference)",
        ratios.set_index(["cam", "comparison", "plant"]),
    )
    stats = tests(ratios)
    show("across plants", stats.set_index(["cam", "comparison", "metric"]))

    out = METRICS_DIR / "summary.json"
    summary = {"qa": flags, "ratios": records(ratios), "tests": records(stats)}
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
