import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from helpers.configs import METRICS_DIR

METRICS = ["flow_mm_s", "flow_rel_s", "shape_change", "area_cv"]
MIN_SIGNAL_RATIO = 5
MIN_WILCOXON_N = 6

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


def ratio_to(mean, base, conds):
    ref = mean[mean["condition"] == base].drop(columns="condition")
    out = mean[mean["condition"].isin(conds)].merge(
        ref, on=["cam", "plant"], suffixes=("", "_base")
    )
    for m in METRICS:
        out[m] = out[m] / out.pop(m + "_base")
    out.insert(3, "baseline", base)
    return out


def tests(ratios):
    rows = []
    for (cam, base, cond), g in ratios.groupby(["cam", "baseline", "condition"]):
        for m in METRICS:
            log = np.log(g[m].dropna())
            n = len(log)
            rows.append(
                {
                    "cam": cam,
                    "comparison": f"{cond} / {base}",
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
    counts = df.groupby(["cam", "condition"]).size().unstack(fill_value=0)
    show("clips per camera and condition", counts)

    mean = df.groupby(["cam", "plant", "condition"], as_index=False)[METRICS].mean()
    flow = mean.pivot_table(index=["cam", "plant"], columns="condition", values="flow_mm_s")
    show("net plant flow, mm/s (mean of repeats)", flow)

    ratios = pd.concat(
        [
            ratio_to(mean, "ref", ["hover_30", "hover_50", "wind_ref"]),
            ratio_to(mean, "hover_50", ["hover_30"]),
        ]
    )
    show(
        "per plant: condition / baseline (x, 1.0 = no difference)",
        ratios.set_index(["cam", "plant", "condition", "baseline"]),
    )
    stats = tests(ratios)
    show("across plants", stats.set_index(["cam", "comparison", "metric"]))

    out = METRICS_DIR / "summary.json"
    summary = {"qa": flags, "ratios": records(ratios), "tests": records(stats)}
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
