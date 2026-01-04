"""
Statistical analysis and visualisation of simulation results.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def load_results(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(results: list) -> pd.DataFrame:
    agg = defaultdict(lambda: {"n": 0, "loc": 0, "overrides": 0, "end_time": 0})
    for r in results:
        key = (r["risk"], r["authority"], r["tug"])
        agg[key]["n"] += 1
        agg[key]["loc"] += r["loss_of_control"]
        agg[key]["overrides"] += r["master_overrides"]
        agg[key]["end_time"] += r["end_time"]

    rows = []
    for (risk, auth, tug), d in agg.items():
        rows.append({
            "risk": risk,
            "authority": auth,
            "tug": tug,
            "n_runs": d["n"],
            "loss_rate_pct": round(100 * d["loc"] / d["n"], 1),
            "mean_overrides": round(d["overrides"] / d["n"], 2),
            "mean_end_time": round(d["end_time"] / d["n"], 2),
        })
    
    df = pd.DataFrame(rows)
    return df.sort_values(["risk", "authority", "tug"]).reset_index(drop=True)


def create_pivot(df: pd.DataFrame, value_col: str) -> dict:
    pivots = {}
    for risk in df["risk"].unique():
        sub = df[df["risk"] == risk]
        pivots[risk] = sub.pivot(index="authority", columns="tug", values=value_col)
    return pivots


def plot_grouped_bar(pivot: pd.DataFrame, title: str, ylabel: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax, rot=0, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Authority Mode")
    ax.legend(title="Tug Regime", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8, padding=2)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to sim_results.json")
    parser.add_argument("--out", default="output", help="Output directory")
    args = parser.parse_args()

    results = load_results(Path(args.input))
    df = aggregate(results)
    
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(outdir / "scenario_aggregate.csv", index=False)

    loss_pivots = create_pivot(df, "loss_rate_pct")
    override_pivots = create_pivot(df, "mean_overrides")

    for risk, pivot in loss_pivots.items():
        pivot.to_csv(outdir / f"loss_{risk}.csv")
        plot_grouped_bar(
            pivot,
            f"Loss-of-Control Rate ({risk.capitalize()} Risk)",
            "Loss Rate (%)",
            outdir / f"fig_loss_{risk}"
        )

    for risk, pivot in override_pivots.items():
        pivot.to_csv(outdir / f"overrides_{risk}.csv")
        plot_grouped_bar(
            pivot,
            f"Mean Master Overrides ({risk.capitalize()} Risk)",
            "Overrides",
            outdir / f"fig_overrides_{risk}"
        )

    print(f"Analysis complete. Results saved to {outdir}/")
    print("\nSummary by Risk Level:")
    print(df.groupby("risk").agg({
        "loss_rate_pct": "mean",
        "mean_overrides": "mean"
    }).round(1))


if __name__ == "__main__":
    main()
