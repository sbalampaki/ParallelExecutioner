#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSR = Path("/projects/sb2ea/csr")
RESULTS = Path("/projects/sb2ea/results/e0")
MANIFEST = Path("/projects/sb2ea/manifest")

WINDOW_ORDER = ["W6", "W12", "W24", "W48", "Wfull"]
PRIMARY = "W24"

P_TABLE = [16, 32, 64, 96]                                  # required by plan
P_CURVE = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192]    # for the figure

GATE_GINI = 0.40
GATE_P99_OVER_MEDIAN = 10.0

HILL_FRACTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
CHART_ID_MAX = None      # filled from itemid_dict.parquet when --by-source


def load_counts(window: str) -> np.ndarray:
    off = np.fromfile(CSR / window / "offsets.i64", dtype=np.int64)
    if off.size < 2:
        raise ValueError(f"{window}: offsets too short")
    return np.diff(off)


def gini(x: np.ndarray) -> float:
    """Gini of the work distribution. 0 = every stay equal, 1 = one stay has all."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    s = x.sum()
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * x).sum()) / (n * s) - (n + 1.0) / n)


def hill(x: np.ndarray, k: int) -> float:

    x = np.sort(np.asarray(x, dtype=np.float64))
    x = x[x > 0]
    n = x.size
    if k < 2 or k >= n:
        return float("nan")
    tail = x[n - k:]
    thresh = x[n - k - 1]
    if thresh <= 0:
        return float("nan")
    return float(1.0 / np.mean(np.log(tail / thresh)))


def describe(counts: np.ndarray) -> dict:
    c = counts.astype(np.float64)
    nz = c[c > 0]
    q = np.percentile(c, [50, 75, 90, 95, 99, 99.9])
    med = q[0]
    return {
        "n_stays": int(c.size),
        "n_records": int(c.sum()),
        "n_empty": int((c == 0).sum()),
        "mean": float(c.mean()),
        "std": float(c.std(ddof=1)) if c.size > 1 else 0.0,
        "cv": float(c.std(ddof=1) / c.mean()) if c.mean() > 0 else 0.0,
        "min": int(c.min()),
        "p50": float(q[0]), "p75": float(q[1]), "p90": float(q[2]),
        "p95": float(q[3]), "p99": float(q[4]), "p999": float(q[5]),
        "max": int(c.max()),
        "p99_over_median": float(q[4] / med) if med > 0 else float("inf"),
        "max_over_median": float(c.max() / med) if med > 0 else float("inf"),
        "gini": gini(c),
        "top1pct_share": float(np.sort(c)[-max(1, c.size // 100):].sum() / c.sum()),
        "hill": {f"k={f:g}": hill(nz, max(2, int(f * nz.size)))
                 for f in HILL_FRACTIONS},
    }

def max_load_block(counts: np.ndarray, p: int) -> int:
    """Contiguous blocks of equal stay count."""
    bounds = np.linspace(0, counts.size, p + 1).astype(np.int64)
    cum = np.concatenate(([0], np.cumsum(counts)))
    return int(np.max(cum[bounds[1:]] - cum[bounds[:-1]]))


def max_load_nzbalanced(counts: np.ndarray, p: int) -> int:
    """Contiguous blocks chosen so record counts are as equal as possible."""
    cum = np.cumsum(counts)
    total = int(cum[-1])
    targets = np.arange(1, p) * (total / p)
    cuts = np.searchsorted(cum, targets, side="left")
    bounds = np.concatenate(([0], cuts, [counts.size])).astype(np.int64)
    cum0 = np.concatenate(([0], cum))
    loads = cum0[bounds[1:]] - cum0[bounds[:-1]]
    return int(loads.max()) if loads.size else total


def max_load_cyclic(counts: np.ndarray, p: int) -> int:
    pad = (-counts.size) % p
    padded = np.concatenate([counts, np.zeros(pad, dtype=counts.dtype)])
    return int(padded.reshape(-1, p).sum(axis=0).max())


def max_load_greedy(counts: np.ndarray, p: int) -> int:
    """LPT: descending order, each task to the currently least-loaded worker."""
    import heapq
    order = np.sort(counts)[::-1]
    heap = [0] * p
    heapq.heapify(heap)
    for w in order:
        lo = heapq.heappop(heap)
        heapq.heappush(heap, lo + int(w))
    return int(max(heap))


STRATEGIES = {
    "block": max_load_block,
    "nzbalanced": max_load_nzbalanced,
    "cyclic": max_load_cyclic,
    "greedy": max_load_greedy,
}


def partition_rows(counts: np.ndarray, window: str, ps: list[int]) -> list[dict]:
    total = int(counts.sum())
    max_stay = int(counts.max())
    sorted_desc = np.sort(counts)[::-1]
    rows = []
    for p in ps:
        # No partitioning can beat this: one stay is atomic.
        bound = max(total / p, max_stay)
        for name, fn in STRATEGIES.items():
            ml = fn(counts, p)
            rows.append({
                "window": window, "strategy": name, "p": p,
                "total_work": total, "max_load": ml,
                "speedup": total / ml if ml else float("nan"),
                "efficiency": (total / ml) / p if ml else float("nan"),
                "bound_speedup": total / bound,
                "granularity_limited": bool(max_stay > total / p),
            })
        # D8 contrast: same strategy, size-sorted layout.
        ml = max_load_block(sorted_desc, p)
        rows.append({
            "window": window, "strategy": "block_sorted_desc", "p": p,
            "total_work": total, "max_load": ml,
            "speedup": total / ml if ml else float("nan"),
            "efficiency": (total / ml) / p if ml else float("nan"),
            "bound_speedup": total / bound,
            "granularity_limited": bool(max_stay > total / p),
        })
    return rows


def fig_ccdf(all_counts: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for w in WINDOW_ORDER:
        if w not in all_counts:
            continue
        c = np.sort(all_counts[w][all_counts[w] > 0])
        ccdf = 1.0 - np.arange(c.size) / c.size
        ax.loglog(c, ccdf, lw=1.4, label=w)
    ax.set_xlabel("records per stay")
    ax.set_ylabel("P(X > x)")
    ax.set_title("Records-per-stay complementary CDF")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "ccdf_records_per_stay.png", dpi=150)
    plt.close(fig)


def fig_lorenz(all_counts: dict[str, np.ndarray], stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect balance")
    for w in WINDOW_ORDER:
        if w not in all_counts:
            continue
        c = np.sort(all_counts[w].astype(np.float64))
        cum = np.concatenate(([0.0], np.cumsum(c) / c.sum()))
        x = np.linspace(0, 1, cum.size)
        ax.plot(x, cum, lw=1.4, label=f"{w} (G={stats[w]['gini']:.3f})")
    ax.set_xlabel("cumulative share of stays")
    ax.set_ylabel("cumulative share of records")
    ax.set_title("Lorenz curve of work")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "lorenz.png", dpi=150)
    plt.close(fig)


def fig_hill(all_counts: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for w in WINDOW_ORDER:
        if w not in all_counts:
            continue
        c = all_counts[w][all_counts[w] > 0]
        ks = np.unique(np.geomspace(10, max(20, c.size // 5), 60).astype(int))
        ax.plot(ks, [hill(c, int(k)) for k in ks], lw=1.3, label=w)
    ax.axhline(2.0, color="k", ls=":", lw=1)
    ax.text(0.02, 0.06, "alpha = 2: infinite variance below this line",
            transform=ax.transAxes, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("k (order statistics used)")
    ax.set_ylabel("Hill tail index alpha")
    ax.set_title("Hill plot (smaller alpha = heavier tail)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "hill_plot.png", dpi=150)
    plt.close(fig)


def fig_speedup(rows: list[dict], windows: list[str]) -> None:
    show = [w for w in WINDOW_ORDER if w in windows]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.4),
                             sharey=True, squeeze=False)
    for ax, w in zip(axes[0], show):
        sub = [r for r in rows if r["window"] == w]
        ps = sorted({r["p"] for r in sub})
        ax.plot(ps, ps, "k--", lw=1, label="linear")
        for name in list(STRATEGIES) + ["block_sorted_desc"]:
            ys = [next(r["speedup"] for r in sub
                       if r["p"] == p and r["strategy"] == name) for p in ps]
            ax.plot(ps, ys, marker="o", ms=3, lw=1.3, label=name)
        bs = [next(r["bound_speedup"] for r in sub if r["p"] == p) for p in ps]
        ax.plot(ps, bs, color="0.4", ls=":", lw=1.2, label="atomic-stay bound")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_title(w)
        ax.set_xlabel("workers p")
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel("theoretical max speedup")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Achievable speedup from the work distribution alone")
    fig.tight_layout()
    fig.savefig(RESULTS / "max_speedup.png", dpi=150)
    plt.close(fig)



def by_source(window: str) -> dict:
    """Split the tail into chart-only and lab-only using the disjoint id ranges.

    Labs are episodic and order-driven; chart vitals are continuous and
    protocol-driven. If the two have visibly different tail indices, that
    identifies which mechanism generates the imbalance.
    """
    import duckdb
    k_chart = duckdb.connect().execute(
        f"SELECT count(*) FROM read_parquet('{CSR / 'itemid_dict.parquet'}') "
        f"WHERE source = 'chart'"
    ).fetchone()[0]

    off = np.fromfile(CSR / window / "offsets.i64", dtype=np.int64)
    item = np.fromfile(CSR / window / "itemid.i32", dtype=np.int32)
    is_lab = item >= k_chart
    lab_cum = np.concatenate(([0], np.cumsum(is_lab)))
    lab_counts = np.diff(lab_cum[off])
    all_counts = np.diff(off)
    chart_counts = all_counts - lab_counts

    out = {}
    for name, c in (("chart", chart_counts), ("lab", lab_counts)):
        d = describe(c.astype(np.int64))
        out[name] = d
        print(f"    {name:<6} n_rec={d['n_records']:>12,}  median={d['p50']:>8.0f}  "
              f"p99={d['p99']:>9.0f}  p99/med={d['p99_over_median']:>7.2f}  "
              f"G={d['gini']:.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=",".join(WINDOW_ORDER))
    ap.add_argument("--by-source", action="store_true")
    args = ap.parse_args()

    windows = [w.strip() for w in args.windows.split(",")]
    missing = [w for w in windows if not (CSR / w / "offsets.i64").is_file()]
    if missing:
        print(f"missing offsets for {missing} -- run csr.py first", file=sys.stderr)
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)

    all_counts, stats = {}, {}
    print("=" * 78)
    print("Records-per-stay distribution")
    print("=" * 78)
    print(f"  {'window':<7} {'stays':>8} {'records':>13} {'median':>8} {'p90':>8} "
          f"{'p99':>9} {'max':>9} {'p99/med':>8} {'gini':>6} {'CV':>6}")
    for w in windows:
        c = load_counts(w)
        all_counts[w] = c
        d = describe(c)
        stats[w] = d
        print(f"  {w:<7} {d['n_stays']:>8,} {d['n_records']:>13,} "
              f"{d['p50']:>8.0f} {d['p90']:>8.0f} {d['p99']:>9.0f} "
              f"{d['max']:>9,} {d['p99_over_median']:>8.2f} "
              f"{d['gini']:>6.3f} {d['cv']:>6.2f}")

    print(f"\n  Hill tail index alpha (smaller = heavier; alpha<2 = infinite variance)")
    print(f"  {'window':<7} " + " ".join(f"{f'k={f:g}':>9}" for f in HILL_FRACTIONS))
    for w in windows:
        vals = " ".join(f"{stats[w]['hill'][f'k={f:g}']:>9.3f}" for f in HILL_FRACTIONS)
        print(f"  {w:<7} {vals}")

    print("\n" + "=" * 78)
    print("Analytical partitioning bounds")
    print("=" * 78)
    rows = []
    for w in windows:
        rows += partition_rows(all_counts[w], w, sorted(set(P_CURVE + P_TABLE)))
        c = all_counts[w]
        total, mx = int(c.sum()), int(c.max())
        p_star = total / mx
        print(f"\n  {w}: total={total:,}  largest stay={mx:,}  "
              f"p* = {p_star:.1f} workers")
        print(f"    (beyond p*, the largest single stay exceeds the ideal share "
              f"and caps speedup regardless of strategy)")
        print(f"    {'p':>5} " + " ".join(f"{s:>12}" for s in
                                          list(STRATEGIES) + ["bound"]))
        for p in P_TABLE:
            sub = {r["strategy"]: r for r in rows
                   if r["window"] == w and r["p"] == p}
            line = " ".join(f"{sub[s]['speedup']:>12.2f}" for s in STRATEGIES)
            print(f"    {p:>5} {line} {sub['block']['bound_speedup']:>12.2f}")
        for p in P_TABLE:
            sub = {r["strategy"]: r for r in rows
                   if r["window"] == w and r["p"] == p}
            eb, eg = sub["block"]["efficiency"], sub["greedy"]["efficiency"]
            print(f"    p={p:<4} block efficiency {eb:6.1%}   "
                  f"greedy {eg:6.1%}   recoverable {eg - eb:+6.1%}")

    if args.by_source:
        print("\n" + "=" * 78)
        print("Tail by source (chart vs lab), disjoint dense id ranges")
        print("=" * 78)
        for w in windows:
            if (CSR / w / "itemid.i32").is_file():
                print(f"  {w}")
                stats[w]["by_source"] = by_source(w)

    print("\n" + "=" * 78)
    print(f"Gate (D13), binding window {PRIMARY}")
    print("=" * 78)
    gate = {}
    for w in windows:
        d = stats[w]
        g_ok = d["gini"] > GATE_GINI
        r_ok = d["p99_over_median"] > GATE_P99_OVER_MEDIAN
        gate[w] = {"gini": d["gini"], "gini_pass": bool(g_ok),
                   "p99_over_median": d["p99_over_median"],
                   "ratio_pass": bool(r_ok), "pass": bool(g_ok and r_ok)}
        mark = "<-- PRIMARY" if w == PRIMARY else ""
        print(f"  {w:<7} gini {d['gini']:.3f} {'PASS' if g_ok else 'FAIL':<5} "
              f"(> {GATE_GINI})   p99/median {d['p99_over_median']:6.2f} "
              f"{'PASS' if r_ok else 'FAIL':<5} (> {GATE_P99_OVER_MEDIAN})  {mark}")

    verdict = gate.get(PRIMARY, {}).get("pass", False)
    print()
    if verdict:
        print(f"  GATE PASSED on {PRIMARY}. Proceed to Days 4-5.")
    elif gate.get("Wfull", {}).get("pass"):
        print(f"  GATE FAILED on {PRIMARY} but PASSED on Wfull.")
        print("  Pre-registered response (D13): promote Wfull to primary for the")
        print("  performance study; keep W <= 24 h for clinical metrics only,")
        print("  which D3 already requires because of outcome leakage. This is a")
        print("  re-scope, not a rescue, and belongs in the methods section.")
    else:
        print(f"  GATE FAILED on {PRIMARY} and on Wfull.")
        print("  Re-scope before writing kernels -- that is what the gate is for.")

    # Aggregate outputs. Counts and quantiles only; no subject_id, no stay_id.
    import csv as _csv
    with (RESULTS / "distribution_summary.csv").open("w", newline="") as fh:
        keys = [k for k in stats[windows[0]] if k not in ("hill", "by_source")]
        wtr = _csv.DictWriter(fh, fieldnames=["window"] + keys)
        wtr.writeheader()
        for w in windows:
            wtr.writerow({"window": w,
                          **{k: stats[w][k] for k in keys}})

    with (RESULTS / "partitioning_bounds.csv").open("w", newline="") as fh:
        wtr = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    (MANIFEST / "e0_summary.json").write_text(json.dumps({
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primary_window": PRIMARY,
        "gate_thresholds": {"gini": GATE_GINI,
                            "p99_over_median": GATE_P99_OVER_MEDIAN},
        "gate": gate, "verdict_primary_pass": bool(verdict),
        "stats": stats,
    }, indent=2) + "\n")

    fig_ccdf(all_counts)
    fig_lorenz(all_counts, stats)
    fig_hill(all_counts)
    fig_speedup(rows, windows)

    print(f"\n  csv     -> {RESULTS}")
    print(f"  figures -> ccdf_records_per_stay.png, lorenz.png, "
          f"hill_plot.png, max_speedup.png")
    print(f"  summary -> {MANIFEST / 'e0_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
