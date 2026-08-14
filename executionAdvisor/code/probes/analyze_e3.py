#!/usr/bin/env python3

import argparse
import glob
import math
import sys

import numpy as np
import pandas as pd

KEY = ["csr_variant", "n_threads", "thread_placement", "partitioner",
       "schedule", "chunk_size", "row_order", "whitelist_frac", "tile"]

L3_PER_SOCKET = 20e6      # Broadwell, platform.md
L2_PER_CORE   = 262144
L1D_PER_CORE  = 32768
F_FEATURES = 160
SOCKETS = 2


def load(paths, thread_paths=None):
    frames = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            frames.append(pd.read_csv(f))
    if not frames:
        sys.exit("no input files matched")
    d = pd.concat(frames, ignore_index=True)


    tp = thread_paths or [p.replace(".csv", "_threads.csv") for p in paths]
    tf = []
    for p in tp:
        for f in sorted(glob.glob(p)):
            tf.append(pd.read_csv(f))
    if tf:
        t = pd.concat(tf, ignore_index=True)
        def _order(g):
            sk = g.sort_values("tid").socket.values
            if len(sk) < 2 or len(np.unique(sk)) < 2:
                return "single"
            return "grouped" if (sk[1:] != sk[:-1]).sum() <= 1 else "alternating"
        o = t.groupby("run_id").apply(_order, include_groups=False).rename("_ord")
        d = d.merge(o, on="run_id", how="left")
        d["_ord"] = d["_ord"].fillna("unknown")
        # Fold into thread_placement so every existing groupby separates
        # them without touching the rest of the script.
        two = d.thread_placement.isin(["spread", "smt"])
        d.loc[two, "thread_placement"] = (d.loc[two, "thread_placement"]
                                          + "/" + d.loc[two, "_ord"])
        n = d.loc[two, "_ord"].nunique()
        if n > 1:
            print(f"  [thread ordering recovered from threads.csv: "
                  f"{sorted(d.loc[two, '_ord'].unique())}]")
    else:
        print("  [no threads.csv found -- thread ordering NOT separated; "
              "alternating and grouped rows are pooled and unusable]")
   
    if "thread_placement" not in d.columns:
        d["thread_placement"] = "unknown"
    d["tile"] = d.notes.str.extract(r"tile=(\d+)").astype(float).astype("Int64")
    d["pred_eff"] = d.notes.str.extract(r"pred_eff=([0-9.]+)").astype(float)
    d["migrated"] = d.notes.str.extract(r"migrated=(\d+)").astype(float)
    d["mrec_s"] = d.throughput_rec_s / 1e6
    # records per cycle: the clock-invariant measure of work done
    d["rec_per_cycle"] = np.where(d.achieved_ghz > 0,
                                  d.throughput_rec_s / (d.achieved_ghz * 1e9),
                                  np.nan)
    return d[d.warmup == 0].copy()


def med(g, col="mrec_s"):
    return g[col].median()


def summarise(d):
    """Median over repetitions, per (node, configuration)."""
    g = d.groupby(["node"] + KEY, dropna=False)
    out = g.agg(mrec_s=("mrec_s", "median"),
                rec_cyc=("rec_per_cycle", "median"),
                ghz=("achieved_ghz", "median"),
                work_imb=("work_imbalance", "median"),
                time_imb=("time_imbalance", "median"),
                pred_eff=("pred_eff", "first"),
                iqr_pct=("mrec_s", lambda s: 100 * (s.quantile(.75) - s.quantile(.25)) / s.median()),
                n_reps=("mrec_s", "size")).reset_index()
    return out


def at_place(sub, override=None, label=""):

    if sub.empty or "thread_placement" not in sub.columns:
        return sub
    t = override if override else sub.thread_placement.mode().iloc[0] \
        if len(sub.thread_placement.mode()) else None
    if t is None:
        return sub
    out = sub[sub.thread_placement == t]
    if label:
        print(f"  [{label}: placement = {t}, {len(out)} rows]")
    return out


def at_tile(sub, override=None, label=""):

    if sub.empty or sub.tile.isna().all():
        return sub
    if override is not None:
        t = override
    else:
        m = sub.tile.mode()
        if len(m) == 0:
            return sub
        t = m.iloc[0]
    out = sub[sub.tile == t]
    if label:
        print(f"  [{label}: tile = {t}, {len(out)} node-configuration rows]")
    return out


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timings", nargs="+")
    ap.add_argument("--threads-csv", nargs="*", default=None,
                    help="threads.csv glob(s); default: derived from the "
                         "timings path by substituting _threads.csv")
    ap.add_argument("--placement", default=None,
                    help="restrict comparisons to one placement "
                         "(socket0|socket1|spread|smt); default: modal")
    ap.add_argument("--tile", type=int, default=None,
                    help="restrict comparisons to this tile size "
                         "(default: the modal tile in each subset)")
    a = ap.parse_args()

    d = load(a.timings, a.threads_csv)
    s = summarise(d)

    section("0. INTEGRITY")
    n_fail = int((d.bitidentical == "fail").sum())
    n_pass = int((d.bitidentical == "pass").sum())
    n_skip = int((d.bitidentical == "skip").sum())
    print(f"  bit-identity   pass {n_pass}   skip {n_skip}   FAIL {n_fail}")
    if n_fail:
        print("  ** FAIL is a race or a build that did not link the shared")
        print("     kernel_core.o. Nothing below is interpretable.")
        return 1
    mig = int(d.migrated.fillna(0).sum())
    print(f"  cpu migrations {mig}   (nonzero breaks first-touch locality)")
    print(f"  nodes          {sorted(d.node.unique())}")
    print(f"  configurations {len(s.groupby(KEY, dropna=False))} distinct, "
          f"{len(s)} node-configuration pairs")

    section("1. REPRODUCIBILITY")
    # duplicates within a node = same treatment run twice at different points
    dup = (s.groupby(["node"] + KEY, dropna=False).size())
    print("  within-repetition IQR (median over configurations): "
          f"{s.iqr_pct.median():.2f}%")
    per_cfg = s.groupby(KEY, dropna=False).mrec_s
    spread = (100 * (per_cfg.max() - per_cfg.min()) / per_cfg.median()).dropna()
    print(f"  across-node spread (median over configurations):  "
          f"{spread.median():.2f}%   worst {spread.max():.2f}%")
    print("\n  Node is a treatment, not noise: if the across-node spread is")
    print("  comparable to the effects being measured, single-node results")
    print("  cannot be reported without it.")

    section("2. THREAD SCALING — P6 (raw vs clock-corrected)")
    for variant in ["W24", "Wfull"]:
        base = s[(s.csr_variant == variant) & (s.partitioner == "block")
                 & (s.schedule == "precomputed")
                 & (s.row_order == "canonical")]
        base = base[base.whitelist_frac.between(0.30, 0.45)]
        base = at_tile(base, a.tile, f"{variant} thread scaling")
        if base.empty:
            continue
        g = base.groupby(["thread_placement", "n_threads"]).agg(
            mrec=("mrec_s", "median"), cyc=("rec_cyc", "median"),
            ghz=("ghz", "median"), timb=("time_imb", "median")).reset_index()
        if g.empty:
            continue
        print(f"\n  {variant}")
    
        t1 = g[g.n_threads == 1]
        shared = t1.iloc[0] if not t1.empty else None
        for place in sorted(g.thread_placement.unique()):
            gp = g[g.thread_placement == place].sort_values("n_threads").copy()
            if shared is not None:
                base1, note = shared, ""
            else:
                base1 = gp.iloc[0]
                note = f"  (no T=1 anywhere; normalised at T={int(base1.n_threads)})"
            # efficiency = speedup x base_T / T. Dividing by T alone (the
            # earlier form) reported 50% for a branch based at T=2 and 8%
            # for one based at T=12.
            gp["speedup_raw"] = gp.mrec / base1.mrec
            gp["eff_raw"] = 100 * gp.speedup_raw * base1.n_threads / gp.n_threads
            gp["speedup_cyc"] = gp.cyc / base1.cyc
            gp["eff_cyc"] = 100 * gp.speedup_cyc * base1.n_threads / gp.n_threads
            print(f"   [{place}]{note}")
            print(f"    {'T':>3} {'M rec/s':>9} {'GHz':>6} {'raw x':>7} {'raw %':>7} "
                  f"{'cyc x':>7} {'cyc %':>7} {'time_imb':>9}")
            for _, r in gp.iterrows():
                print(f"    {int(r.n_threads):>3} {r.mrec:>9.1f} {r.ghz:>6.2f} "
                      f"{r.speedup_raw:>7.2f} {r.eff_raw:>7.1f} "
                      f"{r.speedup_cyc:>7.2f} {r.eff_cyc:>7.1f} {r.timb:>9.4f}")
        g = g[g.thread_placement == ("spread" if "spread" in set(g.thread_placement)
                                     else g.thread_placement.iloc[0])].copy()
        g = g.sort_values("n_threads")
        marg = g.mrec.diff() / g.n_threads.diff()
        knee = None
        for i in range(1, len(g)):
            if marg.iloc[i] < 0.25 * marg.iloc[1]:
                knee = int(g.n_threads.iloc[i]); break
        print(f"    P6 predicted saturation at 12-13 cores; "
              f"observed marginal collapse at {knee if knee else '>16'}")

    section("3. TILE SIZE — P7 (L3, refuted) vs P7' (L2, per core)")
    print(f"  P7  : C* = 15625/ceil(T/2)   shared L3 per socket -- REFUTED,")
    print(f"        the observed optimum did not move with T.")
    print(f"  P7' : the tile is per-thread PRIVATE and written-then-read, so")
    print(f"        there is no cross-thread reuse for a shared cache to hold.")
    print(f"        The binding capacity is per core:")
    print(f"          L2  {L2_PER_CORE/1024:.0f} KB -> C = {L2_PER_CORE/(F_FEATURES*8):.0f} stays")
    print(f"          L1d {L1D_PER_CORE/1024:.0f} KB  -> C = {L1D_PER_CORE/(F_FEATURES*8):.0f} stays")
    print(f"        so the optimum should be INDEPENDENT of T, and tile=0")
    print(f"        (direct write) should beat any tile above L2.")
    tl = s[(s.csr_variant == "W24") & (s.partitioner == "block")
           & (s.schedule == "precomputed") & (s.row_order == "canonical")
           & s.whitelist_frac.between(0.30, 0.45)]
    tl = at_place(tl, a.placement, "tile sweep")
    if not tl.empty:
        piv = tl.pivot_table(index="tile", columns="n_threads",
                             values="mrec_s", aggfunc="median")
        print("\n  median M rec/s")
        print(piv.round(1).to_string())
        print(f"\n    {'T':>3} {'C* predicted':>13} {'C observed':>11} "
              f"{'best':>9} {'C<=C*':>9} {'gap':>9}")
        for T in sorted(piv.columns):
            col = piv[T].dropna()
            if len(col) < 3:
                continue
            per_sock = math.ceil(T / 2)
            cstar = L3_PER_SOCKET / (per_sock * F_FEATURES * 8)
            cl2 = L2_PER_CORE / (F_FEATURES * 8)
            obs = col.idxmax()
            # P7 says the tile should FIT, so the predicted best is the
            # largest swept C at or below C*, not the nearest in absolute
            # distance -- the shape is a step, not a peak.
            below = [c for c in col.index if float(c) <= cstar]
            pick = max(below) if below else min(col.index)
            pen = 100 * (col.max() - col[pick]) / col.max()
            below2 = [c for c in col.index if float(c) <= cl2]
            p2 = max(below2) if below2 else min(col.index)
            print(f"    {T:>3} {cstar:>13.0f} {obs:>11} {col.max():>9.1f} "
                  f"{col[pick]:>9.1f} {pen:>8.1f}%   L2 C<={cl2:.0f}: "
                  f"{col[p2]:>8.1f} ({100*(col.max()-col[p2])/col.max():>4.1f}%)")
        if tl.tile.nunique() < 2:
        
            print("\n  only one tile size in this data -- no sweep, so P7/P7'")
            print("  are not tested here. See D37 for the settled result.")
        else:
            print("\n  READ AS:")
            print("    optimum tracks C*(L3) down as T rises -> P7 confirmed.")
            print("    optimum flat in T, and best at C <= L2 -> P7' confirmed;")
            print("      the binding capacity is per core, not shared.")
            print("    tile=0 best at every T -> tiling is pure overhead (D37).")

    
    section("4. PARTITIONERS  (a-priori assignment; A8 / D35)")
    for variant in ["W24", "Wfull"]:
        pp = s[(s.csr_variant == variant) & (s.schedule == "precomputed")
               & s.whitelist_frac.between(0.30, 0.45)]
        pp = at_place(pp, a.placement, f"{variant} partitioners")
        pp = at_tile(pp, a.tile, f"{variant} partitioners")
        if pp.empty:
            continue
        piv = pp.pivot_table(index=["row_order", "partitioner"],
                             columns="n_threads",
                             values=["mrec_s", "work_imb", "time_imb"],
                             aggfunc="median")
        print(f"\n  {variant}")
        print(piv.round(3).to_string())
    print("\n  CAVEAT (D36): at Wfull p>=8 the seed contributes 3-5 pp to block")
    print("  efficiency, so single-seed partitioner differences below ~5 pp on")
    print("  that variant are not resolvable. W24's seed noise is 0.2-0.3 pp.")

    section("5. SCHEDULES  (runtime dispatch; chunk = A5's m knob)")
    sc = s[(s.csr_variant == "W24")
           & (s.row_order == "canonical")
           & s.whitelist_frac.between(0.30, 0.45)
           & (s.schedule != "precomputed")]
    sc = at_place(sc, a.placement, "schedules")
    sc = at_tile(sc, a.tile, "schedules")
    if not sc.empty:
        piv = sc.pivot_table(index=["schedule", "chunk_size"],
                             columns="n_threads",
                             values=["mrec_s", "work_imb", "time_imb"],
                             aggfunc="median")
        print(piv.round(3).to_string())
    print("\n  work_imbalance means the OPPOSITE thing here. For a-priori")
    print("  partitioners it measures scheduling quality; for dynamic/guided a")
    print("  LARGE work_imbalance with time_imbalance ~ 1.00 is the schedule")
    print("  succeeding -- it handed threads unequal record counts precisely to")
    print("  equalise time. A5/P1/P1prime must be fitted on partitioner rows only.")

    section("6. ARITHMETIC INTENSITY  (D32)")
    wl = s[(s.csr_variant == "W24") & (s.partitioner == "block")
           & (s.schedule == "precomputed")
           & (s.row_order == "canonical")]
    wl = at_place(wl, a.placement, "intensity")
    wl = at_tile(wl, a.tile, "intensity")
    if not wl.empty:
        wl = wl.copy()
        wl["hit"] = pd.cut(wl.whitelist_frac, [-0.01, 0.05, 0.5, 1.01],
                           labels=["none", "natural", "all"])
        wl["cyc_per_rec"] = 1.0 / wl.rec_cyc
        piv = wl.pivot_table(index="hit", columns="n_threads",
                             values="rec_cyc", aggfunc="median", observed=False)
        pivc = wl.pivot_table(index="hit", columns="n_threads",
                              values="cyc_per_rec", aggfunc="median",
                              observed=False)
        print("\n  CYCLES PER RECORD (clock-corrected; lower is better).")
        print("  Reported this way round because records/cycle is ~0.06 at one")
        print("  thread and rounds to nothing.")
        print(pivc.round(3).to_string())
        print("\n  Scaling efficiency vs 1 thread, per intensity:")
        for h in piv.index:
            row = piv.loc[h].dropna()
            if 1 in row.index and len(row) > 1:
                eff = 100 * (row / row[1]) / row.index
                print(f"    {str(h):<8} " +
                      "  ".join(f"T{int(t)}={e:.0f}%" for t, e in eff.items()))
        print("\n  Same code, same data, same schedule -- only the hit rate")
        print("  moves. If the knee moves with intensity, the saturation point")
        print("  is a property of the memory system and not of the kernel.")

    section("7. IMBALANCE DECOMPOSITION")
    ap_ = at_place(s[s.schedule == "precomputed"], a.placement,
                   "imbalance").copy()
    ap_["contamination"] = ap_.time_imb / ap_.work_imb
    print("  For a-priori partitioners, time_imbalance / work_imbalance is the")
    print("  contamination from memory contention and clock drift.")
    print("  D32 measured the straggler-ratio floor at 1.03-1.05 with a")
    print("  balanced control, so anything near that is nothing.\n")
    g = ap_.groupby(["csr_variant", "row_order", "thread_placement",
                     "n_threads"]).agg(
        work_imb=("work_imb", "median"), time_imb=("time_imb", "median"),
        contam=("contamination", "median")).reset_index()
    print(g.round(4).to_string(index=False))
    adv = g[g.row_order == "sorted_desc"]
    if not adv.empty:
        print("\n  On the adversarial layout, time_imbalance is far BELOW")
        print("  work_imbalance: when most threads finish early the straggler")
        print("  gets the whole memory system and runs faster than it would")
        print("  under contention. The memory system COMPRESSES measured")
        print("  imbalance, so wall-clock understates how badly balanced the")
        print("  work actually was.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
