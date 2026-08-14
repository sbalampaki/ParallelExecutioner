#!/usr/bin/env python3

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--min-threads", type=int, default=12,
                    help="only runs with at least this many threads")
    a = ap.parse_args()

    th = pd.read_csv(a.threads)
    th["rate"] = th.n_records / th.secs

    out = []
    for rid, g in th.groupby("run_id"):
        g = g.sort_values("tid")
        if len(g) < a.min_threads:
            continue
        trans = int((g.socket.values[1:] != g.socket.values[:-1]).sum())
        order = ("grouped" if trans <= 1
                 else "alternating" if trans >= len(g) - 2 else f"other({trans})")
        slow = g.loc[g.rate.idxmin()]
        out.append(dict(run_id=rid, T=len(g), order=order,
                        slow_tid=int(slow.tid), slow_cpu=int(slow.cpu),
                        slow_sock=int(slow.socket),
                        deficit=100 * (slow.rate / g.rate.median() - 1)))
    d = pd.DataFrame(out)
    if d.empty:
        raise SystemExit("no runs with enough threads -- lower --min-threads")

    print(f"=== slowest thread per run, {len(d)} runs ===\n")
    for order, g in d.groupby("order"):
        print(f"  --- {order}  ({len(g)} runs) ---")
        print("    slowest tid:", g.slow_tid.value_counts().to_dict())
        print("    slowest cpu:", g.slow_cpu.value_counts().to_dict())
        print(f"    median deficit vs that run's median thread: "
              f"{g.deficit.median():+.1f}%\n")

    print("READ IT AS:")
    print("  alternating picks out cpu 8 (whatever tid that is)  -> CPU-ATTACHED")
    print("  alternating picks out tid 8 (whatever cpu that is)  -> TID-ATTACHED")
    print("  no clear winner under alternating -> the effect needs grouped")
    print("  placement to appear at all, which is a third answer and the")
    print("  most interesting one.\n")

    # per-cpu and per-tid profiles under each ordering, for the full picture
    j = th.merge(th.groupby("run_id").size().rename("T"), on="run_id")
    j = j[j["T"] >= a.min_threads]
    tr = (j.sort_values(["run_id", "tid"]).groupby("run_id")
            .socket.apply(lambda s: int((s.values[1:] != s.values[:-1]).sum())))
    j["order"] = j.run_id.map(lambda r: "grouped" if tr[r] <= 1 else "alternating")
    for order, g in j.groupby("order"):
        norm = g.groupby("run_id").rate.transform("median")
        g = g.assign(rel=100 * (g.rate / norm - 1))
        print(f"=== {order}: relative rate (%) ===")
        print("  by cpu:", {int(k): round(v, 1)
                            for k, v in g.groupby("cpu").rel.median().items()})
        print("  by tid:", {int(k): round(v, 1)
                            for k, v in g.groupby("tid").rel.median().items()})
        print()


if __name__ == "__main__":
    main()
