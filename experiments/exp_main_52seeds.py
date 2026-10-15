"""最终定稿主表 —— 52 种子（由功效分析确定，power=0.8）。

数据口径与 _v2_main.py 完全一致（修正后协议），仅种子数提升。
产出：_final52_points.csv + _final52_table.txt，论文主表直接引用。
"""
import numpy as np, time, io, csv, sys
sys.path.insert(0, r"D:\Users\86151\Desktop\论文投稿")
import _v2_main as V
from _sched_compare import bootstrap_ci

N_SEEDS = 52


def main():
    t0 = time.perf_counter()
    pts = {}
    cfgs = {
        'fixed':    [(4.0, th) for th in [0.1, 0.3, 1.0]] + [(8.0, th) for th in [0.3, 1.0]],
        'event':    [(r, 0) for r in [2.0, 3.0, 4.0, 6.0, 8.0]],
        'targeted': [(r, 0) for r in [2.0, 3.0, 4.0, 6.0, 8.0]],
        'hybrid':   [(4.0, h) for h in [5, 12, 20]] + [(6.0, h) for h in [5, 12]],
        'random':   [(r, p) for r in [4.0, 8.0] for p in [0.3, 1.0]],
        'roundrobin': [(r, 0) for r in [4.0, 6.0, 8.0]],
        'bandit':   [(r, 0) for r in [3.0, 4.0, 6.0, 8.0]],
    }
    for mode, lst in cfgs.items():
        pts[mode] = []
        for rad, p in lst:
            S, Vv, O = [], [], []
            for s in range(N_SEEDS):
                m, e = V.run(mode, rad, p, s)
                S.append(m['success']); Vv.append(m['survival']); O.append(m['comm_overhead'])
            pts[mode].append((float(np.mean(O)), float(np.mean(S)), S, float(np.mean(Vv))))
            print(f"{mode:11s} r={rad:4.1f} p={p:5.2f}  oh={np.mean(O):5.2f}  "
                  f"succ={np.mean(S):6.1%}±{np.std(S):.3f}", flush=True)

    bins = np.arange(0, 5.01, 0.4)
    order = ['fixed', 'event', 'targeted', 'hybrid', 'random', 'roundrobin', 'bandit']
    lines = ["bin              " + "  ".join(f"{m:16s}" for m in order)]
    for b0, b1 in zip(bins[:-1], bins[1:]):
        row = {}
        for mode in order:
            raw = [x for (o, s, SS, v) in pts[mode] if b0 <= o < b1 for x in SS]
            if raw:
                lo, hi = bootstrap_ci(raw)
                row[mode] = (np.mean(raw), lo, hi)
        if not row:
            continue
        cells = []
        for mode in order:
            if mode in row:
                m, lo, hi = row[mode]
                cells.append(f"{m:5.1%}[{lo:.0%},{hi:.0%}]")
            else:
                cells.append("       --       ")
        lines.append(f"({b0:.2f},{b1:.2f})  " + "  ".join(cells))
    print()
    for ln in lines:
        print(ln)
    with io.open(r"D:\Users\86151\Desktop\论文投稿\_final52_points.csv", "w", encoding="utf-8", newline='') as f:
        w = csv.writer(f); w.writerow(["strategy", "overhead", "success", "survival"])
        for m in pts:
            for o, s, _, v in pts[m]:
                w.writerow([m, o, s, v])
    io.open(r"D:\Users\86151\Desktop\论文投稿\_final52_table.txt", "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
