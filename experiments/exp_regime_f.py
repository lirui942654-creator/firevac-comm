"""E6: 双场景完整矩阵 —— spreading-fire 场景下全部 7 策略。

背景：E5 初测发现 spreading-fire 下 targeted 优势消失（信息结构效应）。
本实验补齐 spreading-fire × 7 策略的完整矩阵，与 wall 场景(_final52)并列，
支撑论文升级后的核心主张：
  "最优调度策略取决于 (预算, 信息结构) 二元组——
   离散目标信息 -> 定向分发；动态场信息 -> 广播分发。"

策略：fixed / event / targeted / hybrid / random / roundrobin / bandit(ε)
种子：40/点（论文如实标注；wall 为 52）
"""
import numpy as np, time, io, csv, sys
from collections import deque
sys.path.insert(0, r"D:\Users\86151\Desktop\论文投稿")
import evac_env as E

GRID, SEEDS, LOSS = 20, 40, 0.1
FED_DOWN, SMOKE_RATE = 0.30, 0.15


def run(mode, radius, param, seed, n_agents=6):
    env = E.FireEvacEnv(grid=GRID, n_agents=n_agents, max_steps=200,
                        comm_mode='fixed', comm_radius=radius, comm_frac=1.0,
                        comm_loss=LOSS, use_incap=True, fed_down=FED_DOWN, seed=seed)
    env.communicate = lambda act=None: None
    env.reset()
    g = GRID
    env.exit = (g - 1, g - 1)
    quads = [(2, 2), (g - 5, 2), (2, g // 2 - 4), (g - 5, g // 2 - 4),
             (g // 2 - 2, 2), (g // 2 + 2, 2)]
    for i in range(n_agents):
        env.pos[i] = quads[i % len(quads)]
        env.pos[i, 0] += int(env.rng.integers(-1, 2))
        env.pos[i, 1] += int(env.rng.integers(-1, 2))
    env.fed[:] = 0; env.down[:] = False; env.alive[:] = True
    env.use_incap = True; env.fed_down = FED_DOWN
    env._occupied_by_other = lambda r, c, i: env._down_at(r, c)

    bel_exit = [tuple(env.exit)] * n_agents
    belief = np.zeros((n_agents, g, g), bool)
    bknown = np.zeros((n_agents, g, g), bool)
    last_sig = [None] * n_agents
    since_sent = np.zeros(n_agents, int)
    token = 0
    Q = np.zeros((n_agents, 2), np.float32); Ncnt = np.zeros((n_agents, 2), int)
    EPS = 0.15
    rng = env.rng

    def nav(i):
        pos = tuple(env.pos[i]); goal = bel_exit[i]
        parent = {pos: None}; q = deque([pos]); found = False
        while q:
            cur = q.popleft()
            if cur == goal: found = True; break
            r, c = cur
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0 <= nr < g and 0 <= nc < g and (nr, nc) not in parent and not belief[i, nr, nc]:
                    parent[(nr, nc)] = cur; q.append((nr, nc))
        if not found:
            r, c = pos
            for cand in [(E.DOWN, 1, 0), (E.RIGHT, 0, 1), (E.UP, -1, 0), (E.LEFT, 0, -1)]:
                nr, nc = min(max(r + cand[1], 0), g - 1), min(max(c + cand[2], 0), g - 1)
                if not belief[i, nr, nc]:
                    return cand[0]
            return int(rng.integers(4))
        cur = goal
        while parent[cur] is not None and parent[cur] != pos:
            cur = parent[cur]
        dr, dc = cur[0]-pos[0], cur[1]-pos[1]
        return (E.DOWN if dr == 1 else E.UP if dr == -1 else
                E.RIGHT if dc == 1 else E.LEFT if dc == -1 else E.STAY)

    for step in range(env.max_steps):
        for i in np.where(env.alive & ~env.down)[0]:
            r, c = env.pos[i]; rf, cf = max(r-2, 0), max(c-2, 0)
            sub = env.fire[rf:rf+5, cf:cf+5]
            for a in range(5):
                for b_ in range(5):
                    rr, cc = rf+a, cf+b_
                    if 0 <= rr < g and 0 <= cc < g:
                        belief[i, rr, cc] = sub[a, b_]; bknown[i, rr, cc] = True
        alive_idx = np.where(env.alive & ~env.down)[0]
        senders = []
        if mode == 'fixed':
            for i in alive_idx:
                if rng.random() <= param:
                    senders.append(i)
        elif mode == 'event':
            for i in alive_idx:
                sig = (bel_exit[i], tuple(belief[i][bknown[i]].tolist()))
                if sig != last_sig[i]:
                    senders.append(i); last_sig[i] = sig
        elif mode == 'targeted':
            for i in alive_idx:
                if any((~bknown[j] & bknown[i]).any() and
                       np.linalg.norm(env.pos[i] - env.pos[j]) <= radius
                       for j in alive_idx if j != i):
                    senders.append(i)
        elif mode == 'hybrid':
            H = param
            for i in alive_idx:
                sig = (bel_exit[i], tuple(belief[i][bknown[i]].tolist()))
                if sig != last_sig[i] or since_sent[i] >= H:
                    senders.append(i); last_sig[i] = sig; since_sent[i] = 0
        elif mode == 'random':
            for i in alive_idx:
                nbs = [j for j in alive_idx if j != i and
                       np.linalg.norm(env.pos[i] - env.pos[j]) <= radius]
                if nbs and rng.random() <= param:
                    senders.append((i, [nbs[rng.integers(len(nbs))]]))
        elif mode == 'roundrobin':
            i = alive_idx[token % len(alive_idx)]
            token += 1
            senders.append(i)
        elif mode == 'bandit':
            for i in alive_idx:
                a = int(rng.integers(2)) if rng.random() < EPS else int(np.argmax(Q[i]))
                if a == 0:
                    senders.append(i)
        # 执行发送
        round_recv = np.zeros(n_agents, int)
        for item in senders:
            i, targets = (item[0], alive_idx) if isinstance(item, tuple) else (item, alive_idx)
            for j in alive_idx:
                if j == i: continue
                if np.linalg.norm(env.pos[i] - env.pos[j]) <= radius:
                    env.metrics["n_send"] += 1
                    if rng.random() < LOSS: continue
                    was_unknown_fire = int((~bknown[j]).sum())
                    mask = bknown[i]
                    belief[j, mask] |= belief[i, mask]; bknown[j, mask] = True
                    newly = was_unknown_fire - int((~bknown[j]).sum())
                    round_recv[i] += newly
        if mode == 'bandit':
            sent_ids = set(s[0] if isinstance(s, tuple) else s for s in senders)
            for i in alive_idx:
                a = 0 if i in sent_ids else 1
                Ncnt[i, a] += 1
                Q[i, a] += (round_recv[i] - 0.05 * (env.metrics['n_send'] / n_agents) - Q[i, a]) / Ncnt[i, a]
        sent_ids = set(s[0] if isinstance(s, tuple) else s for s in senders)
        for i in alive_idx:
            if i not in sent_ids:
                since_sent[i] += 1
        # FED 由 env.step 内置的 ISO 13571 公式累积（勿在此重复加算——
        # 之前双重计费导致失能过快，已修复：K=5 -> 浓烟约40步失能）
        acts = np.array([nav(i) for i in range(n_agents)], np.int64)
        _, _, term, trunc, info = env.step(acts)
        if term or trunc:
            break
    return info['metrics'], env


def main():
    # sanity
    _, e0 = run('fixed', 4.0, 0.5, 0)
    assert e0.fire.sum() > 5, "火没有蔓延"
    print(f"sanity OK: fire_cells={int(e0.fire.sum())}, n_down={e0.metrics['n_down']}")
    t0 = time.perf_counter()
    rows = []
    cfgs = {
        'fixed':    [(4.0, th) for th in [0.1, 0.3, 1.0]] + [(8.0, th) for th in [0.3, 1.0]],
        'event':    [(r, 0) for r in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]],
        'targeted': [(r, 0) for r in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]],
        'hybrid':   [(4.0, h) for h in [5, 12, 20]] + [(6.0, 12)],
        'random':   [(r, p) for r in [4.0, 8.0] for p in [0.3, 1.0]],
        'roundrobin': [(r, 0) for r in [4.0, 6.0, 8.0]],
        'bandit':   [(r, 0) for r in [3.0, 4.0, 6.0, 8.0]],
    }
    total = sum(len(l) for l in cfgs.values())
    done = 0
    for mode, lst in cfgs.items():
        for rad, p in lst:
            S, Vv, O = [], [], []
            for s in range(SEEDS):
                m, e = run(mode, rad, p, s)
                S.append(m['success']); Vv.append(m['survival']); O.append(m['comm_overhead'])
            print(f"{mode:11s} r={rad:4.1f} p={p:5.2f}  oh={np.mean(O):5.2f}  "
                  f"succ={np.mean(S):6.1%}  surv={np.mean(Vv):6.1%}  "
                  f"[{done+1}/{total}]", flush=True)
            rows.append([mode, rad, p, float(np.mean(O)), float(np.mean(S)),
                         float(np.mean(Vv)), S])
            done += 1
    # 分桶表
    bins = np.arange(0, 4.01, 0.4)
    order = ['fixed', 'event', 'targeted', 'hybrid', 'random', 'roundrobin', 'bandit']
    lines = ["bin              " + "  ".join(f"{m:15s}" for m in order)]
    for b0, b1 in zip(bins[:-1], bins[1:]):
        row = {}
        for mode in order:
            raw = [x for (m, r, p, o, s, v, SS) in rows if m == mode and b0 <= o < b1 for x in SS]
            if raw:
                row[mode] = (np.mean(raw), np.std(raw) / np.sqrt(len(raw)) * 1.96)
        if not row:
            continue
        cells = []
        for mode in order:
            if mode in row:
                m, half = row[mode]
                cells.append(f"{m:5.1%}±{half:4.1%}")
            else:
                cells.append("      --      ")
        lines.append(f"({b0:.2f},{b1:.2f})  " + "  ".join(cells))
    print()
    for ln in lines:
        print(ln)
    with io.open(r"D:\Users\86151\Desktop\论文投稿\_spreadfire_full.csv", "w", encoding="utf-8", newline='') as f:
        w = csv.writer(f); w.writerow(["strategy", "radius", "param", "overhead", "success", "survival"])
        for m, r, p, o, s, v, SS in rows:
            w.writerow([m, r, p, o, s, v])
    io.open(r"D:\Users\86151\Desktop\论文投稿\_spreadfire_table.txt", "w", encoding="utf-8").write("\n".join(lines))
    # 与 wall 场景的排序对比
    print("\n=== 双场景对照（各桶最优策略） ===")
    print("bucket        spreading-fire-best      wall-best(52seed)")
    wall_best = {(0.0, 0.4): 'targeted 86.2%', (0.4, 0.8): 'roundrobin 92.9%',
                 (1.2, 1.6): 'fixed 95.2%', (2.4, 2.8): 'event/hybrid 95.8%'}
    for b0, b1 in zip(bins[:-1], bins[1:]):
        best, bv = None, -1
        for m, r, p, o, s, v, SS in rows:
            if b0 <= o < b1 and s > bv:
                best, bv = m, s
        if best:
            wk = None
            for (w0, w1), wv in wall_best.items():
                if abs(w0-b0) < 0.2:
                    wk = wv
            print(f"({b0:.2f},{b1:.2f})   {best} {bv:6.1%}          {wk or '--'}")
    print(f"\n耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
