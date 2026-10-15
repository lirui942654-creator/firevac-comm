"""Contextual Bandit 调度器 —— 前景研究：学习型调度何时能追平手工规则？

动机（来自 E3 泛化实验的发现）：
  N=4→12 时 targeted 对 fixed 的优势递减（80/63 → 78.9/66.7），
  说明"是否值得发送"应该依赖上下文，而非固定规则。

设计：每个 agent 一个线性 contextual bandit
  上下文 x = [本地烟雾, 邻居未知比例, 自身已知出口, 自身FED, 1(bias)]  (5维)
  两臂: 0=send, 1=wait
  回报: 本步"新点亮"的邻居数 − 0.1×(本步发送数/N)   # 信息增益 − 预算成本
  策略: LinUCI 风格 θ_a = A_a^{-1} b_a, argmax_a θ_a·x, ε-greedy 探索

对比目标（全部 52 种子口径可比）：
  contextual bandit vs targeted（极低开销王者） vs fixed（下界参照）
判定：
  若 bandit 在 matched-overhead 低预算段追平或超过 targeted -> 论文正面结果
  （"学习型调度首次追平手工"）; 若仍被压制 -> 强化 F3 负结果并分析原因。
"""
import numpy as np, time, io, csv, sys
from collections import deque
sys.path.insert(0, r"D:\Users\86151\Desktop\论文投稿")
import evac_env as E
from _v2_main import GRID, FED_DOWN, SMOKE_RATE


def run(mode, radius, param, seed):
    p_loss = 0.1
    env = E.FireEvacEnv(grid=GRID, n_agents=6, max_steps=200,
                        comm_mode='fixed', comm_radius=radius, comm_frac=1.0,
                        comm_loss=p_loss, use_incap=True, fed_down=FED_DOWN, seed=seed)
    # 复用 v2 的场景构建
    from _v2_main import build
    build(env, seed)
    g = GRID
    N = 6
    bel_exit = [None] * N
    lead = int(np.argmin(np.linalg.norm(env.pos - np.array(env.exit), axis=1)))
    bel_exit[lead] = tuple(env.exit)
    belief = np.zeros((N, g, g), bool)
    bknown = np.zeros((N, g, g), bool)
    rng = env.rng

    # ---- contextual bandit 参数 ----
    DX = 5
    A = [np.eye(DX) for _ in range(2)]          # 每臂协方差
    b = [np.zeros(DX) for _ in range(2)]
    ALPHA = 0.3                                  # UCB 探索系数
    last_ctx_send = [None] * N

    def context(i):
        r, c = env.pos[i]
        smoke_local = env.smoke[r, c]
        alive_idx = np.where(env.alive & ~env.down)[0]
        nbs = [j for j in alive_idx if j != i and
               np.linalg.norm(env.pos[i] - env.pos[j]) <= radius]
        unknown_frac = (np.mean([bel_exit[j] is None for j in nbs]) if nbs else 0.0)
        return np.array([smoke_local, unknown_frac,
                         1.0 if bel_exit[i] is not None else 0.0,
                         env.fed[i] / FED_DOWN, 1.0], np.float32)

    def see_exit(i):
        return np.linalg.norm(env.pos[i] - np.array(env.exit)) <= 5.0

    def search_move(i):
        r, c = env.pos[i]
        cand = [(r, min(c+1, g-1)), (min(r+1, g-1), c), (r, max(c-1, 0)), (max(r-1, 0), c)]
        for nr, nc in cand:
            if not env.fire[nr, nc] and not env._down_at(nr, nc) and (nr, nc) != (r, c):
                if nr == r+1: return E.DOWN
                if nr == r-1: return E.UP
                if nc == c+1: return E.RIGHT
                if nc == c-1: return E.LEFT
        return E.STAY

    def nav(i):
        pos = tuple(env.pos[i]); goal = bel_exit[i]
        if goal is None:
            return search_move(i)
        parent = {pos: None}; q = deque([pos]); found = False
        while q:
            cur = q.popleft()
            if cur == goal: found = True; break
            r, c = cur
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0 <= nr < g and 0 <= nc < g and (nr, nc) not in parent and not belief[i, nr, nc]:
                    parent[(nr, nc)] = cur; q.append((nr, nc))
        if not found:
            return search_move(i)
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
            if see_exit(i) and bel_exit[i] is None:
                bel_exit[i] = tuple(env.exit)
        alive_idx = np.where(env.alive & ~env.down)[0]
        senders = []
        if mode == 'bandit':
            for i in alive_idx:
                x = context(i)
                p = [np.dot(np.linalg.solve(A[a], b[a]), x) + ALPHA * np.sqrt(x @ np.linalg.solve(A[a], x))
                     for a in range(2)]
                a = int(np.argmax(p))
                if a == 0 and bel_exit[i] is not None:
                    senders.append((i, x, a))
        elif mode == 'targeted':
            for i in alive_idx:
                if bel_exit[i] is None:
                    continue
                if any(bel_exit[j] is None and
                       np.linalg.norm(env.pos[i] - env.pos[j]) <= radius
                       for j in alive_idx if j != i):
                    senders.append((i, None, 0))
        elif mode == 'fixed':
            for i in alive_idx:
                if rng.random() <= param:
                    senders.append((i, None, 0))
        # 执行发送
        round_recv = np.zeros(N, int)
        for item in senders:
            i = item[0]
            for j in alive_idx:
                if j == i: continue
                if np.linalg.norm(env.pos[i] - env.pos[j]) <= radius:
                    env.metrics["n_send"] += 1
                    if rng.random() < p_loss: continue
                    mask = bknown[i]
                    belief[j, mask] |= belief[i, mask]; bknown[j, mask] = True
                    if bel_exit[i] is not None and bel_exit[j] is None:
                        bel_exit[j] = bel_exit[i]
                        env.metrics["n_recv"] += 1
                        round_recv[i] += 1
        # bandit 更新
        if mode == 'bandit':
            sent_ids = {it[0] for it in senders}
            for i in alive_idx:
                if i in sent_ids:
                    it = next(t for t in senders if t[0] == i)
                    x, a = it[1], it[2]
                    r_i = round_recv[i] - 0.1 * (env.metrics['n_send'] / N)
                    A[a] += np.outer(x, x)
                    b[a] += r_i * x
        # FED
        for i in np.where(env.alive & ~env.down)[0]:
            r, c = env.pos[i]
            env.fed[i] += SMOKE_RATE * env.smoke[r, c]
            if env.fed[i] > FED_DOWN:
                env.down[i] = True
        acts = np.array([nav(i) for i in range(N)], np.int64)
        _, _, term, trunc, info = env.step(acts)
        if term or trunc:
            break
    return info['metrics'], env


def main():
    from _sched_compare import bootstrap_ci
    t0 = time.perf_counter()
    NS = 30
    pts = {}
    cfgs = {
        'bandit':   [(r, 0) for r in [2.0, 2.5, 3.0, 4.0, 6.0, 8.0]],
        'targeted': [(r, 0) for r in [2.0, 3.0, 4.0, 6.0, 8.0]],
        'fixed':    [(4.0, th) for th in [0.1, 0.3, 1.0]],
    }
    for mode, lst in cfgs.items():
        pts[mode] = []
        for rad, p in lst:
            S, O = [], []
            for s in range(NS):
                m, e = run(mode, rad, p, s)
                S.append(m['success']); O.append(m['comm_overhead'])
            pts[mode].append((float(np.mean(O)), float(np.mean(S)), S))
            print(f"{mode:9s} r={rad:4.1f} p={p:5.2f}  oh={np.mean(O):5.2f}  succ={np.mean(S):6.1%}", flush=True)
    print("\n=== matched-overhead: bandit vs targeted vs fixed ===")
    bins = np.arange(0, 5.01, 0.4)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        row = {}
        for mode in pts:
            raw = [x for (o, s, SS) in pts[mode] if b0 <= o < b1 for x in SS]
            if raw:
                lo, hi = bootstrap_ci(raw)
                row[mode] = (np.mean(raw), lo, hi)
        if not row:
            continue
        cells = "  ".join(f"{m}: {row[m][0]:5.1%}[{row[m][1]:.0%},{row[m][2]:.0%}]"
                          if m in row else f"{m}: --" for m in ['bandit', 'targeted', 'fixed'])
        verdict = ""
        if 'bandit' in row and 'targeted' in row:
            d = row['bandit'][0] - row['targeted'][0]
            overlap = not (row['bandit'][1] > row['targeted'][2] or row['targeted'][1] > row['bandit'][2])
            verdict = f"  bandit-vs-targeted {d:+.1%} ({'CI重叠' if overlap else 'CI分离'})"
        print(f"({b0:.2f},{b1:.2f})  {cells}{verdict}")
    with io.open(r"D:\Users\86151\Desktop\论文投稿\_cbandit_points.csv", "w", encoding="utf-8", newline='') as f:
        w = csv.writer(f); w.writerow(["strategy", "overhead", "success"])
        for m in pts:
            for o, s, _ in pts[m]:
                w.writerow([m, o, s])
    print(f"\n耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
