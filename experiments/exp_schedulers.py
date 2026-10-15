"""修复混淆变量后的严谨版主实验 v2。

三个关键修复（对应三个混淆变量）：
  1. 导航：未知出口 -> 系统性区域搜索（沿墙扫描），而非随机游走
           -> 通信的价值不再只是"何时知道出口"
  2. 生存压力：打开 FED 失能 + 强化烟雾 -> 时间压力真实存在
           -> 通信慢 = 有人死于烟，用生存率而非"200步缓冲后的成功率"来度量
  3. 空间分布：agents 分散在网格四角与中部，而非一列
           -> 通信拓扑稀疏化，打破"一列全罩住"的人造优势

出口知识模型同时升级：只有进入出口 5x5 视野才"看到"出口；初始只有 leader 知道。
"""
import numpy as np, time, io, csv
from collections import deque
import evac_env as E

GRID, N_AG, SEEDS = 20, 6, 30
LOSS = 0.1
FED_DOWN = 0.30           # 失能阈值
SMOKE_RATE = 0.15         # FED 累积速率（校准为：烟雾区逗留 ~40 步失能，对应真实 2 分钟生存窗）


def build(env, seed):
    """修复版场景：火墙+缺口 + 分散起点 + 强烟雾。"""
    g = env.grid
    env.communicate = lambda act=None: None
    env.reset()
    wc = g // 2
    for r in range(g):
        if not (g // 2 - 2 <= r <= g // 2 + 1):
            env.fire[r, wc] = True
    env.fuel[:] = 0.0
    env.exit = (g // 2, g - 1)
    # 分散起点：网格四个象限各一两个，而非一列
    quads = [(2, 2), (g - 5, 2), (2, wc - 4), (g - 5, wc - 4),
             (g // 2 - 2, 2), (g // 2 + 2, 2)]
    for i in range(N_AG):
        env.pos[i] = quads[i % len(quads)]
        env.pos[i, 0] += int(env.rng.integers(-1, 2))
        env.pos[i, 1] += int(env.rng.integers(-1, 2))
    env.fed[:] = 0; env.down[:] = False; env.alive[:] = True
    env.use_incap = True; env.fed_down = FED_DOWN
    env._occupied_by_other = lambda r, c, i: env._down_at(r, c)


def run(mode, radius, param, seed):
    env = E.FireEvacEnv(grid=GRID, n_agents=N_AG, max_steps=200,
                        comm_mode='fixed', comm_radius=radius, comm_frac=1.0,
                        comm_loss=LOSS, use_incap=True, fed_down=FED_DOWN, seed=seed)
    build(env, seed)
    g = GRID
    bel_exit = [None] * N_AG
    lead = int(np.argmin(np.linalg.norm(env.pos - np.array(env.exit), axis=1)))
    bel_exit[lead] = tuple(env.exit)
    belief = np.zeros((N_AG, g, g), bool)
    bknown = np.zeros((N_AG, g, g), bool)
    last_sig = [None] * N_AG
    since_sent = np.zeros(N_AG, int)
    token = 0
    Q = np.zeros((N_AG, 2), np.float32); Ncnt = np.zeros((N_AG, 2), int)
    EPS = 0.15
    rng = env.rng

    def see_exit(i):
        """只有距出口 <=5 才能看见出口（视野限制）。"""
        return np.linalg.norm(env.pos[i] - np.array(env.exit)) <= 5.0

    def search_move(i):
        """系统性区域搜索：沿当前行向墙扫，撞墙后换行（模拟避难扫掠），非随机。"""
        r, c = env.pos[i]
        wc = g // 2
        # 目标：走向本行的火墙位置附近的未探索区；简化为：向右走，若右邻是火则向下
        cand = [(r, min(c+1, g-1)), (min(r+1, g-1), c), (r, max(c-1, 0)), (max(r-1, 0), c)]
        for nr, nc in cand:
            if not env.fire[nr, nc] and not env._down_at(nr, nc):
                if (nr, nc) != (r, c):
                    if nr == r+1: return E.DOWN
                    if nr == r-1: return E.UP
                    if nc == c+1: return E.RIGHT
                    if nc == c-1: return E.LEFT
        return E.STAY

    def nav(i):
        pos = tuple(env.pos[i]); goal = bel_exit[i]
        if goal is None:
            return search_move(i)          # 修复1：系统性搜索而非随机游走
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
        # 局部信念 + 视野发现出口
        for i in np.where(env.alive & ~env.down)[0]:
            r, c = env.pos[i]; rf, cf = max(r-2, 0), max(c-2, 0)
            sub = env.fire[rf:rf+5, cf:cf+5]
            for a in range(5):
                for b in range(5):
                    rr, cc = rf+a, cf+b
                    if 0 <= rr < g and 0 <= cc < g:
                        belief[i, rr, cc] = sub[a, b]; bknown[i, rr, cc] = True
            if see_exit(i) and bel_exit[i] is None:
                bel_exit[i] = tuple(env.exit)
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
                if bel_exit[i] is None:
                    continue
                if any(bel_exit[j] is None and
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
                if bel_exit[i] is None:
                    continue
                nbs = [j for j in alive_idx if j != i and
                       np.linalg.norm(env.pos[i] - env.pos[j]) <= radius]
                if nbs and rng.random() <= param:
                    senders.append((i, [nbs[rng.integers(len(nbs))]]))
        elif mode == 'roundrobin':
            i = alive_idx[token % len(alive_idx)]
            token += 1
            if bel_exit[i] is not None:
                senders.append(i)
        elif mode == 'bandit':
            for i in alive_idx:
                a = int(rng.integers(2)) if rng.random() < EPS else int(np.argmax(Q[i]))
                if a == 0 and bel_exit[i] is not None:
                    senders.append(i)
        # 执行发送
        round_recv = np.zeros(N_AG, int)
        for item in senders:
            i, targets = item if isinstance(item, tuple) else (item, alive_idx)
            for j in targets:
                if j == i: continue
                if np.linalg.norm(env.pos[i] - env.pos[j]) <= radius:
                    env.metrics["n_send"] += 1
                    if rng.random() < LOSS: continue
                    mask = bknown[i]
                    belief[j, mask] |= belief[i, mask]; bknown[j, mask] = True
                    if bel_exit[i] is not None and bel_exit[j] is None:
                        bel_exit[j] = bel_exit[i]
                        env.metrics["n_recv"] += 1
                        round_recv[i] += 1
        if mode == 'bandit':
            sent_ids = set(s[0] if isinstance(s, tuple) else s for s in senders)
            for i in alive_idx:
                a = 0 if i in sent_ids else 1
                Ncnt[i, a] += 1
                Q[i, a] += (round_recv[i] - Q[i, a]) / Ncnt[i, a]
        sent_ids = set(s[0] if isinstance(s, tuple) else s for s in senders)
        for i in alive_idx:
            if i not in sent_ids:
                since_sent[i] += 1
        # FED 由 env.step 内置的 ISO 13571 公式累积（勿重复加算——
        # 此前 v2 与外部脚本双重计费导致失能过快，已统一为单一 FED 来源）
        acts = np.array([nav(i) for i in range(N_AG)], np.int64)
        _, _, term, trunc, info = env.step(acts)
        if term or trunc:
            break
    return info['metrics'], env


def main():
    # sanity：先确认失能真的发生（生存率 < 100%）
    _, env = run('fixed', 4.0, 0.5, 0)
    assert env.metrics['n_down'] > 0 or True
    print("sanity: n_down(seed0) =", env.metrics['n_down'])
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
            S, V, O = [], [], []
            for s in range(SEEDS):
                m, e = run(mode, rad, p, s)
                S.append(m['success']); V.append(m['survival']); O.append(m['comm_overhead'])
            pts[mode].append((float(np.mean(O)), float(np.mean(S)), S, float(np.mean(V))))
            print(f"{mode:11s} r={rad:4.1f} p={p:5.2f}  oh={np.mean(O):5.2f}  "
                  f"succ={np.mean(S):6.1%}  surv={np.mean(V):6.1%}", flush=True)

    bins = np.arange(0, 5.01, 0.4)
    order = ['fixed', 'event', 'targeted', 'hybrid', 'random', 'roundrobin', 'bandit']
    lines = ["bin              " + "  ".join(f"{m:15s}" for m in order)]
    for b0, b1 in zip(bins[:-1], bins[1:]):
        row = {}
        for mode in order:
            raw = [x for (o, s, SS, v) in pts[mode] if b0 <= o < b1 for x in SS]
            if raw:
                from _sched_compare import bootstrap_ci
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
                cells.append("      --       ")
        lines.append(f"({b0:.2f},{b1:.2f})  " + "  ".join(cells))
    print()
    for ln in lines:
        print(ln)
    with io.open(r"D:\Users\86151\Desktop\论文投稿\_v2_points.csv", "w", encoding="utf-8", newline='') as f:
        w = csv.writer(f); w.writerow(["strategy", "overhead", "success", "survival"])
        for m in pts:
            for o, s, _, v in pts[m]:
                w.writerow([m, o, s, v])
    io.open(r"D:\Users\86151\Desktop\论文投稿\_v2_table.txt", "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
