"""E7: 信息半衰期梯度实验 —— 论文的预测性规律图（最高杠杆动作）。

假设（由 E5 发现形式化）：
  调度策略间的成功率差异（design map 的"锐度"）随信息半衰期缩短而衰减。
  - 信息半衰期 ∝ 1/火蔓延速度：蔓延慢 -> 火图信息长期有效 -> 调度优化值得
  - 蔓延快 -> 信息迅速过时 -> 任何调度都无济于事

操作化：
  火蔓延概率 p_fire ∈ {0.02, 0.05, 0.10, 0.20} 四个梯度
  （p→0 即 wall 场景的极限，半衰期→∞）
  每个梯度测 3 策略 × 30 种子：fixed(0.3) / targeted / roundrobin
  因变量：调度优势 = max(策略) - min(策略)  （"锐度"）
  以及 targeted - fixed （低开销段最有代表性的差值）

预测（可证伪）：
  调度优势随 p_fire 单调递减；p_fire→0.20 时差异落入噪声（CI 重叠）。
  若成立 -> 论文获得一条预测性规律 + 一张核心图；
  若不单调 -> 主张再收窄，如实报告。
"""
import numpy as np, time, io, csv, sys
from collections import deque
sys.path.insert(0, r"D:\Users\86151\Desktop\论文投稿")
import evac_env as E
from _sched_compare import bootstrap_ci

GRID, SEEDS, LOSS = 20, 30, 0.1
FED_DOWN, SMOKE_RATE = 0.30, 0.15


def run(mode, radius, param, seed, p_fire):
    env = E.FireEvacEnv(grid=GRID, n_agents=6, max_steps=200,
                        comm_mode='fixed', comm_radius=radius, comm_frac=1.0,
                        comm_loss=LOSS, use_incap=True, fed_down=FED_DOWN, seed=seed)
    env.communicate = lambda act=None: None
    env.reset()
    g = GRID
    env.exit = (g - 1, g - 1)
    quads = [(2, 2), (g - 5, 2), (2, g // 2 - 4), (g - 5, g // 2 - 4),
             (g // 2 - 2, 2), (g // 2 + 2, 2)]
    for i in range(6):
        env.pos[i] = quads[i % len(quads)]
        env.pos[i, 0] += int(env.rng.integers(-1, 2))
        env.pos[i, 1] += int(env.rng.integers(-1, 2))
    env.fed[:] = 0; env.down[:] = False; env.alive[:] = True
    env.use_incap = True; env.fed_down = FED_DOWN
    env._p_fire = p_fire                    # 打补丁：自定义蔓延速度
    env._occupied_by_other = lambda r, c, i: env._down_at(r, c)

    bel_exit = [tuple(env.exit)] * 6        # 出口已知（信息=火图动态）
    belief = np.zeros((6, g, g), bool)
    bknown = np.zeros((6, g, g), bool)
    token = 0
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
        elif mode == 'targeted':
            for i in alive_idx:
                if any((~bknown[j] & bknown[i]).any() and
                       np.linalg.norm(env.pos[i] - env.pos[j]) <= radius
                       for j in alive_idx if j != i):
                    senders.append(i)
        elif mode == 'roundrobin':
            i = alive_idx[token % len(alive_idx)]
            token += 1
            senders.append(i)
        for i in senders:
            for j in alive_idx:
                if j == i: continue
                if np.linalg.norm(env.pos[i] - env.pos[j]) <= radius:
                    env.metrics["n_send"] += 1
                    if rng.random() < LOSS: continue
                    mask = bknown[i]
                    belief[j, mask] |= belief[i, mask]; bknown[j, mask] = True
        # FED
        for i in np.where(env.alive & ~env.down)[0]:
            r, c = env.pos[i]
            env.fed[i] += SMOKE_RATE * env.smoke[r, c]
            if env.fed[i] > FED_DOWN:
                env.down[i] = True
        acts = np.array([nav(i) for i in range(6)], np.int64)
        # ---- step with custom fire spread ----
        act = acts
        newpos = env.pos.copy()
        for i in np.where(env.alive & ~env.down)[0]:
            a = int(act[i]) if i < len(act) else E.STAY
            if a == E.STAY:
                continue
            r, c = env.pos[i]
            nr = int(np.clip(r + E._DR[a], 0, g - 1)); nc = int(np.clip(c + E._DC[a], 0, g - 1))
            if env.fire[nr, nc] or env._down_at(nr, nc) or env._occupied_by_other(nr, nc, i):
                continue
            newpos[i] = (nr, nc)
        env.pos = newpos
        for i in np.where(env.alive & ~env.down)[0]:
            env.fed[i] += SMOKE_RATE * env.smoke[env.pos[i, 0], env.pos[i, 1]]
            if env.fed[i] > FED_DOWN:
                env.down[i] = True
        nb = (np.roll(env.fire, 1, 0) + np.roll(env.fire, -1, 0)
              + np.roll(env.fire, 1, 1) + np.roll(env.fire, -1, 1))
        env.fire |= (nb > 0) & (env.rng.random((g, g)) < p_fire) & (env.fuel > 0)
        env.smoke = np.clip(0.7 * env.smoke + 0.3 * (
            np.roll(env.smoke, 1, 0) + np.roll(env.smoke, -1, 0)
            + np.roll(env.smoke, 1, 1) + np.roll(env.smoke, -1, 1)) / 4
            + 0.15 * env.fire, 0, 1)
        env.t += 1
        reached = np.array([tuple(env.pos[i]) == env.exit for i in range(6)])
        terminated = bool((env.down | ~env.alive | reached).all())
        if terminated:
            break
    # metrics
    alive_mask = env.alive & ~env.down
    reached = int(np.sum([tuple(env.pos[i]) == env.exit for i in range(6) if env.alive[i]]))
    return dict(success=reached / 6.0, survival=float(alive_mask.mean()))


def main():
    t0 = time.perf_counter()
    rows = []
    print("=== E7: 信息半衰期梯度（p_fire 越大=信息过时越快） ===")
    print(f"{'p_fire':>7} {'fixed':>16} {'targeted':>16} {'roundrobin':>16} {'max-min':>8} {'CI重叠?':>7}")
    for p_fire in [0.02, 0.05, 0.10, 0.20]:
        S = {}
        for mode, rad, p in [('fixed', 4.0, 0.3), ('targeted', 4.0, 0), ('roundrobin', 4.0, 0)]:
            ss = [run(mode, rad, p, s, p_fire)['success'] for s in range(SEEDS)]
            S[mode] = ss
        m = {k: np.mean(v) for k, v in S.items()}
        spread = max(m.values()) - min(m.values())
        # CI 重叠检查（最好 vs 最差）
        best = max(S, key=m.get); worst = min(S, key=m.get)
        b_lo, b_hi = bootstrap_ci(S[best]); w_lo, w_hi = bootstrap_ci(S[worst])
        overlap = not (b_lo > w_hi)
        row = [p_fire, m['fixed'], m['targeted'], m['roundrobin'], spread, overlap]
        rows.append(row)
        print(f"{p_fire:7.2f} {m['fixed']:6.1%}±{np.std(S['fixed']):.2f} "
              f"{m['targeted']:6.1%}±{np.std(S['targeted']):.2f} "
              f"{m['roundrobin']:6.1%}±{np.std(S['roundrobin']):.2f} "
              f"{spread:7.1%} {'是' if overlap else '否':>6}")
    # 判定
    print("\n=== 预测检验：调度优势（max-min）随 p_fire 单调递减？ ===")
    spreads = [r[4] for r in rows]
    monotonic = all(spreads[i] >= spreads[i+1] - 0.02 for i in range(len(spreads)-1))
    print(f"  锐度序列: {' -> '.join(f'{s:.1%}' for s in spreads)}")
    if monotonic:
        print("  >>> 单调递减成立：'调度价值随信息半衰期缩短而衰减' 获得支持，可写进论文")
    else:
        print("  >>> 非单调：预测不成立，主张需再收窄")
    with io.open(r"D:\Users\86151\Desktop\论文投稿\_halflife.csv", "w", encoding="utf-8", newline='') as f:
        w = csv.writer(f)
        w.writerow(["p_fire", "fixed", "targeted", "roundrobin", "spread", "ci_overlap"])
        for r in rows:
            w.writerow(r)
    print(f"\n耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
