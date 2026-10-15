"""多场景泛化实验 —— 证明 Pareto 结论不是单一场景的巧合。

场景变体：
  1. wall（主场景）：火墙 + 单缺口 + 仅 leader 知出口
  2. wall + 随机火源位置（火墙起点扰动）
  3. 多出口：两个出口（主出口 + 备用出口），任一到达即成功
  4. 不同智能体数：4 / 8 / 12
所有变体下，比较 "固定频率通信 vs 事件触发通信" 在同等开销下的成功率。
"""
import numpy as np, time, io
import evac_env as E
from _pareto import GRID

SEEDS = 15


def run_variant(mode, radius, frac, n_agents=6, n_exits=1, fire_jitter=0, seed=0):
    """mode: 'fixed'|'event'; n_exits: 1 或 2; fire_jitter: 火墙起点行扰动。"""
    env = E.FireEvacEnv(grid=GRID, n_agents=n_agents, max_steps=200,
                        comm_mode=mode, comm_radius=radius, comm_frac=frac,
                        comm_loss=0.1, use_incap=False, seed=seed)
    env.communicate = lambda act=None: None
    env.reset()
    g = env.grid
    # 场景覆盖：火墙（可扰动）+ 缺口 + 多出口
    wc = g // 2
    gap_lo, gap_hi = g // 2 - 2, g // 2 + 1
    env.fire[:] = False
    for r in range(g):
        if not (gap_lo <= r <= gap_hi):
            env.fire[r, wc] = True
    env.fuel[:] = 0.0
    if fire_jitter:
        # 把火墙整体向缺口方向平移 fire_jitter 行（扰动缺口位置）
        shift = fire_jitter
        env.fire[:] = False
        for r in range(g):
            gap = (gap_lo + shift <= r <= gap_hi + shift)
            if not gap and 0 <= r < g:
                env.fire[r, wc] = True
    env.exit = (gap_lo + (fire_jitter if fire_jitter else 0), g - 1)
    exits = [env.exit] if n_exits == 1 else [env.exit, (gap_hi + (fire_jitter if fire_jitter else 0) + 1, g - 1)]
    env.pos[:, 1] = 1
    env.pos[:, 0] = np.linspace(2, g - 3, n_agents).astype(int)
    env.fed[:] = 0.0; env.down[:] = False; env.alive[:] = True
    env.bel_exit = [None] * n_agents
    env.fed_down = 0.9; env.use_incap = False
    env._occupied_by_other = lambda r, c, i: env._down_at(r, c)
    env._exits = exits
    # 覆盖 exit/终止判断以支持多出口
    primary = exits[0]
    env.exit = primary
    orig_info = env._info
    def multi_info():
        info = orig_info()
        reached = sum(1 for i in range(n_agents) if env.alive[i] and tuple(env.pos[i]) in exits)
        info['metrics']['success'] = reached / max(1, int(env.alive.sum()))
        info['metrics']['n_reached'] = reached
        return info
    env._info = multi_info
    # 修正终止条件：任一 exit 到达即算
    bel_exit = [None] * n_agents
    lead = int(np.argmin([min(np.linalg.norm(np.array(env.pos[i]) - np.array(e)) for e in exits) for i in range(n_agents)]))
    bel_exit[lead] = exits[0]
    rng = env.rng
    # —— 部分可观测：每智能体维护"信念火图"，只看局部 5x5 + 通信融合 ——
    belief_fire = np.zeros((n_agents, g, g), bool)
    belief_known = np.zeros((n_agents, g, g), bool)

    def update_belief():
        for i in np.where(env.alive & ~env.down)[0]:
            r, c = env.pos[i]
            rf, cf = max(r - 2, 0), max(c - 2, 0)
            sub = env.fire[rf:rf + 5, cf:cf + 5]
            for a in range(5):
                for b in range(5):
                    rr, cc = rf + a, cf + b
                    if 0 <= rr < g and 0 <= cc < g:
                        belief_fire[i, rr, cc] = sub[a, b]
                        belief_known[i, rr, cc] = True

    def step_all():
        acts = np.zeros(n_agents, np.int64)
        for i in np.where(env.alive & ~env.down)[0]:
            if bel_exit[i] is None:
                acts[i] = int(rng.integers(5))
                continue
            from collections import deque
            pos = tuple(env.pos[i]); goal = bel_exit[i]
            parent = {pos: None}; q = deque([pos]); found = False
            while q:
                cur = q.popleft()
                if cur == goal:
                    found = True; break
                r, c = cur
                for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<g and 0<=nc<g and (nr,nc) not in parent and not belief_fire[i, nr, nc]:
                        parent[(nr,nc)] = cur; q.append((nr,nc))
            if not found:
                acts[i] = int(rng.integers(4)); continue
            cur = goal
            while parent[cur] is not None and parent[cur] != pos:
                cur = parent[cur]
            dr, dc = cur[0]-pos[0], cur[1]-pos[1]
            acts[i] = (E.DOWN if dr==1 else E.UP if dr==-1 else E.RIGHT if dc==1 else E.LEFT if dc==-1 else E.STAY)
        return acts

    def gossip():
        send = env.alive & ~env.down
        for i in np.where(send)[0]:
            if mode == 'fixed':
                if rng.random() > frac: continue
            else:
                # event：只在刚学到新信息时发（简化模拟事件触发）
                if bel_exit[i] is None: continue
                if getattr(env, '_sent_flag', {}).get(i): continue
                env._sent_flag = getattr(env, '_sent_flag', {}); env._sent_flag[i] = True
            for j in np.where(send)[0]:
                if j == i: continue
                if np.linalg.norm(env.pos[i] - env.pos[j]) <= radius:
                    env.metrics["n_send"] += 1
                    if rng.random() < 0.1: continue
                    # 融合信念火图（部分可观测下的信息共享）
                    mask = belief_known[i]
                    belief_fire[j, mask] |= belief_fire[i, mask]
                    belief_known[j, mask] = True
                    if bel_exit[i] is not None and bel_exit[j] is None:
                        bel_exit[j] = bel_exit[i]
                        env.metrics["n_recv"] += 1

    # BFS 在"信念火图"上规划（部分可观测），非真实火图
    for _ in range(env.max_steps):
        update_belief()
        gossip()
        acts = step_all()
        _, _, term, trunc, info = env.step(acts)
        if term or trunc:
            break
    return info['metrics']


def main():
    out = ["variant            mode   radius  succ     send/ag"]
    print(out[-1])
    for variant, kw in [
        ("wall(n=6,1出口)",        dict()),
        ("wall(n=6,2出口)",        dict(n_exits=2)),
        ("wall(n=6,缺口扰动)",      dict(fire_jitter=2)),
        ("wall(n=4)",             dict(n_agents=4)),
        ("wall(n=8)",             dict(n_agents=8)),
        ("wall(n=12)",            dict(n_agents=12)),
    ]:
        for mode in ['fixed', 'event']:
            for radius in [4.0, 8.0]:
                S, O = [], []
                for s in range(SEEDS):
                    m = run_variant(mode, radius, 1.0, seed=s, **kw)
                    S.append(m['success']); O.append(m['comm_overhead'])
                row = f"{variant:22s} {mode:6s} r={radius:4.1f}  {np.mean(S):6.1%}  {np.mean(O):6.2f}"
                print(row); out.append(row)
    io.open(r"D:\Users\86151\Desktop\论文投稿\_generalize_result.txt", "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n耗时 {time.perf_counter()-t0:.1f}s")
