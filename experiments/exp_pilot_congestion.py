"""Pilot: 新方向可行性验证 —— 共享"出口排队长度"能否大幅提升存活率？

这是 go/no-go 闸门：若 A(无共享) vs B(有共享) 差异 < 15 个百分点 -> 效应量不足，放弃；
若 >= 30 点 -> 效应量充足，全量投入。

设计(刻意简化以隔离变量)：
  * 两出口在底部，每出口每步放行 CAP=1 人(制造真实拥堵瓶颈)
  * 两条件导航能力完全相同(全知BFS避火)——唯一差异是"是否知道各出口排队长度"
  * 火从中心蔓延，烟雾+FED 致死(ISO 13571 校准：每步 fed += 0.00857*smoke, 阈值0.30)
  * 度量：成功撤离比例、死亡比例、平均撤离步数
"""
import numpy as np

G, N, CAP, FED_DOWN, P_FIRE = 20, 8, 1, 0.30, 0.05
SEEDS = 30
EXITS = [(G - 1, 2), (G - 1, G - 3)]


def run(share, seed):
    rng = np.random.default_rng(seed)
    fire = np.zeros((G, G), bool)
    smoke = np.zeros((G, G))
    fire[G // 2, G // 2] = True
    pos = np.stack([rng.integers(0, 3, N), rng.integers(2, G - 2, N)], axis=1)
    fed = np.zeros(N)
    dead = np.zeros(N, bool)
    escaped = np.zeros(N, bool)
    choice = np.zeros(N, int)          # 上一轮选择的出口(作为排队代理)
    esc_time = np.zeros(N, int)

    def bfs_step(i, target):
        """一步移动：朝 target 走，绕开已知火(两条件导航能力相同)。"""
        r, c = pos[i]
        best, bestd = None, abs(r - target[0]) + abs(c - target[1])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = min(max(r + dr, 0), G - 1)
            nc = min(max(c + dc, 0), G - 1)
            if fire[nr, nc]:
                continue
            d = abs(nr - target[0]) + abs(nc - target[1])
            if d < bestd:
                bestd, best = d, (nr, nc)
        if best is not None:
            pos[i] = np.array(best)

    for t in range(300):
        # 火蔓延 + 烟雾
        nb = (np.roll(fire, 1, 0) + np.roll(fire, -1, 0)
              + np.roll(fire, 1, 1) + np.roll(fire, -1, 1))
        fire |= (nb > 0) & (rng.random((G, G)) < P_FIRE)
        smoke = np.clip(0.7 * smoke + 0.3 * (
            np.roll(smoke, 1, 0) + np.roll(smoke, -1, 0)
            + np.roll(smoke, 1, 1) + np.roll(smoke, -1, 1)) / 4 + 0.15 * fire, 0, 1)

        alive = np.where(~dead & ~escaped)[0]
        if len(alive) == 0:
            break
        # 排队长度(上一轮选择分布)
        q = [int(np.sum(choice[alive] == k)) for k in range(2)]
        # 选出口
        for i in alive:
            d0 = abs(pos[i, 0] - EXITS[0][0]) + abs(pos[i, 1] - EXITS[0][1])
            d1 = abs(pos[i, 0] - EXITS[1][0]) + abs(pos[i, 1] - EXITS[1][1])
            cost0 = d0 + q[0] if share else d0
            cost1 = d1 + q[1] if share else d1
            choice[i] = 0 if cost0 <= cost1 else 1
        # 移动
        for i in alive:
            bfs_step(i, EXITS[choice[i]])
        # FED(含排队等待时的暴露)
        for i in alive:
            r, c = pos[i]
            fed[i] += 0.00857 * smoke[r, c]
            if fed[i] > FED_DOWN:
                dead[i] = True
        # 出口放行(容量受限)
        for k, ex in enumerate(EXITS):
            at = [i for i in alive if tuple(pos[i]) == ex and not dead[i]]
            for i in at[:CAP]:
                escaped[i] = True
                esc_time[i] = t
        # 已到出口但超容量的 agent 停留 -> 继续累积 FED(排队代价)

    return dict(escaped=escaped.mean(), dead=dead.mean(),
                time=esc_time[escaped].mean() if escaped.any() else 999)


def main():
    print("=== Pilot: 出口拥堵信息共享的效应量 ===")
    print("(导航能力相同, 唯一差异 = 是否知道各出口排队长度)")
    res = {}
    for share in [False, True]:
        E_, D_, T_ = [], [], []
        for s in range(SEEDS):
            m = run(share, s)
            E_.append(m['escaped']); D_.append(m['dead'])
            if m['time'] < 999:
                T_.append(m['time'])
        res[share] = (np.mean(E_), np.mean(D_), np.mean(T_) if T_ else 0)
        tag = "B 有共享" if share else "A 无共享"
        print(f"{tag}: 撤离={np.mean(E_):6.1%}  死亡={np.mean(D_):6.1%}  "
              f"平均撤离步数={np.mean(T_) if T_ else 0:.0f}")
    gap = res[True][0] - res[False][0]
    dgap = res[False][1] - res[True][1]
    print(f"\n撤离率差异: {gap:+.1%}   死亡减少: {dgap:+.1%}")
    if gap >= 0.30:
        print(">>> GO: 效应量充足(>=30点),该方向值得全量投入")
    elif gap >= 0.15:
        print(">>> 中等: 效应量一般(15-30点),可做但需配合其他因素")
    else:
        print(">>> NO-GO: 效应量不足(<15点),放弃该方向")


if __name__ == "__main__":
    main()
