"""FireEvacEnv —— 通信受限动态火灾疏散环境（Gymnasium 兼容，纯 numpy）。

论文"两条腿"里开源 Benchmark 这条腿的核心工件：
  * 元胞自动机火蔓延 + 烟雾扩散 + FED（烟气有效剂量）累积失能
  * 可配置通信信道：半径 / 丢包 / 发送方式（固定频率 or 事件触发=动作）
  * 去中心化 POMDP：每个智能体只看局部火/烟 + 一跳通信
  * 标准 Gym 接口（reset/step/observation_space/action_space/reward）

两种通信模式（对应论文里的对照实验）：
  comm_mode='fixed'  每个智能体每步以 comm_frac 概率向半径内邻居广播（带宽与频率成正比）
  comm_mode='event'  动作空间多一个 SEND（=5）；智能体自己决定何时发送 —— 事件触发通信，
                     由训练出的策略学习"什么时候值得花带宽"（原题 event-triggered 卖点的可学习版）

两种场景：
  scenario='random' 随机火源（默认，泛化实验用）
  scenario='wall'   受控"火墙+单缺口"，只有 leader 知道出口坐标，通信决定出口可达性
                    （Pareto 主实验用，保证结论干净可复现）

依赖：numpy（可选 gymnasium，找不到时用内置最小兼容层）
"""
from __future__ import annotations
import numpy as np

try:
    import gymnasium as _g
    Spaces = _g.spaces
    class _EnvBase(_g.Env):
        pass
except Exception:  # pragma: no cover
    class _Box:
        def __init__(self, *a, **k):
            if len(a) == 3 and isinstance(a[0], tuple):
                self.shape, self.low, self.high = a[0], a[1], a[2]
            else:
                self.low, self.high, self.shape = a[0], a[1], a[2]
    class _Discrete:
        def __init__(self, n, *a, **k):
            self.n = n
    class _Dict:
        def __init__(self, d, *a, **k):
            self.spaces = d
    class Spaces:
        Box = _Box; Discrete = _Discrete; Dict = _Dict
    class _EnvBase:
        pass


UP, DOWN, LEFT, RIGHT, STAY, SEND = 0, 1, 2, 3, 4, 5
_DR = np.array([-1, 1, 0, 0, 0, 0])
_DC = np.array([0, 0, -1, 1, 0, 0])
PATCH = 5
OBS_PATCH = PATCH * PATCH * 2            # 火 + 烟
OBS_SELF = 2 + 1 + 2 + 1                 # 位置 + FED + 出口相对坐标 + 已知标志
OBS_COMM = 3                             # 邻居数(归一) + 邻居出口方向均值(2)
OBS_DIM = OBS_PATCH + OBS_SELF + OBS_COMM


class FireEvacEnv(_EnvBase):
    metadata = {"render_modes": []}

    def __init__(self, grid: int = 20, n_agents: int = 6, max_steps: int = 200,
                 comm_radius: float = 4.0, comm_loss: float = 0.1,
                 comm_frac: float = 1.0, comm_mode: str = 'fixed',
                 scenario: str = 'random', fed_down: float = 0.9,
                 use_incap: bool = False, seed: int = 0):
        super().__init__()
        assert comm_mode in ('fixed', 'event')
        assert scenario in ('random', 'wall')
        self.grid = grid
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.comm_radius = comm_radius
        self.comm_loss = comm_loss
        self.comm_frac = comm_frac
        self.comm_mode = comm_mode
        self.scenario = scenario
        self.fed_down = fed_down
        self.use_incap = use_incap
        self._seed = seed
        self.n_actions = 6 if comm_mode == 'event' else 5
        self.observation_space = Spaces.Dict({
            "obs": Spaces.Box(-1, 1, (n_agents, OBS_DIM), np.float32)})
        self.action_space = Spaces.Discrete(self.n_actions)
        self.reset(seed=seed)

    # ================= 初始化 =================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        g = self.grid
        self.fuel = np.ones((g, g), np.float32)
        self.fire = np.zeros((g, g), bool)
        self.smoke = np.zeros((g, g), np.float32)
        self.exit = (g - 1, g - 1)
        self.pos = self.rng.integers(1, max(2, g // 4), size=(self.n_agents, 2))
        self.fed = np.zeros(self.n_agents, np.float32)
        self.alive = np.ones(self.n_agents, bool)
        self.down = np.zeros(self.n_agents, bool)
        self.bel_exit = [None] * self.n_agents     # 信念中的出口坐标（None=未知）
        self.metrics = dict(n_send=0, n_recv=0, collisions=0, n_down=0)
        self.t = 0
        self._prev_dist = None

        if self.scenario == 'wall':
            wc = g // 2
            gap_lo, gap_hi = g // 2 - 2, g // 2 + 1
            for r in range(g):
                if not (gap_lo <= r <= gap_hi):
                    self.fire[r, wc] = True
            self.fuel[:] = 0.0                     # 火墙不蔓延，保证可逃生
            self.exit = (g // 2, g - 1)            # 出口正对缺口（不在墙上）
            self.pos[:, 1] = 1                     # 左侧边缘一列
            self.pos[:, 0] = np.linspace(2, g - 3, self.n_agents).astype(int)
            # leader：离出口最近的智能体初始知道出口
            lead = int(np.argmin(np.linalg.norm(self.pos - np.array(self.exit), axis=1)))
            self.bel_exit[lead] = tuple(self.exit)
        else:
            fr = self.rng.integers(g // 2, g - 2)
            fc = self.rng.integers(g // 2, g - 2)
            self.fire[fr, fc] = True
            self.bel_exit = [tuple(self.exit)] * self.n_agents   # 随机场景默认都知道出口

        self._prev_dist = self._dists()
        return self._obs(), self._info()

    def _dists(self):
        return np.linalg.norm(self.pos - np.array(self.exit), axis=1)

    # ================= 通信信道 =================
    def communicate(self, act=None):
        """一跳 gossip：传播出口坐标。fixed=按 comm_frac 概率发；event=动作 SEND 才发。"""
        send = self.alive & ~self.down
        will_send = np.zeros(self.n_agents, bool)
        if self.comm_mode == 'fixed':
            will_send[:] = (self.rng.random(self.n_agents) <= self.comm_frac) & send
        else:
            if act is not None:
                act = np.asarray(act).reshape(-1)
                will_send[:] = (act == SEND) & send
        for i in np.where(will_send)[0]:
            for j in np.where(send)[0]:
                if j == i:
                    continue
                if np.linalg.norm(self.pos[i] - self.pos[j]) <= self.comm_radius:
                    self.metrics["n_send"] += 1
                    if self.rng.random() < self.comm_loss:
                        continue
                    if self.bel_exit[i] is not None and self.bel_exit[j] is None:
                        self.bel_exit[j] = self.bel_exit[i]
                        self.metrics["n_recv"] += 1

    # ================= 单步 =================
    def step(self, action):
        g = self.grid
        act = np.asarray(action).reshape(-1)
        self.communicate(act)
        newpos = self.pos.copy()
        for i in np.where(self.alive & ~self.down)[0]:
            a = int(act[i]) if i < len(act) else STAY
            if a == STAY or a == SEND:
                continue
            r, c = self.pos[i]
            nr = int(np.clip(r + _DR[a], 0, g - 1)); nc = int(np.clip(c + _DC[a], 0, g - 1))
            if self.fire[nr, nc] or self._down_at(nr, nc):
                continue
            if self._occupied_by_other(nr, nc, i):
                self.metrics["collisions"] += 1
                continue
            newpos[i] = (nr, nc)
        self.pos = newpos
        # FED 累积 + 失能（ISO 13571 风格：CO 浓度-时间剂量积分）
        #   浓烟 s∈[0,1] 映射为 CO 当量 X_CO = 1200·s ppm（Purser & McAllister, SFPE Handbook
        #   给出浓烟 CO 典型量级 1000–2500 ppm）；失能剂量 35000 ppm·min（ISO 13571；
        #   另见 NRC 报告对 incapacitation dose 的引用）。每步暴露 3 s = 0.05 min。
        #   校准常数 K=5：浓烟（s≈0.86）滞留约 40 步（=120 s ≈ 2 min）失能，
        #   与 ISO 13571 的浓烟可承受窗口一致。
        for i in np.where(self.alive & ~self.down)[0]:
            s = self.smoke[self.pos[i, 0], self.pos[i, 1]]
            x_co = 1200.0 * s                       # ppm CO 当量
            dt_min = 0.05                           # 3 s per step
            K = 5.0                                 # calibrated: ~40 steps in dense smoke
            self.fed[i] += K * (x_co * dt_min) / 35000.0
            if self.use_incap and self.fed[i] > self.fed_down:
                self.down[i] = True
                self.metrics["n_down"] += 1
        # 火蔓延 + 烟雾
        nb = (np.roll(self.fire, 1, 0) + np.roll(self.fire, -1, 0)
              + np.roll(self.fire, 1, 1) + np.roll(self.fire, -1, 1))
        self.fire |= (nb > 0) & (self.rng.random((g, g)) < 0.10) & (self.fuel > 0)
        self.smoke = np.clip(0.7 * self.smoke + 0.3 * (
            np.roll(self.smoke, 1, 0) + np.roll(self.smoke, -1, 0)
            + np.roll(self.smoke, 1, 1) + np.roll(self.smoke, -1, 1)) / 4
            + 0.15 * self.fire, 0, 1)
        self.t += 1
        reached_now = np.array([tuple(self.pos[i]) == self.exit for i in range(self.n_agents)])
        terminated = bool((self.down | ~self.alive | reached_now).all())
        truncated = self.t >= self.max_steps
        reward = self._reward(reached_now)
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ================= 奖励（势函数塑形，可学习性关键） =================
    def _reward(self, reached_now):
        d = self._dists()
        alive_mask = self.alive & ~self.down
        r = 0.0
        if alive_mask.any():
            # 势函数：每向出口前进一步给正奖励（解决稀疏奖励冷启动）
            r += -0.05 * float((d[alive_mask] - self._prev_dist[alive_mask]).sum())
        r += 1.0 * float(reached_now.sum())          # 到达出口
        r += -0.02 * float(self.alive.sum())         # 时间惩罚
        r += -0.05 * float(self.metrics["n_down"])
        self._prev_dist = d
        return float(r)

    # ================= 观测 =================
    def _obs(self):
        g = self.grid
        O = np.zeros((self.n_agents, OBS_DIM), np.float32)
        for i in np.where(self.alive)[0]:
            r, c = self.pos[i]
            rf, cf = max(r - 2, 0), max(c - 2, 0)
            subf = self.fire[rf:rf + PATCH, cf:cf + PATCH].astype(np.float32)
            subs = self.smoke[rf:rf + PATCH, cf:cf + PATCH]
            k = 0
            O[i, k:k + subf.size] = subf.ravel(); k += PATCH * PATCH
            O[i, k:k + subs.size] = subs.ravel(); k += PATCH * PATCH
            O[i, k] = r / g; O[i, k + 1] = c / g
            O[i, k + 2] = self.fed[i]
            if self.bel_exit[i] is not None:
                er, ec = self.bel_exit[i]
                O[i, k + 3] = (er - r) / g; O[i, k + 4] = (ec - c) / g
                O[i, k + 5] = 1.0
            k += OBS_SELF
            # 邻居聚合
            cnt = 0; msg = np.zeros(2, np.float32)
            for j in np.where(self.alive & ~self.down)[0]:
                if j != i and np.linalg.norm(self.pos[i] - self.pos[j]) <= self.comm_radius:
                    cnt += 1
                    if self.bel_exit[j] is not None:
                        er, ec = self.bel_exit[j]
                        msg += [(er - r) / g, (ec - c) / g]
            O[i, k] = min(cnt, 8) / 8.0
            if cnt:
                O[i, k + 1:k + 3] = msg / cnt
        return {"obs": O}

    def _down_at(self, r, c):
        for i in np.where(self.down)[0]:
            if tuple(self.pos[i]) == (r, c):
                return True
        return False

    def _occupied_by_other(self, r, c, i):
        for j in np.where(self.alive & ~self.down)[0]:
            if j != i and tuple(self.pos[j]) == (r, c):
                return True
        return False

    def _info(self):
        alive_mask = self.alive & ~self.down
        reached = int(np.sum([tuple(self.pos[i]) == self.exit for i in range(self.n_agents) if self.alive[i]]))
        m = dict(
            success=reached / max(1, int(self.alive.sum())),
            survival=float(alive_mask.mean()) if len(alive_mask) else 0.0,
            avg_fed=float(self.fed[alive_mask].mean()) if alive_mask.any() else 0.0,
            comm_overhead=self.metrics["n_send"] / max(1, self.n_agents * max(1, self.t)),
            n_send=self.metrics["n_send"], n_reached=reached,
            steps=self.t, n_alive=int(self.alive.sum()),
        )
        return dict(metrics=m)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    env = FireEvacEnv(grid=20, n_agents=6, scenario='wall', comm_mode='event', seed=0)
    o, info = env.reset()
    print("wall 场景 obs 维度:", o["obs"].shape, "| 动作空间:", env.action_space.n)
    for step in range(env.max_steps):
        act = np.array([RIGHT if env.bel_exit[i] else STAY for i in range(env.n_agents)])
        o, r, term, trunc, info = env.step(act)
        if term or trunc:
            break
    print("10 步贪心冒烟测试:", {k: round(v, 3) for k, v in info['metrics'].items()}, f"{time.perf_counter()-t0:.2f}s")
