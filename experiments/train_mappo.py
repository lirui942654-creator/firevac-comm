"""MAPPO 训练器 —— 接 FireEvacEnv，训练"会通信"的疏散策略。

设计（吸取之前 numpy A2C 不收敛的教训）：
  * torch autograd（不再手写反向传播）
  * 共享 actor（每智能体同参数）+ 集中 critic（MAPPO 标准）
  * GAE 优势估计 + PPO clip
  * 环境侧已做势函数奖励塑形（向出口前进给正奖励），解决稀疏奖励冷启动
  * 事件触发版动作空间多一个 SEND，由策略学习"何时值得花带宽"

用法：
    python _train_mappo.py --mode event --iters 300      # 训练事件触发通信策略
    python _train_mappo.py --mode fixed --iters 300      # 训练固定频率通信策略（对照）
    python _train_mappo.py --eval                        # 训练后评估 Pareto 对比
"""
import numpy as np, torch, torch.nn as nn, argparse, time, io
import evac_env as E

DEV = torch.device('cpu')
GRID, N_AG, SEEDS = 20, 6, 20


def make_env(mode, radius, frac, seed):
    return E.FireEvacEnv(grid=GRID, n_agents=N_AG, scenario='wall', max_steps=120,
                         comm_mode=mode, comm_radius=radius, comm_frac=frac,
                         comm_loss=0.1, use_incap=False, seed=seed)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, x):
        with torch.no_grad():
            logits, v = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            return a, dist.log_prob(a), v


def collect(model, mode, radius, frac, n_episodes=8):
    obs_l, act_l, logp_l, rew_l, done_l, val_l = [], [], [], [], [], []
    for ep in range(n_episodes):
        env = make_env(mode, radius, frac, seed=100 + ep)
        obs, _ = env.reset()
        O = obs['obs']
        ep_obs, ep_act, ep_logp, ep_rew, ep_done, ep_val = [], [], [], [], [], []
        for t in range(env.max_steps):
            Ot = torch.as_tensor(O, dtype=torch.float32, device=DEV)
            a, logp, v = model.act(Ot)
            obs, r, term, trunc, _ = env.step(a.numpy())
            ep_obs.append(O); ep_act.append(a.numpy()); ep_logp.append(logp.numpy())
            ep_rew.append(r); ep_done.append(term or trunc); ep_val.append(v.numpy())
            O = obs['obs']
            if term or trunc:
                break
        obs_l += ep_obs; act_l += ep_act; logp_l += ep_logp
        rew_l += ep_rew; done_l += ep_done; val_l += ep_val
    # GAE（全局共享奖励 -> 每步标量；V 取各智能体均值）
    val_s = [float(np.mean(v)) for v in val_l]
    adv = np.zeros(len(rew_l), np.float32)
    gae = 0.0
    for t in reversed(range(len(rew_l))):
        nxt = val_s[t + 1] if t + 1 < len(val_s) and not done_l[t] else 0.0
        delta = rew_l[t] + 0.99 * nxt - val_s[t]
        gae = delta + 0.99 * 0.95 * gae
        adv[t] = gae
    ret = adv + np.array(val_s, np.float32)
    n_per_step = N_AG
    adv_tiled = np.repeat(adv, n_per_step)
    ret_tiled = np.repeat(ret, n_per_step)
    return (np.concatenate(obs_l), np.concatenate(act_l), np.concatenate(logp_l),
            adv_tiled, ret_tiled)


def update(model, opt, O, A, logp_old, adv, ret, epochs=4, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    Ot = torch.as_tensor(O, dtype=torch.float32, device=DEV)
    At = torch.as_tensor(A, dtype=torch.long, device=DEV)
    logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32, device=DEV)
    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=DEV)
    ret_t = torch.as_tensor(ret, dtype=torch.float32, device=DEV)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    idx = np.arange(len(Ot))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), 1024):
            b = idx[s:s + 1024]
            logits, v = model(Ot[b])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(At[b])
            ratio = torch.exp(logp - logp_old_t[b])
            pi_loss = -torch.min(ratio * adv_t[b],
                                 torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]).mean()
            v_loss = ((v - ret_t[b]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


def evaluate(model, mode, radius, frac, seeds=range(SEEDS)):
    S, O = [], []
    for s in seeds:
        env = make_env(mode, radius, frac, seed=1000 + s)
        obs, _ = env.reset()
        O_ = obs['obs']
        for _ in range(env.max_steps):
            Ot = torch.as_tensor(O_, dtype=torch.float32, device=DEV)
            a, _, _ = model.act(Ot)
            obs, r, term, trunc, info = env.step(a.numpy())
            O_ = obs['obs']
            if term or trunc:
                break
        S.append(info['metrics']['success']); O.append(info['metrics']['comm_overhead'])
    return np.mean(S), np.mean(O)


def train(mode, iters=200, lr=3e-4):
    obs_dim = E.OBS_DIM
    n_act = 6 if mode == 'event' else 5
    model = ActorCritic(obs_dim, n_act).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for it in range(iters):
        O, A, logp, adv, ret = collect(model, mode, radius=4.0, frac=1.0, n_episodes=8)
        update(model, opt, O, A, logp, adv, ret)
        if (it + 1) % 20 == 0:
            s, o = evaluate(model, mode, 4.0, 1.0)
            line = f"[{mode}] iter {it+1:4d}  eval success={s:6.1%}  overhead={o:.2f}"
            print(line); log.append(line)
    return model, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='event', choices=['event', 'fixed'])
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    model, log = train(a.mode, iters=a.iters)
    if a.save:
        torch.save(model.state_dict(), a.save)
        print(f"已保存 {a.save}")
    io.open(rf"D:\Users\86151\Desktop\论文投稿\_train_{a.mode}_log.txt", "w",
            encoding="utf-8").write("\n".join(log))
    print(f"训练耗时 {time.perf_counter()-t0:.1f}s（纯 CPU）")
