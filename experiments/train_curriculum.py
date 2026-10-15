"""课程学习版 MAPPO —— 两阶段训练，攻"学习策略画进 Pareto 图"这条腿。

阶段1（导航）：fixed 通信、frac=1.0（信息必然传开）、掩蔽 SEND 动作
             -> 策略专注学"绕缺口导航"，不被 SEND 决策的分摊干扰
阶段2（通信）：切 event 模式，解禁 SEND 动作
             -> 在已会导航的基础上学"何时值得花带宽"

关键实现：模型动作头始终 6 维（含 SEND），阶段1 用 logit 掩蔽禁止 SEND，
阶段2 解禁 —— 避免切换时重建网络导致阶段1 作废。
"""
import numpy as np, torch, torch.nn as nn, argparse, time, io
import evac_env as E
from _train_mappo import ActorCritic, update, evaluate, make_env, collect, DEV, GRID, N_AG


def collect_masked(model, mode, radius, frac, n_episodes=8, mask_send=False):
    obs_l, act_l, logp_l, rew_l, done_l, val_l = [], [], [], [], [], []
    for ep in range(n_episodes):
        env = make_env(mode, radius, frac, seed=200 + ep)
        obs, _ = env.reset()
        O = obs['obs']
        for t in range(env.max_steps):
            Ot = torch.as_tensor(O, dtype=torch.float32, device=DEV)
            with torch.no_grad():
                logits, v = model(Ot)
                if mask_send:
                    logits = logits.clone()
                    logits[:, 5] = -1e9          # 阶段1：禁止 SEND
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            obs, r, term, trunc, _ = env.step(a.numpy())
            obs_l.append(O); act_l.append(a.numpy())
            logp_l.append(dist.log_prob(a).numpy())
            rew_l.append(r); done_l.append(term or trunc); val_l.append(v.numpy())
            O = obs['obs']
            if term or trunc:
                break
    val_s = [float(np.mean(v)) for v in val_l]
    adv = np.zeros(len(rew_l), np.float32); gae = 0.0
    for t in reversed(range(len(rew_l))):
        nxt = val_s[t + 1] if t + 1 < len(val_s) and not done_l[t] else 0.0
        delta = rew_l[t] + 0.99 * nxt - val_s[t]
        gae = delta + 0.99 * 0.95 * gae
        adv[t] = gae
    ret = adv + np.array(val_s, np.float32)
    return (np.concatenate(obs_l), np.concatenate(act_l), np.concatenate(logp_l),
            np.repeat(adv, N_AG), np.repeat(ret, N_AG))


def main(iters=400, lr=3e-4):
    model = ActorCritic(E.OBS_DIM, 6).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    half = iters // 2
    log = []
    t0 = time.perf_counter()
    for it in range(iters):
        phase2 = it >= half
        mode = 'event' if phase2 else 'fixed'
        O, A, logp, adv, ret = collect_masked(model, mode, 4.0, 1.0,
                                              n_episodes=8, mask_send=not phase2)
        update(model, opt, O, A, logp, adv, ret)
        if (it + 1) % 25 == 0:
            tag = 'P2-event' if phase2 else 'P1-nav'
            s, o = evaluate(model, mode, 4.0, 1.0)
            line = f"[{tag}] iter {it+1:4d}  success={s:6.1%}  overhead={o:.2f}"
            print(line, flush=True); log.append(line)
    torch.save(model.state_dict(), r"D:\Users\86151\Desktop\论文投稿\_ckpt_curriculum.pt")
    io.open(r"D:\Users\86151\Desktop\论文投稿\_train_curriculum_log.txt", "w",
            encoding="utf-8").write("\n".join(log))
    print(f"课程训练完成，耗时 {time.perf_counter()-t0:.1f}s，模型已存 _ckpt_curriculum.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=400)
    a = ap.parse_args()
    main(a.iters)
