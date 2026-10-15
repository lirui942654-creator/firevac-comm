"""显著性检验补强 —— 从"CI 重叠判断"升级为正式假设检验。

审稿人可能问:"你只报了 CI,有没有做检验?" 本脚本补齐:
  * Mann-Whitney U(非参数,不假设正态;成功率是比例数据,适用)
  * Welch t 检验作为交叉验证
  * 报告 p 值 + Cohen's d + 检验结论
  * 同时给出 bootstrap CI 便于对照

注意:多重比较用 Holm-Bonferroni 校正(每个桶内多组对比)。
"""
import numpy as np, io, csv, sys
sys.path.insert(0, r"D:\Users\86151\Desktop\论文投稿")
from _sched_compare import bootstrap_ci

BASE = r"D:\Users\86151\Desktop\论文投稿"


def mannwhitney(a, b):
    """Mann-Whitney U 检验(正态近似,含并列校正),返回 (U, p)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(allv)).astype(float) + 1.0
    # 并列取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(allv))
    for i, c in enumerate(cnt):
        if c > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = ranks[idx].mean()
    R1 = ranks[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    # 并列校正的标准差
    N = n1 + n2
    tie = (cnt ** 3 - cnt).sum()
    sigma = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1))))
    z = (U - mu) / sigma
    # 正态双侧 p
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return U, p, z


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = (a.mean() - b.mean()) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    from math import erf, sqrt
    # 正态近似(df 大)
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, df, p


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / (pooled + 1e-9)


def load_seedlevel():
    """从 _final52 数据重建种子级样本(用 _v2_main.run 重跑关键对比,52种子)。"""
    import _v2_main as V
    print("重新采样种子级数据(52 seeds × 3 policy)...")
    samples = {}
    for mode, rad, p in [('targeted', 4.0, 0), ('fixed', 4.0, 0.3), ('roundrobin', 6.0, 0)]:
        S = [V.run(mode, rad, p, s)[0]['success'] for s in range(52)]
        samples[mode] = S
        print(f"  {mode}: mean={np.mean(S):.1%} n={len(S)}")
    return samples


def main():
    samples = load_seedlevel()
    out = []
    out.append("=== 显著性检验 (52 seeds, wall regime) ===")
    out.append("对比                       mean(A)  mean(B)   d     MWU-p      Welch-p   判定")
    pairs = [('targeted', 'fixed'), ('targeted', 'roundrobin'), ('roundrobin', 'fixed')]
    ps = []
    for a_name, b_name in pairs:
        a, b = samples[a_name], samples[b_name]
        U, p_mw, z = mannwhitney(a, b)
        t, df, p_w = welch_t(a, b)
        d = cohen_d(a, b)
        ps.append(p_mw)
        sig = "显著" if p_mw < 0.05 else "不显著"
        out.append(f"{a_name:12s} vs {b_name:12s} {np.mean(a):6.1%} {np.mean(b):6.1%} "
                   f"{d:5.2f}  {p_mw:.2e}  {p_w:.2e}  {sig}")
    # Holm-Bonferroni 校正
    order = np.argsort(ps)
    m = len(ps)
    out.append("\nHolm-Bonferroni 校正后:")
    for k, idx in enumerate(order):
        alpha = 0.05 / (m - k)
        p = ps[idx]
        name = f"{pairs[idx][0]} vs {pairs[idx][1]}"
        keep = "仍显著" if p < alpha else "不再显著"
        out.append(f"  {name:26s} p={p:.2e}  alpha={alpha:.4f}  -> {keep}")
    txt = "\n".join(out)
    print(txt)
    io.open(BASE + r"\_significance.txt", "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main()
