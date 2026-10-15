"""出版级论文图表 —— matplotlib 重制版（科研绘图规范）。

规范：
  * serif 字体（Times New Roman），与正文一致
  * 色盲友好色板（Wong 2011 Nature Methods 推荐）
  * CI 误差带（fill_between）而非误差棒
  * 300 dpi PNG + 矢量 PDF 双输出（PDF 投稿用，PNG 预览用）
  * 无网格垃圾、无图表垃圾（chartjunk），坐标轴留白合理

产出（供你在此基础上改风格）：
  fig1_pareto.pdf/png   —— 双场景 Pareto 主图（wall 52种子 + spreadfire 40种子）
  fig2_trap.pdf/png     —— silence trap 柱状图（低开销段 + 95% CI）
  fig3_halflife.pdf/png —— 信息半衰期边界（三策略 vs p_fire，52种子复测点）

你可以改的地方（改完即是你自己的风格）：
  COLORS 配色板 / marker 样式 / 线宽 / 图例位置 / 标注文字
"""
import numpy as np, csv, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------- 全局风格（可自行修改） ----------------
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Wong 色盲友好色板（Nature Methods 2011）—— 可换成你喜欢的
C = {
    'fixed':      '#0072B2',   # 蓝
    'event':      '#E69F00',   # 橙
    'targeted':   '#D55E00',   # 赭红
    'hybrid':     '#CC79A7',   # 粉紫
    'random':     '#009E73',   # 绿
    'roundrobin': '#56B4E9',   # 天蓝
    'bandit':     '#666666',   # 灰
    'oracle':     '#888888',
}
MARK = {'fixed': 'o', 'event': 's', 'targeted': 'D', 'hybrid': 'v',
        'random': '^', 'roundrobin': 'P', 'bandit': 'X'}
BASE = r"D:\Users\86151\Desktop\论文投稿"


def load_points(path):
    pts = {}
    with io.open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.setdefault(row['strategy'], []).append(
                (float(row['overhead']), float(row['success'])))
    return pts


# ================= 图 1: 双场景 Pareto =================
def fig1():
    wall = load_points(BASE + r"\_final52_points.csv")
    spread = load_points(BASE + r"\_spreadfire_full.csv")
    # event 低开销补测点
    wall.setdefault('event', []).extend([(0.077, 0.611), (0.276, 0.650)])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    order = ['targeted', 'roundrobin', 'fixed', 'event', 'random', 'hybrid', 'bandit']
    for ax, data, title in [
            (axes[0], wall, 'Regime D: discrete goal info (wall)'),
            (axes[1], spread, 'Regime F: continuous field info (spreading fire)')]:
        for mode in order:
            if mode not in data:
                continue
            pl = sorted(data[mode])
            o = [p[0] for p in pl]; s = [p[1] for p in pl]
            ax.plot(o, s, MARK[mode] + '-', color=C[mode], label=mode,
                    markersize=4, markeredgewidth=0.5)
        ax.axhline(1.0, color='#BBBBBB', lw=0.8, ls='--', zorder=0)
        ax.set_xlabel('Communication overhead\n(messages / agent / step)')
        ax.set_title(title, pad=4)
        ax.set_xlim(-0.15, 5.0)
        ax.set_ylim(0.55, 1.02)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Evacuation success')
    axes[0].text(0.03, 0.97, 'oracle', transform=axes[0].transAxes,
                 fontsize=7, color='#888888', va='top')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ['pdf', 'png']:
        fig.savefig(f"{BASE}\\fig1_pareto.{ext}")
    plt.close(fig)
    print("saved fig1_pareto.pdf/png")


# ================= 图 2: silence trap =================
def fig2():
    # 数据来自 _final52 主表低开销桶 (0,0.4) + event 补测
    bars = [('event', 0.611, 0.55, 0.67), ('fixed', 0.689, 0.64, 0.74),
            ('random', 0.768, 0.74, 0.80), ('roundrobin', 0.798, 0.75, 0.85),
            ('targeted', 0.862, 0.84, 0.88)]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = np.arange(len(bars))
    names = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    errs = [[b[1]-b[2] for b in bars], [b[3]-b[1] for b in bars]]
    ax.bar(xs, vals, width=0.62,
           color=[C[n] for n in names], edgecolor='black', linewidth=0.6)
    ax.errorbar(xs, vals, yerr=errs, fmt='none', ecolor='black',
                elinewidth=0.8, capsize=3, capthick=0.8)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.045, f"{v:.0%}", ha='center', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Evacuation success')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('Low-overhead regime (<0.4 msgs/agent/step)', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # trap 标注
    ax.annotate('', xy=(4, 0.875), xytext=(0, 0.63),
                arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.0,
                                connectionstyle='arc3,rad=-0.25'))
    ax.text(1.8, 0.80, 'silence trap:\n~20 pts (CIs disjoint)', fontsize=7.5,
            color='#D55E00', ha='center')
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f"{BASE}\\fig2_trap.{ext}")
    plt.close(fig)
    print("saved fig2_trap.pdf/png")


# ================= 图 3: 信息半衰期边界 =================
def fig3():
    # 数据：_halflife.py 52种子复测 + wall(∞) 端点 + p=0.20
    # (p_fire, fixed, targeted, roundrobin) —— p→0 近似 wall 端点取 fixed r4 θ1=88.5/targeted r8=99.0 不可比，
    # 统一取 r=4 口径：wall 场景 fixed(θ=1)=88.5, targeted(r=4)=86.2, rr(r=4)=79.8 -> 半衰期∞
    data = [
        (0.0,  0.885, 0.862, 0.798),   # ∞（wall，r=4 口径）
        (0.02, 0.961, 0.994, 0.939),
        (0.05, 0.865, 0.897, 0.885),
        (0.08, 0.747, 0.776, 0.772),
        (0.10, 0.712, 0.705, 0.702),
        (0.20, 0.311, 0.256, 0.283),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pf = [d[0] for d in data]
    for k, mode in enumerate(['fixed', 'targeted', 'roundrobin']):
        ys = [d[1 + k] for d in data]
        ax.plot(pf, ys, MARK[mode] + '-', color=C[mode], label=mode,
                markersize=4, markeredgewidth=0.5)
    ax.axvspan(0.06, 0.14, color='#F0E6D2', alpha=0.6, zorder=0)
    ax.text(0.10, 0.47, 'scheduling\nnot effective\n(CIs overlap)', fontsize=7,
            ha='center', color='#8A6D3B')
    ax.text(0.005, 0.47, 'scheduling\neffective', fontsize=7, ha='left',
            color='#2E5E4E')
    ax.set_xlabel('Fire spread probability $p_{fire}$\n(inverse information half-life)')
    ax.set_ylabel('Evacuation success')
    ax.set_ylim(0.2, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower left', fontsize=8)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f"{BASE}\\fig3_halflife.{ext}")
    plt.close(fig)
    print("saved fig3_halflife.pdf/png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
