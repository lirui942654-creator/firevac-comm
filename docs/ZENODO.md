# Zenodo 发布指南

把本仓库发布为可引用的研究工件(Zenodo 会分配 DOI)。

## 步骤

1. **准备版本归档**
   ```bash
   cd D:\Users\86151\Desktop\firevac-comm
   # 清理临时文件后打 zip
   ```

2. **GitHub 发布**(推荐顺序:先 GitHub 后 Zenodo)
   - 在 GitHub 创建公开仓库 `firevac-comm`
   - push 全部内容
   - 打 tag:`git tag v1.0.0 && git push origin v1.0.0`

3. **Zenodo 联动**
   - 访问 https://zenodo.org/login → 用 GitHub 账号登录
   - 启用 `firevac-comm` 仓库的 Zenodo 开关
   - 在 GitHub 发布一个 Release(基于 v1.0.0 tag)
   - Zenodo 自动抓取并分配 DOI

4. **填写 Zenodo 元数据**
   - Title: FireEvac-Comm: A Communication-Constrained Fire Evacuation Benchmark
   - Description: 复制 README 的"Why this exists"段
   - License: MIT
   - Keywords: MARL, evacuation, communication constraints, FED, benchmark
   - Related works: Chen+ TMLR 2026 (Five Ws survey); ISO 13571

5. **引用格式**(拿到 DOI 后填入 README)

## 发布前检查清单

- [ ] 删除所有中文注释?不必——注释保留原文即可,但 README 需完整英文
- [ ] `python experiments/exp_main_52seeds.py` 在全新机器上可跑通?
- [ ] requirements.txt 版本号是否足够具体(torch 是否必需?标为可选?)
- [ ] LICENSE 文件存在?
- [ ] results/ 里的 CSV 是否与脚本输出列一致?
