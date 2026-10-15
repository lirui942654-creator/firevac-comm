# GitHub Desktop 中文操作对照表

> GitHub Desktop 3.6.7 界面跟随 Windows 系统语言，无法在软件内切换中文。
> 本表给出常用按钮的中英文对照，你按中文名找即可。

## 发布仓库（你现在要做的）

| 界面英文 | 中文含义 | 位置 |
|---------|---------|------|
| File | 文件 | 左上角菜单 |
| Add local repository | 添加本地仓库 | File 菜单第 3 项 |
| Publish repository | 发布仓库 | 顶部工具栏按钮 |
| Push origin | 推送到远程 | 顶部工具栏 |
| Fetch origin | 从远程获取 | 顶部工具栏 |
| History | 历史记录 | 左侧标签页 |
| Changes | 更改 | 左侧标签页 |

## 发布流程（按顺序点）

1. 点左上角 **File**（文件）
2. 点 **Add local repository**（添加本地仓库）
3. 在路径框填：`D:\Users\86151\Desktop\firevac-comm`
4. 点 **Add repository**（添加）
5. 点顶部 **Publish repository**（发布仓库）
6. 弹窗里：
   - Name 保持 `firevac-comm`
   - **取消勾选** "Keep this code private"（保持私有）
   - 点 **Publish repository**（发布）

完成后仓库地址：https://github.com/lirui942654-creator/firevac-comm

## 如果 Push 失败（Connection was reset）

这说明当前网络到 GitHub 不通。解决办法按优先级：

1. **换网络**：切换到手机热点再试（最常见有效）
2. **用代理/VPN**：如果你有，开启后重试
3. **GitHub Desktop 重试**：Pull requests → 什么都不点，直接再点 Push origin
4. **稍后重试**：网络是间歇性的，过几小时可能就通了

## 之后常用操作

| 目的 | 操作 |
|------|------|
| 保存代码改动 | 左下角填 Summary（摘要）→ 点 Commit to main → Push origin |
| 看修改历史 | 左侧点 History（历史记录） |
| 撤销改动 | Changes 标签页里右键文件 → Discard changes |

## Zenodo 拿 DOI（代码推上去之后）

按 `docs/ZENODO.md` 走：GitHub 仓库页面 → Releases → Create release → Zenodo 自动分配 DOI。
