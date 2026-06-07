---
name: local-dev
description: >
  本地 commit + git bundle 打包 + 拷贝到共享目录，替代 git push 的本地开发方案。
  当用户说"打包""发过去""bundle""收包""拆包""本地开发"时触发。
  也可通过关键词触发：打包提交、离线开发、git bundle、收取代码等。
---

# 本地开发技能（local-dev）

公司限制直接推送，因此使用 **git bundle + 共享文件夹** 实现两台电脑之间的代码搬运。

核心原理：`git bundle` 将 git 提交打包成单个文件，拷贝到共享目录后在另一端还原，效果等同于 push/pull。

---

## 前置配置（首次使用需完成）

### 方案一：store.py（Google API，推荐）

基于 `backup/` 目录下的已有脚本改造，通过 Google API 直接 put/get 文件。

**安装依赖**：
```bash
pip install google-auth google-auth-oauthlib requests
```

**配置认证**：
1. 将 `credentials.json`（OAuth 2.0 Client）放到 `.claude/skills/local-dev/` 目录
2. 运行 `python3 .claude/skills/local-dev/gen_token.py` 生成 `token.json`
3. 检查状态：`python3 .claude/skills/local-dev/store.py setup`

配置好后，`pack.sh` 和 `unpack.sh` 会自动检测 `token.json` 并使用 store.py。

**手动使用 store.py**：
```bash
python3 .claude/skills/local-dev/store.py put <file>          # 放文件
python3 .claude/skills/local-dev/store.py get <name>           # 取指定文件
python3 .claude/skills/local-dev/store.py latest               # 取最新 bundle
python3 .claude/skills/local-dev/store.py ls                   # 列出远端文件
python3 .claude/skills/local-dev/store.py setup                # 检查配置状态
```

### 方案二：共享文件夹

在两台电脑都能访问的共享文件夹下创建 `packs/` 子目录，用于存放 bundle 文件。

### 配置文件

首次使用时在项目根目录创建 `.pack.conf`（可选，不配也能用 store.py）：

```bash
# 共享文件夹本地路径（方案二需要）
SHARED_PATH="$HOME/Library/CloudStorage/xxx/packs"

# store.py 远端文件夹名（默认 packs）
REMOTE_FOLDER="packs"

# 本地暂存目录
STAGING_DIR=".packs"
```

将 `.pack.conf` 和 `.packs/` 加入 `.gitignore`（已添加）。

---

## 发送端工作流（打包 → 拷出）

用户说"打包""发过去""pack"时执行以下步骤：

### 步骤 1：确认并提交

```bash
git status
```

如有未提交的修改，先帮用户 commit（遵循正常 commit 流程）。

### 步骤 2：运行打包脚本

```bash
bash .claude/skills/local-dev/pack.sh
```

脚本会自动：
- 判断增量/全量
- 创建 git bundle
- 验证 bundle
- 通过 store.py 或共享目录发出（按优先级自动选择）
- 记录本次打包点

也可加参数：
```bash
bash .claude/skills/local-dev/pack.sh --full          # 强制全量
bash .claude/skills/local-dev/pack.sh --no-copy        # 只打包不拷贝
bash .claude/skills/local-dev/pack.sh --branch dev     # 指定分支
```

### 步骤 3：输出摘要

告知用户：
- 包文件名和大小
- 包含的 commit 范围
- 拷贝状态
- 接收端操作提示

---

## 接收端工作流（拷入 → 还原）

用户说"收包""拆包""unpack""还原"时执行以下步骤：

### 步骤 1：运行还原脚本

```bash
bash .claude/skills/local-dev/unpack.sh
```

脚本会自动：
- 从 store.py 远端、共享目录和本地缓存收集当前仓库的 bundle
- 按时间从旧到新验证并合并可用 bundle
- fetch + fast-forward/merge 到当前分支
- 更新记录

也可加参数：
```bash
bash .claude/skills/local-dev/unpack.sh --file path/to.bundle   # 指定文件
bash .claude/skills/local-dev/unpack.sh --list                   # 只列出可用包
bash .claude/skills/local-dev/unpack.sh --no-merge               # 只获取不合并
```

### 步骤 2：输出摘要

告知用户：
- 合并了哪些 commit
- 当前 HEAD 状态
- 是否有冲突需要解决

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 打包发送 | `bash .claude/skills/local-dev/pack.sh` |
| 收包还原 | `bash .claude/skills/local-dev/unpack.sh` |
| 查看包内容 | `git bundle list-heads <file.bundle>` |
| 验证包 | `git bundle verify <file.bundle>` |
| 查看上次记录 | `cat .packs/.last-pack-hash` |
| 清理旧包 | `ls -la .packs/*.bundle` |

---

## 注意事项

- `.packs/` 和 `.pack.conf` 已在 `.gitignore` 中；`pack.sh` 首次运行时会自动追加（如果缺失）
- 增量包依赖接收端有对应的前置提交，如果接收端落后太多，用全量包
- 包文件包含完整的 git 对象，注意存放位置的安全性
- 如果两端都有新提交（分叉），合并时可能产生冲突，需手动解决
- 定期清理 `.packs/` 目录中的旧包文件
- `credentials.json` 和 `token.json` 不入 git（被 `*.json` 规则覆盖）
