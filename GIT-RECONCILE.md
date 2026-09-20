# Git 对账状态

## 当前状态（2026-09-21）

`github.com:443`（git 传输）间歇性不可达，`api.github.com` 可达。
因此本次推送走的是 **Git Data API**，造成本地与远端历史分叉：

| | SHA | 说明 |
|---|---|---|
| 本地 HEAD | `cc62ad5` | 直接 `git commit` 产生，无 parent |
| 远端 main | `cd8a72f` | API 产生的 commit，parent 是种子提交 `1b8abba`（README.md） |

**两者的 tree 内容完全相同**（远端 tree 就是按本地 `git ls-files` 的文件列表与内容构建，
60 个 blob 逐一比对过）。差异只在提交历史，不在内容。

## 网络恢复后的对账命令

```bash
cd /c/Users/26671/lpr-kirin8020-app
git fetch origin
git reset --hard origin/main     # 丢弃本地 cc62ad5，采用远端历史；内容相同，无损失
```

## 为什么不用 `git push --force`

git 传输不可达时无法推送。API 路线已验证可用，且内容一致，
所以对账只需在传输恢复后执行一次 `fetch + reset`。

## 文档仓库（~/Desktop/车牌识别）同样处理过

它此前也用 Contents API 推过 `docs/spec.md`，已用
`git rebase origin/main` 解决（重复提交被自动跳过），本地与远端一致。
