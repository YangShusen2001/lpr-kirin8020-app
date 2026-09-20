# Git 对账状态

## 结论先说

**本仓库当前不需要对账。** `core.autocrlf=false` + `tools/parse_rq4.py` 起
推送一律走 `_push_api_range.ps1`（按提交范围推送、blob 由 `git cat-file` 原样上传），
本地与远端 **tree SHA 完全相同**，内容逐位等价。

## 分叉曾经是怎么产生的（已根治）

`github.com:443`（git 传输）在本机间歇性不可达，`api.github.com` 一直可达。
早期推送因此降级到 **Git Data API**，而当时的脚本按**工作区文件**重建整棵树
（`_push_api.ps1`）：`[IO.File]::ReadAllBytes` 读的是**检出后的文件**。
配合 `core.autocrlf=true` 与缺失的 `.gitattributes`，本地提交里存 LF、
API 推上去的成了 CRLF —— 于是每个 commit 都表现为「8 个文件 572 增 / 572 删」
的假差异，`git push` 报 `fetch first`。

| | 症状 |
|---|---|
| `git diff --ignore-cr-at-eol` | 无差异（说明**只是行尾**） |
| 本地 `^{tree}` vs 远端 `.tree.sha` | 不同 |

### 根治

1. `git config core.autocrlf false` —— 不再让检出/提交之间改写行尾。
2. 推送改用 `~/Desktop/_push_api_range.ps1`：
   - 只推送 `remote_head..local_head` 之间的提交；
   - 先 `git merge-base --is-ancestor` 校验是 fast-forward，否则拒绝改写历史；
   - blob 由 `git cat-file blob` + **cmd 重定向**取出（PowerShell 的 `>` 会
     把 LF 改写成 CRLF，是同一个坑的另一副面孔）；
   - 已存在的 blob 复用远端 SHA，不重复上传；
   - 用 `base_tree` 挂在父提交的树上，未改动的文件自然继承。
3. 推送后**用 tree SHA 校验**，而不是肉眼比 diff：
   ```powershell
   git rev-parse 'HEAD^{tree}'
   gh api repos/YangShusen2001/lpr-kirin8020-app/git/commits/<remote_sha> --jq '.tree.sha'
   ```
   两者相等即内容逐位等价。

## 历史上的一次分叉（已解决，仅存档）

`github.com:443` 不可达期间，`T7`（`984e3d5`）一度只落在本地。网络恢复后按
`git reset --hard origin/main` 对齐远端历史（内容等价，无损失），再
`git commit -C t7-local-backup` 重放同一个提交信息，推送成功（`6bc6fcb`）。

## 为什么尽量不用 `git push --force`

git 传输不可达时无法推送，API 路线是唯一选择；但 API 会**重建 commit**，
SHA 必然与本地不同。所以每次 API 推送后本地要跟上：

```bash
git fetch origin
git reset --hard origin/main     # 仅在 tree SHA 已核对相等后执行
```

## 文档仓库（~/Desktop/车牌识别）同样处理过

它此前也用 Contents API 推过 `docs/spec.md`，已用
`git rebase origin/main` 解决（重复提交被自动跳过），本地与远端一致。
