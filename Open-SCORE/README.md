# Open-SCORE：S1 / S2 可复现实验工程

Open-SCORE 当前执行一个两天最小方案：S1 只在 HAD 中训练一个覆盖六种人数规模的 QMIX；S2 只训练一个“全局态势 → Red 最终胜率”的普通 MLP。第二随机种子和 SMAClite/SMAClite-AD 都是有时间才做的附加项。

## 阅读入口

- **执行只看 [两天最小实验计划](docs/05_实验设计与预期证据.md)**：每一步都写明了类型、目标、输入、输出、固定参数和预期结果。
- [方法说明](docs/03_方法简版.md)及其 [HTML 浏览版](docs/03_方法简版.html)只描述包含 PSRO、S3、S4 在内的长期研究路线；它们不定义本轮实验，也不能覆盖协议。

若 README、背景文档、代码注释或历史记录与实验协议不一致，一律以 `docs/05_实验设计与预期证据.md` 为准。修改正式实验必须先修订该协议并留下 Git 记录，不能在运行过程中临时改变参数或验收口径。

## 仓库边界

- `configs/`：代码使用的配置；旧版完整方案仍在其中，但不定义两天 MVP；
- `scripts/`：安装、审计、训练、评估、汇总与绘图入口；
- `src/`：HAD、SMAClite-AD、Stage 1 和 Stage 2 实现；
- `tests/`：不产生科研结论的工程回归测试；
- `outputs/`：运行期临时文件，默认不纳入 Git；

## 仅做工程检查

以下命令只检查安装和代码，不是实验，也不能生成性能结论。正式运行命令必须从权威实验协议逐条复制。

```powershell
D:\Software\Anaconda\envs\torch310\python.exe -m pip install -e '.[stage2,test]'
D:\Software\Anaconda\envs\torch310\python.exe -m pytest -q
```

旧的短测试和历史报告不属于本次结果。两天 MVP 的所有结果统一放在 `outputs/two_day_mvp/`。
