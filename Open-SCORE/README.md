# Open-SCORE：S1 / S2 可复现实验工程

Open-SCORE 当前只执行两个阶段：S1 在未经修改的 SMAClite、独立 SMAClite-AD 和自研 HAD 上复现并训练规则/智能策略；S2 用冻结的 HAD 前向推演数据训练可校准、可解释的局部评估器。PSRO、S3 和 S4 暂不进入本轮实验与验收。

## 阅读入口

- **执行只看 [S1 / S2 唯一权威实验协议](docs/05_实验设计与预期证据.md)**：环境安装、上游版本、Git 冻结、场景顺序、算法、超参数、种子、数据切分、验收门槛、预期结果、失败处理和最终产物均以此为准。
- [方法说明](docs/03_方法简版.md)及其 [HTML 浏览版](docs/03_方法简版.html)只描述包含 PSRO、S3、S4 在内的长期研究路线；它们不定义本轮实验，也不能覆盖协议。

若 README、背景文档、代码注释或历史记录与实验协议不一致，一律以 `docs/05_实验设计与预期证据.md` 为准。修改正式实验必须先修订该协议并留下 Git 记录，不能在运行过程中临时改变参数或验收口径。

## 仓库边界

- `configs/`：协议引用的机器可读配置；
- `scripts/`：安装、审计、训练、评估、汇总与绘图入口；
- `src/`：HAD、SMAClite-AD、Stage 1 和 Stage 2 实现；
- `tests/`：不产生科研结论的工程回归测试；
- `outputs/`：运行期临时文件，默认不纳入 Git；
- `artifacts/final/`：正式实验完成后才允许保留的精简结果包。

## 仅做工程检查

以下命令只检查安装和代码，不是实验，也不能生成性能结论。正式运行命令必须从权威实验协议逐条复制。

```powershell
D:\Software\Anaconda\envs\torch310\python.exe -m pip install -e '.[stage2,test]'
D:\Software\Anaconda\envs\torch310\python.exe -m pytest -q
```

旧 pilot、smoke、dry-run、单 seed 曲线和历史报告均不属于正式证据。只有满足协议中的来源、完整性、独立测试、样本量、稳定性和统计门槛后，结果才可进入最终报告。
