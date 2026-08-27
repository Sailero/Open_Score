# Upstream sources and reproduction policy

第三方仓库不直接提交进本目录。正式实验在 upstream/external（已加入 .gitignore）中克隆，并为每个仓库记录 commit SHA、许可证、安装日志和必要 patch。

## 第一优先级

### ALMA

- Paper: https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html
- Code: https://github.com/shariqiqbal2810/ALMA
- License: MIT（GitHub 当前显示）
- Role: 科学骨架、AQL allocator、COPA/heuristic、SaveTheCity、SC2 multi-army。
- Gate: 三天内跑通环境和短 rollout；旧 Python 3.7 / SC2 依赖失败则 clean-room 迁移 allocator。

### JaxMARL / SMAX

- Paper: https://arxiv.org/abs/2311.10090
- Code: https://github.com/FLAIROx/JaxMARL （当前重定向到 bold-lab-ai/JaxMARL）
- Docs: https://jaxmarl.foersterlab.com/environments/smax/
- License: Apache-2.0（GitHub 当前显示）
- Role: 高吞吐 rollout、MAPPO/QMIX 基线和第二环境候选。

## 方法参考与 clean-room baseline

### Carion structured prediction

- Paper: https://proceedings.neurips.cc/paper_files/paper/2019/hash/3c3c139bd8467c1587a41081ad78045e-Abstract.html
- Role: unary/pairwise learned score + LP/QUAD。
- Independent repository: 本轮未检索到；按论文和补充材料 clean-room 实现。

### Double-IC

- Paper: https://doi.org/10.1016/j.eswa.2025.127271
- Role: 双方竞争联盟与 fictitious play / iterated best response。
- Code: 本轮未检索到官方代码；只复现与三路任务对应的最小版本并公开差异。

### HHMARL 2D

- Paper: https://arxiv.org/abs/2309.11247
- Code: https://github.com/IDSIA/hhmarl_2D
- Role: 先训练下层技能、后训练 commander 的接口参考。
- License warning: 2026-08-27 GitHub 页面未见明确 LICENSE；不复制或再分发其代码，许可确认前只 clean-room 借鉴论文思想。

## 外部规模环境

### MAgent2

- Code: https://github.com/Farama-Foundation/MAgent2
- Docs: https://magent2.farama.org/environments/battle/
- License: MIT
- Role: 大规模双方战斗/死亡压力测试。
- Platform: 官方当前说明支持 Linux/macOS；Windows 使用 WSL/Docker。

### SMACv2

- Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/764c18ad230f9e7bf6a77ffc2312c55e-Abstract-Datasets_and_Benchmarks.html
- Code: https://github.com/oxwhirl/smacv2
- Role: 程序化单位类型/位置泛化；默认固定规模和脚本敌方，不能单独证明开放人口。

## 版本记录模板

正式克隆成功后新增 upstream/LOCKS.md，逐项记录：

~~~text
name:
url:
commit:
license:
retrieved_at:
environment_or_method_used:
local_patch:
original_smoke_command:
original_smoke_result:
~~~

本轮命令行连接 GitHub 在提交前出现超时/重置，因此没有虚构 commit SHA，也没有把不完整 clone 当作复现资产。网页调研不替代后续 SHA 锁定。
