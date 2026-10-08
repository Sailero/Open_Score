# LEAF main1009 轻量运行版

本分支 `leaf/main1009-lite` 保留 main1009 五个循环候选的完整训练、评估和报告流程。运行源码取自 `leaf/main1009-dual-route` 的 `dc04965`，算法、1M 步预算、种子 0/1/2 与原 HAD 接口不变。

只包含运行源码、配置、上游许可证和执行指南。无旧实验输出、模型权重、论文附件、网页或父提交历史。使用远端已经配置好的环境即可。

```bash
git clone --depth 1 --single-branch --branch leaf/main1009-lite https://github.com/Sailero/Open_Score.git LEAF1009-lite
cd LEAF1009-lite/Open-SCORE
python scripts/leaf1009.py plan --suite loop_candidates
python scripts/leaf1009.py train --devices all --jobs-per-gpu 4 --resume
```

每张 48GB 4090 同时运行四个任务；共五个方法、三个种子，15 个训练任务。多服务器共用一个代码与输出目录时，各服务器执行同一训练命令，任务由共享文件锁领取。

训练和各评估阶段共同生成 `Open-SCORE/outputs/main1009`：模型、恢复点、日志、逐场景数据、汇总、图表及唯一正式报告。已有训练需在所有阶段指定原来的 `--output /已有绝对路径/main1009`，保留进度。

完成全部 15 个训练任务后，按 [执行指南](docs/LEAF_main1009_循环候选与远端指南.md) 依次运行校准、选型及评估。候选流程无需旧实验；旧基线比较需要通过 `--source-output` 指向已有 main0928 的相应权重、配置和完成记录。
