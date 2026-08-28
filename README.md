# Third Paper：Open-SCORE

当前工作区已经收敛为一个研究目录：[Open-SCORE](Open-SCORE/README.md)。历史 SAGA 方案、原始 HAD 顶层目录和旧 `02` 目录已从主分支移除；删除内容可在远程分支 `archive/pre-s1-full-20260828` 完整恢复。

> **一句话：Open-SCORE 学习可跨人数复用的小规模单目标攻防能力，再用可校准的局部胜负预测驱动双方实时分组，从而在伤亡、增援和未见总规模下维持全局胜率。**

快速验证：

```powershell
cd Open-SCORE
python -m pytest -q
python scripts/smoke_stage1.py
conda run -n gpu_py_310 python scripts/train_stage1_psro.py --mode pilot2v2 --device cuda
```

当前只实现并验证 S1；S2–S4 保留方法接口，但不能视为已完成实验。
