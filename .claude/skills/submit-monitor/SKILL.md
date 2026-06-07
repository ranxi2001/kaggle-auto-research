---
name: submit-monitor
description: >
  Submit predictions to Kaggle and track leaderboard performance.
  Validates submission format, submits via API, records scores, analyzes CV-LB gap.
  Trigger: "submit", "提交", "check leaderboard", "排名", "track score",
  "submission history", "提交历史", "compare submissions".
---

# Submit & Monitor Skill

自动提交预测结果并追踪排名。

## Workflow

1. **生成预测**
   - 加载最佳模型 from `models/`
   - 在测试集上推理
   - 对集成模型执行加权平均

2. **格式验证**
   - 对照 `sample_submission.csv` 验证格式
   - 检查：列名、行数、数值范围、缺失值
   - 阻止不合格提交

3. **智能提交**
   - 检查是否超过每日提交限制 (`max_daily`)
   - 检查是否满足提升阈值 (`best_threshold`)
   - 通过 Kaggle API 提交: `kaggle competitions submit`

4. **分数追踪**
   - 获取提交分数（API 轮询）
   - 更新 `submissions/history.json`
   - 记录：时间戳、模型版本、特征版本、CV 分数、LB 分数

5. **差距分析**
   - CV vs LB 分数差距分析
   - 过拟合/欠拟合检测
   - 分数趋势可视化

## Usage

```bash
kar submit <competition-name>
kar submit <competition-name> --force         # 跳过阈值检查
kar submit <competition-name> --dry-run       # 只验证不提交
kar submit <competition-name> --history       # 查看历史
```

或在 Claude Code 中：
```
> 提交最新模型的预测结果
> 看一下提交历史和分数趋势
> 分析一下 CV 和 LB 的 gap
```

## Output

```
submissions/
├── sub_001_lgbm_v1.csv      # 提交文件
├── sub_002_ensemble_v2.csv
└── history.json              # 提交历史记录
    [
      {
        "id": "sub_001",
        "timestamp": "2026-06-06T10:00:00",
        "model_version": "v001",
        "features_version": "v002",
        "cv_score": 0.812,
        "lb_score": 0.798,
        "rank": 156
      }
    ]
```
