# 独立资源复跑

这些是新运行的步骤资源测量，不是历史基线峰值，也不是从新训练模型贯穿全部步骤的一次重训练。训练、蒸馏复跑各自保存在独立目录；删除与修复使用已冻结的原基线输入，以保持可比性。

| 步骤 | 耗时（秒） | 采样进程显存峰值（MiB） |
|---|---:|---:|
| deletion-resource.json | 5.27 | 838 |
| distillation-resource.json | 45.71 | 3192 |
| repair/fusion-resource.json | 14.72 | 534 |
| repair/lama-color-resource.json | 4.74 | 1320 |
| repair/lama-depth-resource.json | 4.72 | 1320 |
| repair/optimization-resource.json | 127.66 | 1564 |
| training-resource.json | 18.40 | 878 |

每 0.5 秒通过 nvidia-smi 采样进程及其同 session 子进程，排除其他服务。采样峰值可能漏掉采样间短暂尖峰，不能称为严格瞬时峰值。JSON 保留 UTC 开始/结束、命令、退出状态与误差；压缩包保存完整样本及日志。修复删除核对 exact_match=true，删除 7088/211981。

复跑用于补齐资源可行性证据，不替代用户已验收的质量结果；历史资源缺失仍保留为事实。
