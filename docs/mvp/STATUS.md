# MVP 当前验收状态

2026-10-13：MVP-A（bag 单目标）和 MVP-B（toys 三目标）的技术闭环及用户动态视觉验收均通过。本轮技术 MVP 主闭环完成；归因仍需下一轮无生成补全对照，完整核对与例外见 [验收记录](ACCEPTANCE.md)。

| 检查项 | 当前证据与状态 |
|---|---|
| 删除集合一致 | 两轮修复入口逐元素核对，exact_match=true |
| 清场评测 | bag 58/58、toys 10/10 修复区 PSNR 改善 |
| 周边保持 | toys 非修复区汇总下降 0.282 dB，已披露 |
| 带物体留出 | bag 13、toys 15 个相机三组渲染，不作为清场 GT |
| 动态视频 | 每轮 240 帧，均已获得用户动态验收通过 |
| 分目标诊断 | toys 三目标各自 10/10 改善；区域为投影支持估计，非语义 GT |
| 运行资源 | 七步独立资源复跑全部退出 0，采样值不冒充历史或瞬时峰值 |
| 复查交付 | 模型身份、配置、冻结数组、Mask、日志、指标和视频已保存；七个脚本语法检查通过 |

[MVP-A 报告](results/bag-manual/README.md) · [MVP-B 报告](results/toys-manual/README.md) · [资源复跑](results/resource-replay/README.md)
