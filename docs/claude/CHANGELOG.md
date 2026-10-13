# 调研文档修订记录

本文件记录历史判断和修订原因；执行计划以 [05-roadmap.md](05-roadmap.md) 为准，源码差异证据以 [06-verification.md](06-verification.md) 为准。

## 2026-10-13 · 结构整理

- 四阶段编号固定为 1–4；PLY-only 基础编辑归入阶段 1，修复深化归入阶段 2，自动候选与仿真能力归入未编号的后续扩展。
- 合并阶段 2 的准备、步骤和验收职责，评测协议集中维护。
- 移除重复 A/B/C/C+ 输入标签及整体质量汇总；能力、操作可行性、监督来源与质量验证结果分开。
- M1 定义为编辑基线；M2 定义为官方修复基线；M3 明确为外部选择到背景修复的成功闭环；M4 为按能力验收的产品交付。

## 2026-10-13 · 接口与基线边界

- 增加 SceneAsset、SelectionArtifact、RecoveryArtifact、CleanSceneAsset 和 RemovalAdapter。
- NativeBaseline 独立于编辑器；ExternalSelection 待实现并单独验证。
- B0/B1 分别为仅删除和官方默认修复；B2/B3 为真实背景观测优先恢复、冻结保护和受限优化增强。
- SuperSplat 官方快照为 3.5.2，旧版重排信息不作为 3.x 已验证行为。

## 2026-10-12 · 源码校正

本套文档经过一次**源码校正**。Inpaint360GS 官方代码已公开，经逐条对照，**早期版本有三处主要错误判断**，已修正；文档中的待验证项和假设仍以各章节标注为准：

| # | 早期错误表述 | 更正后 |
|---|---|---|
| 1 | GS-IoU 是"开放集合下的无先验识别方案" | **GS-IoU 是关联器不是检测器**，依赖 HQSAM + 人工 `target_id` |
| 2 | "有 `cameras.json` 就接近满血模式" | **拆成独立能力字段**；`renderable` ≠ `source_supervised` ≠ `has_holdout` ≠ `has_clean_reference` |
| 3 | "低可见度 Gaussian 是残留碎片的主因，可扩散删除" | **低可见度 ≠ 干扰物**，只能作不确定性分数 |

**另发现两处论文与代码不一致（本次新发现）**：

- **论文式 2 的 GS-IoU 定义与代码不同**：论文是标准 IoU，代码实际执行 `IOU_heighlight`（覆盖率的非线性变换），且 `"IOU"` 分支是死代码，阈值也是 0.1 而非论文的 0.2
- **KMeans 逻辑含 16×16 网格单元 + 30% 截断**，论文未描述

详见 [06-verification.md](06-verification.md)。

---

