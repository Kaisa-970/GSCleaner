# 04 · 架构设计

> 提案：Capability-Tiered Pipeline
>
> 整理日期：2026-10-12 ｜ 状态：**提案，待评审，非已批准设计**
>
> 本文档基于 [00-requirements.md](00-requirements.md) 的五层模型与两条已确认约束（开放干扰物集合 + 双数据边界）。

---

## 一、设计目标

架构要同时满足三个约束：

1. **用户指定删除或保留目标** → 归属层输出可检查、可修正的 Gaussian 集合；自动候选排序作为后续能力
2. **同时支持 PLY-only 与完整 COLMAP 输入** → 能力差异要集中处理，不能散落各处
3. **降级路径必须显式** → 不允许静默劣化

**最容易踩的坑**：在代码各处写 `if has_images:`。那样三个月后必然维护两套半成品。

---

## 二、核心结构：统一数据协议与逐步扩展的执行层

首版采用能力清单、少量 Python 模块和可追溯实验配置；下图中的 Planner / OperationPlan 是流程稳定后的扩展设计，不作为首轮实验前置。五层模型描述问题类型，不要求五个串行优化模块。

```
[Ingest]     能力探测 → CapabilityDescriptor
                 │
[Planner]    能力 → OperationPlan（每步带 feasible / quality_grade）
                 │   ↑ 集中处理所有降级逻辑，且必须显式报告降级原因
                 │
[Execute]    L0 Ingest → L1 Analyze → L2 Remove → L3 Repair → L4 Relight → L5 Export
                 │        ↑ 各层只认 OperationPlan，不认数据来源
                 │
[Report]     输出 Mask（一等产物）+ 干净 PLY + 质量报告 + 降级警告
```

### 稳定数据接口（首版协议提案，尚未实现）

`SceneAsset → SelectionArtifact → RemovalAdapter → RecoveryArtifact → CleanSceneAsset`

| 产物 | 必需内容 |
|---|---|
| `SceneAsset` | Gaussian 文件/格式、资产哈希/版本/谱系、原图及标定引用、坐标系/单位、能力清单；缺失项显式记录 |
| `SelectionArtifact` | 绑定源资产的 Target、Keep、Remove、Protected 集合，2D mask 与相机引用，源索引/稳定 ID 和编辑映射，人工确认与选择来源 |
| `RecoveryArtifact` | 背景分区及 mask、观测来源（原图/渲染/生成）、置信度、可训练 ID/参数、冻结和过渡带范围、后端/配置/转换记录 |
| `CleanSceneAsset` | 输出 PLY/版本与身份谱系、编辑历史、RecoveryArtifact 引用、质量报告、验证状态与不可验证项 |

四类产物均带 `schema_version`、唯一 artifact ID、关联资产哈希和生产工具版本。相机引用包含内外参及坐标约定；无真实尺度时单位标记 unknown。mask 极性、索引所属版本与选区冲突必须显式检查；Protected 与 Remove 相交时停止并要求修正。适配器只能产生本项目协议，算法核心不直接依赖编辑器内部 state。

### RemovalAdapter：选择到修复后端的适配契约

| 模式 | 输入与行为 | 状态 |
|---|---|---|
| `NativeBaseline` | 固定官方 commit、官方支持数据/配置、对象特征/分类器、2D mask 与人工核验目标 ID；沿用官方关联、删除及修复，记录实际删除集合 | 优先复现，独立于 GSCleaner 编辑器；尚未实跑 |
| `ExternalSelection` | SceneAsset + 用户确认的 SelectionArtifact；验证身份映射并生成后端所需数据，或修改删除/准备步骤直接消费外部集合 | 待开发与验证；不是已有格式转换能力 |

官方 `run_inpaint.sh` 不承诺接受任意 `remove_mask_3d`。外部编辑器导出的删后 PLY 也不能替代官方准备流程：它可能缺少被删除 Gaussian、对象特征、图像 mask、相机与场景配置。

`ExternalSelection` 的验证步骤：

1. 校验源资产身份、Gaussian 布局/数量/映射、相机坐标系与后端 checkpoint 的一致性；不能仅凭相同点数套用索引。
2. 检查官方下游实际读取哪些对象特征、目标/周边对象 ID、原始/删后模型、图像 mask 和配置；记录固定 commit 的依赖清单。
3. 区分候选实现：重建官方兼容对象特征/ID，或改造删除与数据准备函数直接消费外部集合。选择方案后逐步实测；不假定“把索引写成 target_id”即可工作。
4. 若由外部集合渲染多视角 mask，明确目标投影、alpha 阈值、遮挡和修复区域推导；外部目标集合不等于 repair mask。
5. 输出带模式、源哈希、后端 commit、转换参数和产物路径的适配清单；失败明确返回缺失项。B0 与修复分支必须校验实际删除集合一致，否则不能归因补全收益。
6. 在小场景核验背景保护、集合/相机对齐和下游可运行，再允许阶段 4 接入。仅“文件能读取”不算适配完成。

阶段 2 首轮使用 NativeBaseline 或官方兼容且人工核验的输入，不等待阶段 1 的 UI 或身份映射。ExternalSelection 在阶段 3–4 单独验收。

## 三、CapabilityDescriptor：能力探测

**项目第一步不是写算法，是探测能力。**

见 [00-requirements.md](00-requirements.md) 第四节约束 2：很多所谓"只有 PLY"其实不只有 PLY。这一步用于发现真实监督和渲染能力；找到相机文件本身不等于升级为 Tier A。

### 更正：能力探测应拆成独立字段，而非单一 Tier

早期版本写过"有 `cameras.json` 就够做渲染 → 直接升到接近满血模式（Tier B）"。**这个判断过强，已更正。**

有效 PLY 可以用虚拟相机渲染并在渲染图上跑 SAM；原始相机位姿和内参用于对齐真实影像。但这**与拥有原始影像不是一回事**：

| 缺失项 | 后果 |
|---|---|
| 真实 RGB | 没有真实的训练监督信号 |
| 训练时的 object feature / classifier | **不能直接使用 Inpaint360GS 的对象移除模型** |
| 真实图像约束 | 2D inpainting 补的是**渲染结果**，不是真实观测 |
| 配准干净参考 | 无法直接验证隐藏背景是否真实还原；holdout 用于新视角检查，不等于干净参考 |
| 渲染图自身的伪影 | 分割/修复的输入已经带误差 |

#### 拆分为独立能力字段

因此探测输出**一组独立布尔字段**，Tier 只是它们的**汇总标签**，不是原始判据：

| 字段 | 含义 | 影响的下游操作 |
|---|---|---|
| `renderable` | 有效 Gaussian 资产与兼容渲染器可用，可自设虚拟相机 | 虚拟视角、NBS 判定、渲染图分割 |
| `segmentable` | 是否有原图或可接受的渲染图可分割 | 2D lifting |
| `source_supervised` | 是否有原图和位姿，可重建 / 重优化 | 3D 重优化、真实监督 |
| `retrainable` | 是否有兼容训练器和训练配置 | 训练期 mask、局部重优化 |
| `has_holdout` | 是否有未参与优化的校准视角与影像 | 新视角表现；带物体的 holdout 不是干净背景 GT |
| `has_clean_reference` | 是否有配准的干净背景参考 | 隐藏背景还原指标 |
| `has_source_cameras` / `has_virtual_cameras` | 原始相机标定 / 已配置虚拟相机 | 区分真实观测与自渲染 |
| `has_metric_scale` | 是否已校准真实尺度 | 深度融合、碰撞几何与仿真对齐 |
| `relightable` | 是否有可用的几何、材质和光照表示 | 阴影清理、逆渲染 |
| `has_object_features` | 是否已有每 Gaussian 的对象特征 | 能否直接用现成对象移除模型 |

**准确表述**："无原图的有效 PLY（有或无原始相机）" 应描述为 **「可用虚拟相机渲染和弱分割，缺少真实影像监督」**。

### 关键判据（修正版）

> 原始位姿不是 `renderable` 的必要条件。有效 PLY 可用自设的虚拟相机渲染；原始标定用于对齐真实图像，而不是决定能否渲染。
>
> **但 `renderable` 不等于 `source_supervised`，也不等于 `has_holdout` / `has_clean_reference`。** 真实影像和干净参考缺失时，仍可检查残留、背景保持度和多视角一致性，但不能验证隐藏背景是否真实还原。

### 探测输出结构

```json
{
  "renderable": true,
  "segmentable": true,
  "source_supervised": false,
  "retrainable": false,
  "has_holdout": false,
  "has_clean_reference": false,
  "has_source_cameras": false,
  "has_virtual_cameras": true,
  "has_metric_scale": false,
  "relightable": false,
  "has_object_features": false,

  "sh_degree": 3,
  "gaussian_count": 1234567,
  "has_training_config": false,
  "per_gaussian_stats": {"visibility": false, "gradient": false}
}
```

`per_gaussian_stats` 的说明见第五节——**普通 PLY 不含这些统计，需自行渲染累计**。

---

## 四、OperationPlan：声明式操作计划

`OperationPlan` 不只描述"做什么"，还要描述**"为什么只能这样做"**。

**这是整套设计的关键。** 它让降级路径**被迫显式**——不会静默劣化；下游消费者可以按 grade 决定是否接受；加新能力（比如接入 COLMAP）只改探测和 planner。

### 结构示例（Tier C，仅 PLY）

```json
{
  "capability": {"renderable": true, "source_supervised": false, "retrainable": true, "has_virtual_cameras": true},
  "ops": [
    {"op": "render_virtual_views", "feasible": true, "quality": "full"},
    {"op": "semantic_lift_2d", "feasible": true, "quality": "degraded", "reason": "rendered_images_only"},
    {"op": "candidate_rank", "feasible": true, "quality": "degraded", "effect": "选择集合需人工确认"},
    {"op": "hole_repair_gen", "feasible": true, "quality": "degraded", "reason": "pseudo_supervision_only"},
    {"op": "source_reoptimize", "feasible": false, "quality": "unavailable", "reason": "no_registered_source_images"},
    {"op": "export_ply", "feasible": true, "quality": "full"}
  ],
  "overall_grade": "pseudo_supervised_experimental",
  "warnings": ["可检查残留、保持度和多视角一致性；缺少干净参考，不能验证隐藏背景真实性"]
}
```

以上为渲染器、兼容训练器均可用时的计划；缺失任一组件时应明确禁用对应操作，不根据 PLY-only 标签一律禁用。

### 字段约定

| 字段 | 含义 |
|---|---|
| `feasible` | 该操作在当前能力下是否可执行 |
| `quality` | `full` / `degraded` / `unavailable` |
| `reason` | 不可行或降级的原因（机器可读） |
| `effect` | 降级对最终结果的实际影响（人可读） |
| `overall_grade` | 整体质量等级，供下游决策 |
| `warnings` | 必须向用户显示的限制 |

**`warnings` 不是日志，是交付物的一部分。** 对应 [00-requirements.md](00-requirements.md) 第六节"需求阶段必须向业务方说清的三件事"。

---

## 五、PLY-only 模式的识别：四类可用信号

PLY-only 若可渲染，可使用虚拟视角分割与 2D lifting，但输入伪影会传播到标签，且缺少真实监督。以下几何与统计信号是补充候选依据，不替代人工确认：

### 信号 1：可见性与资产统计（不确定性依据）

> **重要更正（源码/格式验证后）**：本节早期版本暗示"可从 PLY 读出每个 Gaussian 的被观测质量"。**这个前提不成立，已更正。**

**普通 PLY 不含任何训练期统计。** 标准 PLY 布局只有：

```
x y z / f_dc_0..2 / f_rest_0..M / opacity / scale_0..2 / rot_0..3
```

没有每个 Gaussian 的训练视角计数、梯度累积或 visibility 统计。

**因此要得到 $v_i$ 必须自己重新渲染所有视角并累计每个 Gaussian 的 alpha 贡献** —— 这是一个可做但有成本的操作，不是"读一下 PLY"。

| 子信号 | 是否需要额外计算 |
|---|---|
| per-Gaussian 视角计数 $v_i$ | **需要重渲染所有视角累计 $\alpha_i T_i$** |
| SH 能量分布 | 可从 PLY 直接算（`f_rest_*` 提供了高阶系数） |
| scale / opacity 异常 | 可从 PLY 直接算 |

#### ⚠ 关键更正：低可见度 ≠ 干扰物

早期版本把"低可见度 Gaussian"直接当作待删除/待扩散的证据。**这是错的，而且这个错误会直接导致误删背景。**

**低可见度有大量合法成因**：

- **被物体遮挡的背景**（这正是我们要保留的）
- **墙角、背光面、背面结构**
- **细杆、线缆、栏杆**等薄结构
- **采集轨迹覆盖不到的区域**

**因此低可见度的正确定位是「不确定性分数」，而不是「删除证据」。**

> 它应该用来**触发人工确认**，而不是**自动判定删除**。

这与 [00-requirements.md](00-requirements.md) 第四节"候选生成 + 排序 + 确认"的原则一致——**低可见度是排序信号，不是决策信号。**

#### 关于 $\mathcal{L}_{space}$ 的正确用法

早期版本写过"用 KNN 给低可见度点扩散标签"。**这个表述也不准确。**

Inpaint360GS 的 $\mathcal{L}_{space}$ 是**对象特征蒸馏阶段的空间正则**（见 [03-inpaint360gs.md](03-inpaint360gs.md) 机制 2），它需要：

- 一个正在优化的 object feature 场 $f$
- 与之配套的 2D 监督 $\mathcal{L}_{obj}$
- 完整的训练流程

**它不是在已有 PLY 上做后处理式的标签扩张。** 对一个已经训练好的 PLY，你**没有**这套 feature 场，也就无法直接"应用 $\mathcal{L}_{space}$"。

**正确表述**：

> **在对象特征学习中加入空间一致性正则，并对低置信度区域保留人工确认。**

若要在已有 PLY 上使用，需要重新引入 feature 场并重新蒸馏——这是后续特征蒸馏阶段的工作量，不是免费的后处理。

### 信号 2：几何先验

- **地面平面 RANSAC 拟合** → 每个 Gaussian 的**高度**
- **连通域分析**（box 或 Gaussian 距离下的层次聚类）→ 物体 blob
- **悬浮判据**：不与任何大面积表面连通、且不在地面上的 blob → 高可疑（浮空物体通常就是拍摄期间移动或临时放置的东西）
- **支撑关系**：blob 与地面的接触足迹

> **对仓库、工作台这类结构化场景，这套先验意外地有效** —— 因为干扰物往往**在语义上不连贯于场景结构**（一把临时放置的椅子 vs 结构性的货架）。

### 信号 3：局部统计断裂

一个真正属于背景的表面，其局部几何和颜色统计应与周围一致（同一面墙、同一块地板）。干扰物会在**接缝处产生统计断裂**。

**做法**：对每个 blob 计算其与邻域的几何/颜色相似度，取阈值。

**对"覆盖在背景上的物体"特别有效。**

### 信号 4：有图时的开放词汇（加分项，非依赖）

若可渲染：渲染若干虚拟视角 → CLIP/VLM 打标签 → 打分排序 → 人工确认。

**必须接受它召回不稳定，尤其是小物体。** 因此设计成**排序信号**，而非**决策信号**——与 [01-landscape.md](01-landscape.md) 第 2.6 节一致。

---

## 六、Mask 是一等产物

这一条是 PLY-only 和开放干扰物集合**共同逼出来的**。

> **管线必须把 `mask.json` / `mask.ply`（keep/remove 索引 + score + reason）作为正式产物，而不只是中间变量。**

**理由**：候选排序必然有误，用户需要能看、能改、能重跑。

**这个接口让"自动 + 人工确认"成为产品的一部分，而不是 bug 的补丁。** 在开放干扰物集合下，**这是唯一能让系统真正可用的设计**。

### 三类 mask 与保护区域

- `target_mask_2d`：用户希望删除的目标区域，按视角记录；不是直接删除指令。
- `remove_mask_3d`：映射并经复核后实际删除的 Gaussian 集合。
- `repair_mask_2d`：删除后需要修复的可见区域，独立计算。
- `protected_mask_2d` / `protected_ids_3d`：禁止误删或应保持的背景；阴影等效果区域另行记录。

必须保存相机 ID、标定或虚拟相机参数，以及 mask 到源资产的关联。先检查删除结果，再检查补全，不能把人工目标等同于正确的 3D 删除集合。

### 资产身份与编辑映射

`gaussian_count` 不能证明资产身份。最小协议使用源资产 SHA-256、版本和源索引；消费 mask 时校验哈希，不匹配则拒绝套用。删除、过滤、重排、合并或导出后保存新索引到旧索引的 sidecar 映射。禁用重排仍不能解决过滤后索引变化。

需要跨多轮优化跟踪时使用稳定 ID：split/clone 保存父 ID，新增补全 Gaussian 分配新 ID；生成资产记录父版本。映射未知时标注失效，不用坐标近邻冒充精确身份。选择集必须在过滤前绑定源身份。

### 建议的 mask 格式

```json
{
  "scene_id": "...",
  "asset_hash": "sha256:...",
  "asset_version": 1,
  "id_scheme": "source_index",
  "gaussian_count": 1234567,
  "candidates": [
    {
      "id": 0,
      "gaussian_indices": [1, 5, 9, "..."],
      "score": 0.87,
      "reason": "floating_blob + high_sh_entropy",
      "representative_view": "view_042.png",
      "bbox_3d": [[x0,y0,z0],[x1,y1,z1]],
      "source": "geometric_heuristic | gs_iou | open_vocab | manual"
    }
  ],
  "confirmed_remove": [],
  "confirmed_keep": [],
  "manual_edits": []
}
```

**`source` 字段很重要**：它记录候选来自哪条信号，便于后续统计各条路线的召回率，也便于人工复核时理解依据。

---

## 七、分层执行设计

| 层 | 职责 | Tier A 实现 | Tier C 实现 |
|---|---|---|---|
| **L0 Ingest** | 能力探测 + 场景加载 + 预处理 | 同 | 同 |
| **L1 Analyze** | 候选生成 + 排序 + NBS 判定 | 2D lifting + 语义 + 几何 | 几何 + 虚拟视角弱关联（若可渲染） |
| **L2 Remove** | 执行删除 + mask 产物 | 同 | 同 |
| **L3 Repair** | 空洞修复 | occlusion-aware 跨视角填充 → 生成式 inpaint 剩余 → 重优化 | 插值；渲染伪监督为实验路径 |
| **L4 Relight** | 阴影/光照残留清理 | 务实压平（或逆渲染） | 务实压平 |
| **L5 Export** | GS + 碰撞 mesh | 同 | 同 |

### L3：背景分区、监督与保护

删除后不仅检查空洞，还检查已有背景 Gaussian 的质量。保存分区结果、证据、置信度和所用视角，不能把“存在 Gaussian”当作“背景可靠”。

| 区域 | 判据与证据 | 处理 |
|---|---|---|
| 已有且可靠 | 多视角背景可见、重投影一致，外观与几何无明显异常 | 保留并冻结参数 |
| 已有但受损 | 背景仍有真实观测，但存在纹理混色、模糊、深度异常或欠约束 | 用有效背景像素局部重优化，必要时增删 Gaussian |
| 无有效观测 | 在可用源视角中没有可用背景监督 | 几何先验/生成补全，作为伪监督记录 |
| 状态未知 | 覆盖或几何证据不足、不同视角判断冲突 | 人工复核，禁止自动宣称“从未观测” |

“无有效观测”是相对于已提供数据的判断；虚拟视角渲染不能证明真实采集时见过背景。第 2 层的受损区域可能比删除投影大，`repair_mask_2d` 应包含必要的受损背景和过渡带，而非简单复用目标 mask。

**真实观测不等于可用背景监督**：只有通过目标排除、背景归属、遮挡与深度可靠性检查的原始像素，才参与真实背景损失。原图中被目标遮住的像素必须排除，否则会把已删除物体学回来。阴影/反射污染像素也应单独标记；跨视角渲染填充与生成补全分别记录来源，不能冒充原始干净观测。

**保护不能只靠 loss**：可靠且不需编辑的 Gaussian 默认冻结；只优化受损/新增集合及必要过渡带。记录可训练参数与 densification/剪枝范围，阻止受保护 ID 被自动修改。边界可进行受限联合优化，但需人工或协议明确允许，并报告区域外保持度。全图 SSIM 提供结构约束，不保证参数或渲染保持不变；即使冻结参数，新 Gaussian 的遮挡也可能影响旧区域。

**深度可靠性**：alpha 加权深度是渲染量，不是严格表面深度。用于回投、平面恢复或 TSDF 时，至少检查累积 opacity、深度离散度/多层混合、多视角一致性、边缘与薄结构；记录后端深度定义及尺度。不可靠深度拒绝融合，不能通过增加截断距离把错误当成正确表面。

### L3 的监督来源与质量边界

| 模式 | 可做的修复 |
|---|---|
| **有图像 + 位姿** | 完整：occlusion-aware 跨视角填充 → 生成式 inpaint → 重优化 → 阴影清理 |
| **有图像、无位姿** | 弱：能跑 COLMAP 补位姿就升级；否则只能做几何修补 |
| **仅 PLY** | 删除 + 几何外推；若可渲染且训练器兼容，可实验生成补全与伪监督微调，隐藏背景真实性无法直接验证 |

**关于局部重优化**：原图与标定可提供真实监督；只有 PLY 时仍可用虚拟视角、生成图像和保持损失进行伪监督优化。它是实验路径，不等于真实背景重建。

### 额外硬限制（必须在需求阶段说清）

> **单份静态 PLY 不能可靠恢复运动历史；可人工选择疑似动态残留，但不能把外观异常直接判为动态物体。**
>
> 若干扰物是拍摄期间在场移动的人/车，3DGS 会把它渲染成大量低 opacity 的浮空碎雾。**无运动先验时，连"它是动的"都判断不出来。**

**若采集流程可控，最优解是采集阶段清场重拍**，而非事后补救。

---

## 八、能力分级表

| Tier | 条件 | 识别 | 修复 | 评测 | 适用场景 |
|---|---|---|---|---|---|
| **A** | 图像 + COLMAP + 3DGS | 2D lifting + 语义 | 生成式 inpaint + 阴影清理 | holdout + paired GT | 旗舰交付 |
| **B** | PLY + `cameras.json` | 2D lifting（无原图需自渲染） | 部分 inpaint | 弱（无 GT） | 常见的"伪 PLY"输入 |
| **C** | 仅 PLY | 几何 + 虚拟视角弱关联 | 插值 / 实验性伪监督 | 保持度、残留、多视角一致性；无真实背景 GT | 不能证明隐藏背景真实还原 |
| **C+** | 仅 PLY + 人工 ROI | 用位置先验收窄候选 | 同 C | 同 C | 人工确认选择与质量 |

### 关于 C+ 的提议

即使没有图像，如果用户能提供"干扰物大概在哪、有几个"（一次鼠标框选，或一句"工作台右边那个东西"），**目标选择问题会收窄；隐藏背景补全仍需实验验证**。

> **建议把它设计成一等公民，而不是"自动失败后的降级选项"。**
>
> 人工 ROI 有望降低选择成本和漏删率；收益需用实际场景测量，不预设成功率数字。

---

## 九、工具链分工

见 [02-tools.md](02-tools.md) 第六节。摘要：

```
┌─ 主控（Python，自研）──────────────────────────────┐
│  CapabilityDescriptor → OperationPlan → 执行/报告   │
└──────┬─────────────────────────────────────────────┘
       │
       ├─ 独立算法后端 ──→ 官方 Inpaint360GS 基线 / 兼容训练器
       │
       ├─ 可选编辑与训练适配器 ──→ LichtFeld Studio
       │     · 训练期 mask 监督（改善第 2 层）
       │     · 验证内置 SAM2.1 是否能对渲染图点击分割并输出可消费 mask
       │     · Python 插件 / MCP 驱动编辑
       │     · CLI headless 批处理
       │
       ├─ 自研研究组件（实质工作）
       │     · L1 mask lifting + 低可见度补偿
       │     · L1 NBS 判定（式 8 深度 + 虚拟轨迹）
       │     · L3 修复
       │     · L4 阴影清理
       │
       └─ 人工复核与交付 ──→ SuperSplat
             · Box / Flood Fill 手工修正
             · mask 标注（反哺评测集）
             · 按选择集导出
```

算法主线不依赖 LichtFeld 编译；适配器通过文件或进程接口交换资产与映射。文件边界本身不构成许可证兼容性结论，具体集成方式另行核验。

---

## 十、关键可复用机制（来自 Inpaint360GS）

详见 [03-inpaint360gs.md](03-inpaint360gs.md) 第六节。

| 优先级 | 机制 | 对应层 | 复用成本 |
|---|---|---|---|
| **P0** | GS-IoU 关联 + K-Means 前景二分 | L1 | **低** |
| **P0** | 深度直接取自 Gaussian 场（免对齐） | L1 | **极低** |
| **P1** | 虚拟相机圆周轨迹 + PCA 位姿对齐 | L1 | 低 |
| **P1** | KNN 空间相似度损失 | L1 | 低 |
| **P2** | 3D 混合监督（全图 SSIM + masked L1/LPIPS） | L3 | 低 |
| **P2** | 递归条件 inpainting | L3 | 中 |

---

## 十一、待评审的设计决策

| # | 决策点 | 备选 | 影响 |
|---|---|---|---|
| 1 | OperationPlan 用 JSON 还是代码内 DSL | JSON / Python dataclass | JSON 便于跨语言和存档；DSL 便于表达复杂依赖 |
| 2 | 主控语言 | Python（生态）/ C++（与自研引擎一致） | 用户强项是 C++，但 ML 生态在 Python |
| 3 | mask 索引格式 | JSON / 压缩二进制 / 与 PLY 同构 | 大场景（百万级 Gaussian）下 JSON 可能体积过大 |
| 4 | L4 走务实压平还是逆渲染 | 务实 / GS-ID / ComGS | 取决于 [README.md](README.md) 待决问题第 4 条 |
| 5 | 后续训练器适配 | 官方基线 / gsplat / LichtFeld 适配器 | 首轮复用官方训练流程，不自研训练器 |
