# 06 · 源码验证记录

> 对象：Inpaint360GS 官方代码（`github.com/dfki-av/Inpaint360GS`，commit `d54c893`）
>
> 验证日期：2026-10-12 ｜ 方式：克隆仓库 + 逐条对照源码
>
> 目的：把文档中关于**代码行为**的断言从"推测"转为"已核实"，并记录论文与代码不一致之处。

---

## 验证结论总表

| # | 断言 | 来源 | 结论 |
|---|---|---|---|
| 1 | GS-IoU 是关联器，不是检测器 | 外部评审 | **✅ 成立** |
| 2 | 需人工指定 `target_id` / `target_surronding_id` | 外部评审 | **✅ 成立** |
| 3 | `--recursive_guide` 默认关闭且为实验性 | 外部评审 | **✅ 成立** |
| 4 | SuperSplat 导出会打乱 Gaussian 索引 | 外部评审 | **✅ 成立** |
| 5 | PLY 不含 per-Gaussian 训练统计 | 外部评审 | **✅ 成立** |
| 6 | 式 8 仍需 LaMa 补深度 | 外部评审 | **✅ 成立** |
| 7 | **论文式 2 与代码实现定义不同** | **本次发现** | **⚠ 新发现** |
| 8 | KMeans 逻辑含 16×16 网格单元（不是 16×16 像素 patch） + 30% 截断 | **本次发现** | **⚠ 新发现** |

---

## 1. GS-IoU 的职责：关联器，不是检测器 ✅

**代码证据**（`run_seg.sh`）：2D mask 由 HQSAM 产生，GS-IoU 在其后。

```bash
# 2. Generate 2D Segmentation Masks
python seg/raw_mask_sam.py --dataset_path data/${dataset_name}/ \
    --scene_name ${scene} --image_folder images_${resolution} --method hqsam

# 3. 3D Mask Association (Linking 2D masks to 3D space)
python seg/mask_associate.py --source_path ... --mask_generator hqsam ...
```

**代码证据**（`run_remove.sh`）：目标 ID 人工硬编码，且最后一步是交互式工具。

```bash
target_id="26"
target_surronding_id="24,10"
...
# 4. Interactive Mask Refinement
cd Segment-and-Track-Anything
python app.py
```

**结论**：链路为 `HQSAM 2D mask → GS-IoU 关联 → 人工指定 target_id → 删除`。**GS-IoU 不产生候选，候选来自 HQSAM。**

**对文档的修正**：[03-inpaint360gs.md](03-inpaint360gs.md) 价值 1 已重写，从"开放集合下的无先验识别方案"改为"2D 候选 mask → 3D 物体 ID 的跨视角关联机制"。

---

## 2. ⚠ 论文式 2 与代码实现不一致（本次新发现）

**论文式 2**：

$$\text{GS-IoU}_{ij} = \frac{|P_i \cap P_j|}{|P_i \cup P_j|}$$

**代码实现**（`seg/mask_associate.py:125-134`）：

```python
associate_mode = "IOU_heighlight"          # ← 硬编码
if associate_mode == "IOU":
    gs_iou = [num_intersection[i] / (num_union[i] + 1e-8) ...]        # 论文式(2)，走不到
elif associate_mode == "IOU_heighlight":
    gs_iou = [num_intersection[i] / (num_cur + num_intersection[i] + 1e-8) ...]  # 实际执行
...
selected_mask = torch.argmax(gs_iou)
if gs_iou[selected_mask] < 0.1:            # ← 论文写 σ = 0.2
    selected_mask = self.num_classes
```

| 项 | 论文 | 代码 |
|---|---|---|
| 分母 | $\|P_i \cup P_j\|$ | $\|P_{\text{cur}}\| + \|P_i \cap P_j\|$ |
| 语义 | 标准 IoU（对称） | **当前视角覆盖率的非线性变换**（不对称） |
| `"IOU"` 分支 | 论文式 2 | **死代码**，`associate_mode` 硬编码为 `IOU_heighlight` |
| 阈值 σ | 0.2 | **0.1** |

令 $r=I/N_{cur}$，则实际分数 $s=r/(1+r)$，最大值为 0.5。代码阈值 0.1 对应覆盖率约 11.1%；与标准 IoU 比较时需分别校准阈值。

**影响**：标准 IoU 对集合大小差异敏感，覆盖率不敏感。当物体在不同视角可见比例差异大时，两种度量的**排序会不同**，进而改变关联结果。

**论文中没有任何关于 `IOU_heighlight` 模式的描述**——复现者按式 2 实现会得到与官方代码不同的行为，且难以定位原因。

**行动项**：已列入 [05-roadmap.md](05-roadmap.md) 阶段 3（官方默认基线建立后对比两种度量）。

---

## 3. ⚠ KMeans 逻辑的实际细节（本次新发现）

**论文 Fig. 3 只提到"K-Means 区分前景背景"。代码实际逻辑**（`seg/mask_associate.py:288-333`）：

| 步骤 | 实现 | 论文是否提及 |
|---|---|---|
| 1 | 图像切 **16×16 网格单元（不是 16×16 像素 patch）**（`--patch`，默认 16） | ✗ |
| 2 | 只在 mask 覆盖非空的 patch 内工作（`patch_coverage > 0`） | ✗ |
| 3 | 每 patch 对深度 `KMeans(n_clusters=2, random_state=0)` | ✓ |
| 4 | 取 `cluster_centers_.argmin()` 为前景簇 | ✓ |
| 5 | **只取前景簇中最近的 30%** | ✗ |
| 6 | patch 内 Gaussian < 2 时退化为最近 50% | ✗ |

```python
# 步 5 的代码
num_to_select = int(len(sorted_indices) * 0.3)
selected_indices = sorted_indices[:num_to_select]
```

**30% 是经验值**，换场景未必成立——这是"有效工程启发式而非普适算法"的直接证据。

**命令行参数说明**（`--patch` 的 help 原文）：

```
higher value, smaller object segmentated, but slower
```

---

## 4. `--recursive_guide` 默认关闭且标注为实验性 ✅

**参数定义**（`LaMa/bin/predict_color.py:236`）：

```python
parser.add_argument("--recursive_guide", action='store_true', help="Enable recursive guidance")
```

`action='store_true'` → **默认 False**。

**调用处**（`run_inpaint.sh`）**未传该参数**：

```bash
python bin/predict_color.py --data_name 360_${scene}_virtual
```

**作者自己的注释**（`LaMa/bin/predict_color.py:102`）：

```python
# Note: Recursive guide is experimental and may not suit all scenes. Suggest to disable it in default."
```

**结论**：论文把递归条件补全写进方法描述，但**官方代码将其作为实验性选项且默认关闭**。

**对文档的修正**：[03-inpaint360gs.md](03-inpaint360gs.md) 机制 4 已标注为"可选实验分支"，[05-roadmap.md](05-roadmap.md) 阶段 2B 列入开/关对照实验。

---

## 5. 式 8 仍需 LaMa 补深度 ✅

**代码证据**（`run_inpaint.sh` 完整链路）：

```bash
python tools/prepare_lama_data.py -s ... --inpaint2lama
conda activate lama
python bin/predict_color.py --data_name 360_${scene}_virtual
python bin/predict_depth.py --data_name 360_${scene}_virtual     # ← 深度仍需补全
python tools/prepare_lama_data.py -s ...
python edit_object_removal_plyfusion.py -s ...                   # 反投影融合
python edit_object_inpaint.py -s ...                             # 3DGS 微调
```

**结论**：式 8 消除的是**单目深度尺度对齐依赖**，不是**完全不需要深度补全模型**。

**对文档的修正**：[03-inpaint360gs.md](03-inpaint360gs.md) 价值 2 已重写。

---

## 6. SuperSplat 版本与身份映射（版本已核验，行为待实测）

2026-10-13 通过官方仓库核验：main commit `f2a1b4975b4766e0ee0c95e63c2737dc2ecb6e5b`，package.json 版本 **3.5.2**。来源：[固定快照](https://github.com/playcanvas/supersplat/blob/f2a1b4975b4766e0ee0c95e63c2737dc2ecb6e5b/package.json)。

旧文档关于 1.0.0 起 Morton 重排、`reorder`/`skipReorder` 开关、压缩格式始终重排和性能增益的描述，本轮不作为 3.x 已验证事实。按固定 commit 分别验证加载重排、选择存储/读回和导出身份追踪，详见 [02-tools.md](02-tools.md) 第 2.6 节。

一般身份约束仍成立：删除/过滤会改变位置索引，关闭重排不能替代源资产哈希、版本和编辑映射。跨轮训练 split/clone 需父 ID，补全新增需新 ID。尚未执行编辑器往返测试，不宣称现成稳定 ID 导出能力。

---

## 7. PLY 不含 per-Gaussian 训练统计 ✅

**标准 PLY 布局**：

```
x y z / f_dc_0..2 / f_rest_0..M / opacity / scale_0..2 / rot_0..3
```

**不含**：训练视角计数、梯度累积、visibility 统计。

**推论**：要得到 $v_i$ 必须**自行重新渲染所有视角并累计 $\alpha_i T_i$**。

**更重要的推论**：**低可见度 ≠ 干扰物**。墙角、细杆、被遮挡的背景天然低可见。**它只能作为不确定性分数，不能作为删除证据。**

**对文档的修正**：[04-architecture.md](04-architecture.md) 第五节信号 1 已重写，并新增"低可见度 ≠ 干扰物"的专门更正。

---

## 附：仓库信息

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/dfki-av/Inpaint360GS` |
| Commit | `d54c893285c6cb27788e05cce607e7d3cca6388a` |
| 分支 | `main` |
| 许可证 | **Apache License 2.0** |
| 数据集发布 | 2026-01-19（据 README 新闻条目） |
| 主要入口 | `run_seg.sh` → `run_remove.sh` → `run_inpaint.sh` |

**许可证说明**：仓库自身为 **Apache 2.0**（非 GPL）。但注意其代码"Modified from" Gaussian-Grouping 与 Gaga，且 `LaMa/`、`Segment-and-Track-Anything/` 为内嵌的第三方项目，各有自己的许可证——**商用前需逐一确认**。

---

## 验证方法说明

- 仓库克隆至 `/tmp/src_check/repo`（按未信任数据处理，隔离于工作目录之外）
- 所有代码引用均给出文件路径与行号，可自行复核
- 第 6 条（SuperSplat）为外部信息，**未在本机实测**，已明确标注
