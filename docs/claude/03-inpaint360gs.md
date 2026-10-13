# 03 · Inpaint360GS 深度解剖

> **Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes**
> Shaoxiang Wang, Shihong Zhang, Christen Millerdurai, Rüdiger Westermann, Didier Stricker, Alain Pagani
> DFKI / RPTU / TU Munich ｜ WACV 2026, pp. 117–127
>
> 整理日期：2026-10-12 ｜ 状态：调研完成
>
> **信息来源说明**：项目页（`dfki-av.github.io/inpaint360gs/`）只有摘要和视频，**所有公式、消融表、效率数据均来自 arXiv 全文**（arXiv:2511.06457，ar5iv HTML 版，622KB）。
>
> 项目页一处摘要区块标注为 "CVPR 2026"，而正式引用与 CVF 收录均为 **WACV 2026**，**以 WACV 为准**。

---

## 一、为什么这篇是本项目最直接相关的工作

它同时补上了前几轮暴露的两个最大缺口：

1. **用户指定目标后的跨视角归属**——它提供基于 2D mask 和 Gaussian 集合的关联机制，不负责自动发现应删除的对象
2. **真空洞的精确界定**（[00-requirements.md](00-requirements.md) 第 3 层）——它的深度定义技巧和虚拟相机轨迹是通用的 NBS 识别工具

**但它不以"清洁背景资产"为目标**，这是它和本项目的分歧点：作者自己承认**没有解决阴影残留**。

---

## 二、问题定位

作者把 360° 场景 inpainting 的困难拆成三条：

1. **在 360° 的 3D 场里定位目标物体** —— 需要基于灵活 prompt（VLM 或点击）的可编辑表示
2. **定义"从未见过"（NBS, never-before-seen）区域** —— 多物体遮挡下，删掉物体后到底哪块需要补，很难界定
3. **跨视角一致的高质量 inpainting** —— 结构连续 + 纹理连续

**关键判断：前两条不是 inpainting 问题，是"归属于哪个物体"和"补哪里"的问题。**

这正对应 [00-requirements.md](00-requirements.md) 第 1 层（归属判定）和第 3 层（真空洞识别）。作者自己也把大量篇幅花在 mask 关联和 NBS 识别上，而不是修补本身。

> **这是这篇论文最有价值的地方——它承认了难点不在生成，在界定。**

---

## 三、四个核心机制

### 机制 1：Key Object Management System —— 用 GS-IoU 做跨视角 ID 关联

**全文最有新意的部分，也是对本项目最有用的部分。**

#### 要解决的问题

GauGroup 用 DEVA 做视频级 mask 传播，把图像序列当视频处理。作者指出这在**稀疏视角下会失效**（原文："Specifically, it fails under sparse-view settings"），且视频级 2D 标签传播的误差会污染下游编辑。

#### 做法 (a)：K-Means 前景/背景二分

把 Gaussian 投影到 2D 后，**落在 mask 内的 Gaussian 不全是物体**——背景物体也会投影到同一个 2D 区域。

论文 Fig. 3 用卡车举例：**红点是卡车前景，蓝点是与卡车 mask 重叠的背景。**

**源码实现**：对局部投影区域按 Gaussian 深度做 K=2 聚类，再保留靠近相机的簇；这不是一般 3D 欧氏坐标聚类。论文叙述与代码细节需分开记录。

> 这一步很朴素，但**直击 2D 投影的根本歧义**。实现成本极低，是"高性价比技巧"。

#### 做法 (b)：GS-IoU 做跨视角关联

维护 **Key Object Database** $\mathcal{D}_{\text{ID}} = \{P_1, P_2, \dots, P_Q\}$，其中 $P_i$ 是第 $i$ 个物体的 Gaussian 索引集：

$$P_i = \{g_i^1, g_i^2, \dots, g_i^m\}$$

每来一个新视角，把它产生的临时物体与库中所有条目比对：

$$\text{GS-IoU}_{ij} = \frac{|P_i \cap P_j|}{|P_i \cup P_j|}, \qquad 0 \le \text{GS-IoU}_{ij} \le 1 \tag{式 2}$$

超过阈值 $\sigma$ → 继承已有 ID；否则 → 建新 ID。

#### 思想来源（值得注意）

作者明说这是**类比 SLAM 里的 keyframe overlap check**——测量两帧之间共享的可见内容。

> **把 SLAM 的共视关系搬到 Gaussian 集合上，这是全文最漂亮的一步。**

**三个天然优势**：
- 适合**非连续帧**（不像 DEVA 依赖时序）
- 天然支持**多物体**
- 阈值 $\sigma$ 是唯一的超参，调参面小

#### 作者的诚实说明

Key Object Database 里的 ID **"need not be perfect"**——它只是用来关联原始 2D mask，产出一致的物体 mask $O$，**真正的精修交给下一阶段的蒸馏**。

> **这个分工设计很聪明**：把粗关联和细优化解耦。粗阶段可以容错，因为错误会在蒸馏阶段被 2D 监督 + 3D 空间正则纠正。

### 机制 2：Object ID 蒸馏 —— α-blending + 空间相似度损失

沿用 GauGroup 的路线。每个 Gaussian 挂一个可学习的 identity feature $f$，用**同一套 α-blending** 渲染出 feature map：

$$\mathbf{F} = \sum_{i \in \mathcal{N}} f_i \alpha_i T_i \tag{式 3}$$

$$\hat{O} = \text{softmax}(\Phi(\mathbf{F})), \qquad \hat{O} \in \mathbb{R}^{H \times W \times Q}$$

其中 $\Phi(\cdot)$ 是线性投影到 $Q$ 维（$Q$ = 场景中不同物体的总数）。

$$\mathcal{L}_{obj} = \text{CrossEntropy}(\hat{O}, O)$$

#### 它自己加的那一项（关键）

对每个 Gaussian 的 feature，与它在**3D 空间中的 $k$ 近邻** $\mathcal{K}(f_i) = \{f_i^1, \dots, f_i^k\}$ 做余弦相似度约束：

$$\mathcal{L}_{space} = 1 - \sum_{i \in k} \frac{f_i \cdot f_k}{\|f_i\| \, \|f_k\|} \tag{式 4}$$

$$\mathcal{L}_{Dis} = \mathcal{L}_{obj} + \lambda \mathcal{L}_{space} \tag{式 5}$$

#### 为什么需要这一项？

纯 2D 监督在**物体边界和细结构**处没有足够约束——那些区域的 Gaussian 在多数视角下被遮挡或处在 mask 边界，**梯度信号弱**。

**3D 空间一致性正则相当于把标签在 3D 里扩散**，这正是 [00-requirements.md](00-requirements.md) 第 1 层"低可见度 Gaussian 补偿"的做法。

作者报告它 "significantly accelerates convergence and enables more efficient distillation, particularly around complex object boundaries and fine structures"。

#### 消融证据（重要信号）

去掉物体关联后（改用 DEVA 生成的 mask）：**PSNR 24.40 → 23.31，FID 33.93 → 66.75（几乎翻倍）**。

> **归属判定的质量是整个 pipeline 的性能瓶颈，不是修复阶段。**
>
> 这对本项目是个重要信号：**应该把研发重心放在识别层，而不是修补层。**

### 机制 3：虚拟相机视角 + 深度直接取自 Gaussian 场

#### (a) 虚拟视角

删掉物体后，用 **PCA 对原始相机位姿做对齐**，生成一条**以被删物体为中心的虚拟圆周轨迹** $\mathcal{P} = \{p_i\}_{i=1}^{L}$，每个位姿渲染出 $(C_i, D_i)$，再用 SAM-Tracking 提取 NBS mask $M_i$：

$$\mathcal{V} = \big\{(C_i, D_i, M_i) \;\big|\; (C_i, D_i) = \texttt{render}(p_i, \mathcal{R}), \; M_i = \texttt{SAMT}(C_i)\big\}_{i=1}^{L} \tag{式 6}$$

**为什么需要虚拟视角？**

原始训练视角是围绕场景转的，**对某个具体物体而言，视角分布可能很偏**——某些角度看不太到物体底部。虚拟圆周轨迹保证了对被删物体周围区域的**均匀覆盖**，从而可靠地界定 NBS 区域。

消融：去掉它 PSNR 24.18、FID 38.74。

**多物体遮挡的处理（很实用）**：用 **YOLOv8** 检测相互遮挡的实例，**把遮挡物临时删掉**去提 NBS mask，inpaint 完**再放回来**。

> 这个"临时移除 → 修复 → 恢复"的流程很干净，避免了预定义的 3D 学习掩码。

**轨迹半径是自适应控制的**，确保充分覆盖被遮挡区域而不引入极端视角。

#### (b) 深度来自 Gaussian 场本身（全文我认为最有工程价值的技巧）

$$\mathbf{D} = \sum_{i \in \mathcal{N}} z_i \alpha_i T_i \tag{式 8}$$

其中 $z_i$ 是该 Gaussian 在**相机坐标系下的 $z$ 坐标**。

**它替代了什么？**

| 方法 | 深度来源 |
|---|---|
| InFusion | 深度补全模型 |
| AuraFusion360 / GScream | **Marigold** 零样本深度估计 **+ 尺度对齐** |
| **Inpaint360GS** | **式 8，直接算** |

作者直接指出这些方法的问题是 **"depth alignment issues"**——单目深度模型输出的深度和 3DGS 的尺度不一致，需要额外对齐，而对齐不准就是伪影来源。

**而式 8 的深度和 RGB 用的是同一套 α-blending 权重**，所以：

> **深度和颜色天然同尺度、天然一致，对齐问题从根上消失了。**

**消融**：去掉深度引导后 PSNR 24.40 → 24.15，FID 33.93 → 35.75，影响相对温和。但作者强调它主要贡献在**效率**上（"significantly accelerates convergence"）。

**注意**：缺失深度区域的纹理复杂度通常低于颜色图像，因此可以同样用 LaMa 有效地 inpainting 深度。给定补好的 $D_{inp}$ 和 $C_{inp}$，配合 mask $M$ 融合得到 NBS 区域的点云，用于初始化 Gaussian。

**本项目可复用其流程，但需要官方代码依赖、权重与兼容训练环境；并非零额外模型依赖。**

### 机制 4：递归条件 inpainting + 3D 混合监督

#### (a) 递归条件 2D inpainting

不做逐帧独立 inpainting（那样跨视角纹理不一致），而是：

- 第一帧用 **LaMa** 独立补
- **从第二帧起，把上一帧的 inpainting 结果作为条件参考**

编码器把已补帧 $C_t$ 和目标帧 $C_{t+1}$ 编到共享隐空间，在特征空间联合优化 10 步后解码：

$$\ell_t, \ell_{t+1} = \text{Encoder}(C_t, C_{t+1})$$

$$\mathcal{L}_{\text{2DInp}} = \left\|(C_t - \hat{C}_t)\right\|_1 + \left\|M_{t+1} \odot (C_{t+1} - \hat{C}_{t+1})\right\|_1 \tag{式 7}$$

$$C_{t+1} = \text{Decoder}(\ell_t, \ell_{t+1})$$

**注意第一项不带 mask**——已补帧要**整体**保持一致，不只是被补区域。这强制了纹理在时间上的连续性。

**作者选 LaMa 而非扩散模型的理由写得很直白**："To avoid hallucination artifacts, we adopt Fourier convolutions LaMa"——这是降低幻觉风险的选型动机，不代表 LaMa 不会生成错误内容。

> **值得记住的工程取舍**：扩散模型画得漂亮，但在需要跨视角一致的任务上，**幻觉是负债**。这与 [01-landscape.md](01-landscape.md) 第 3.4 节的判断一致。

#### ⚠ 递归条件 inpainting 应视为可选实验分支，不是已默认配置

**源码验证结果**：`--recursive_guide` 是 `action='store_true'`（**默认关闭**），而 `run_inpaint.sh` 调用时**没有传这个参数**：

```bash
python bin/predict_color.py --data_name 360_${scene}_virtual    # 无 --recursive_guide
```

`LaMa/bin/predict_color.py:102` 有作者自己的注释：

```python
# Note: Recursive guide is experimental and may not suit all scenes. Suggest to disable it in default."
```

**因此**：论文把递归条件补全作为方法描述的一部分，但**官方代码把它作为实验性选项且默认关闭**。

**实施建议**：拆成两个实验，在自己的相机轨迹上比较跨视角一致性：

| 实验 | 配置 |
|---|---|
| **A. 独立逐视角 LaMa** | 默认，代码作者推荐 |
| **B. 递归条件 LaMa** | 显式传 `--recursive_guide` |

> **不能直接把递归版本当作已验证的默认配置。** 论文的效果增益可能来自其他组件（虚拟视角、深度引导、3D 混合监督），递归这一项的实际贡献需要自己消融确认。

#### (b) 3D 混合监督 —— 一个反直觉的发现

**直觉做法**：把 mask 外的 Gaussian 设为不可训练，只优化 mask 内。

**作者说这样不行**："empirical observations indicate that this strategy tends to produce noisy textures and unstable boundary transitions"

**方案：局部 + 全局混合监督**

$$\begin{aligned}
\mathcal{L}_{\text{3DInp}} = \;& (1-\lambda_1)\,\big\| M \odot (C_{inp} - \hat{C}) \big\|_1 \\
& + \lambda_1 \, \mathcal{L}_{\textrm{D-SSIM}}(C_{inp}, \hat{C}) \\
& + \lambda_2 \, \mathcal{L}_{\textrm{LPIPS}}(C_{inp}, \hat{C}, M)
\end{aligned} \tag{式 9}$$

**关键细节**：**SSIM 是在全图上算的，不乘 mask。**

作者解释："SSIM … is sensitive to localized inconsistencies when computed within small masks, applying it over the full image stabilizes optimization and improves boundary smoothness"

> **因为 mask 区域往往很小，在小区域里算 SSIM 会因局部不一致给出极不稳定的梯度；放到全图算，它反而变成全局结构锚点，防止局部优化跑偏。**

消融：去掉混合监督 PSNR 24.01（比完整版低 0.39），FID 38.42。

**注意：LPIPS 是带 mask 的（第三项传入 $M$），SSIM 不带。** 这个不对称设计是有意的。

---

## 四、实验数据

### Table 1：主结果（Inpaint360GS 数据集，11 场景，1/4 分辨率）

| 方法 | PSNR↑ | masked PSNR↑ | SSIM↑ | masked SSIM↑ | LPIPS↓ | masked LPIPS↓ | FID↓ |
|---|---|---|---|---|---|---|---|
| SPIn-NeRF | 19.71 | 34.53 | 0.5000 | 0.9854 | 0.5002 | 0.0140 | 229.95 |
| GScream | 20.95 | 28.47 | 0.7380 | 0.9819 | 0.2715 | 0.0161 | 206.25 |
| AuraFusion360 | 23.15 | 35.78 | 0.7923 | 0.9872 | 0.1915 | 0.0097 | 47.71 |
| GauGroup | 23.20 | 35.73 | 0.7928 | 0.9862 | 0.1770 | 0.0102 | 65.87 |
| **Inpaint360GS** | **24.40** | **36.29** | **0.8370** | **0.9886** | **0.1300** | **0.0078** | **35.93** |

**相对最强基线（GauGroup）**：PSNR +1.20 dB，masked PSNR +0.51 dB，SSIM +0.044，LPIPS **−26.6%**（0.1770 → 0.1300），FID **−45.5%**（65.87 → 35.93）。

**注意 SPIn-NeRF 的 masked SSIM 高达 0.9854 而全图 SSIM 只有 0.5000**。该差异说明指标对 mask 定义、裁剪和背景处理非常敏感；黑色遮罩或裁剪本身可能抬高 masked SSIM，必须按固定协议实现，不能仅据此断言全图指标被稀释。

### Table 2：效率（单张 NVIDIA H100）

| 场景 | 方法 | Object ID | 训练(min) | Inpaint(min) | 总计(min) | 存储(MB) |
|---|---|---|---|---|---|---|
| **bear** | SPIn-NeRF | ✗ | 79 | 196 | 275 | 336 |
| | GScream | ✗ | – | 52 | 52 | 73.2 |
| | AuraFusion360 | ✗ | 25 | 26 | 51 | 448.9 |
| | GauGroup | ✓ | 55 | 20 | 75 | 774.8 |
| | **Ours** | ✓ | **21.5** | **2.5** | **24** | 477.5 |
| **kitchen** | SPIn-NeRF | ✗ | 59 | 148 | 207 | 336 |
| | GScream | ✗ | – | 30 | 30 | 67.9 |
| | AuraFusion360 | ✗ | 20 | 43 | 63 | 183.5 |
| | GauGroup | ✓ | 27 | 13 | 40 | 897.4 |
| | **Ours** | ✓ | **12** | **3** | **15** | 663.5 |

**口径澄清（重要）**：

- **"快 5–10×"说的是 inpainting 阶段**（论文正文口径；摘要另有一处提到对 GauGroup 的 3D inpainting 阶段快约 5×、对 AuraFusion360 快约 20×）
- **训练仍需 12–21.5 分钟**，总时间 **24 / 15 分钟**才是规划吞吐量该用的数字
- 存储比 GauGroup 少约 30%（"model that is 30% more compact"；kitchen 897.4 → 663.5 MB 也对得上）

> **宣传口径容易让人误以为总时间很短。本项目估算算力成本时应以总计时间为准。**

### Table 3：消融

| 配置 | PSNR | SSIM | LPIPS | FID |
|---|---|---|---|---|
| a) 无物体关联（用 DEVA） | 23.31 | 0.7921 | 0.1932 | **66.75** |
| b) 无深度引导 | 24.15 | 0.8199 | 0.1256 | 35.75 |
| c) 无虚拟相机位姿 | 24.18 | 0.7987 | 0.1574 | 38.74 |
| d) 无条件 inpainting | 24.23 | 0.8156 | 0.1420 | 37.57 |
| e) 无 3D 混合监督 | 24.01 | 0.7997 | 0.1398 | 38.42 |
| **f) 完整** | **24.40** | **0.8370** | **0.1300** | **33.93** |

**读法**：

- **物体关联的贡献是压倒性的**（去掉它 FID 几乎翻倍到 66.75，是所有消融里最差的）
- 其余四项各自贡献 0.2 dB 量级，但**每一项都在起作用**——没有一个是凑数的

> **注意 Table 3 完整版的 FID 是 33.93，而 Table 1 中同一方法报的是 35.93。** 两表口径差异未在论文中明确说明，引用时应注意区分。

### 数据集

自建，**11 场景**（**7 单物体 + 4 多物体**，含遮挡），室内外都有。三个关键设计：

1. **相机参数（曝光、白平衡、ISO）固定** —— 消除光度变化，**直接针对 SPIn-NeRF 数据集的已知缺陷**
2. **含物体 + 无物体两个序列** —— 提供真值
3. **测试视角的 COLMAP 稀疏点云被剔除** —— 防止评测泄漏

评测在 object-free 测试图像上进行，全部实验在 **1/4 分辨率**下进行。

---

## 五、批判性评估

### 5.1 对本项目的四个直接价值

#### 价值 1：GS-IoU 关联机制 = 2D 候选 mask 到 3D 物体 ID 的跨视角关联器

> **重要更正（2026-10-12，源码验证后）**：本节早期版本把 GS-IoU 描述为"无先验的物体枚举器／候选检测器"。**这个表述是错的**，已更正。原因见下。

**GS-IoU 的实际职责是"关联"，不是"检测"。** 它把**不同视角已经得到的 2D mask** 关联成一致的物体 ID。它仍然依赖：

| 依赖 | 来源 |
|---|---|
| **2D 实例 mask 的产生** | HQSAM（`seg/raw_mask_sam.py --method hqsam`） |
| **人工指定要删哪个物体** | `run_remove.sh` 中的 `target_id` |
| **人工指定周边遮挡物** | `run_remove.sh` 中的 `target_surronding_id` |
| **人工修正补全区域** | 最后一步启动 Segment-and-Track-Anything 交互式 refine |

代码证据（`run_remove.sh`）：

```bash
target_id="26"                    # 人工硬编码
target_surronding_id="24,10"      # 人工硬编码
...
# 4. Interactive Mask Refinement
cd Segment-and-Track-Anything
python app.py                     # 交互式，需要人工操作
```

**准确的说法**：

> **GS-IoU 是「2D 候选 mask → 3D Gaussian 物体 ID」的跨视角关联机制，不是独立的候选检测器。**

它的价值在于**关联的鲁棒性**（比 DEVA 的视频级传播更适合稀疏视角、多物体），而非**无先验的发现能力**。

### 关于"开放干扰物集合"这一约束

**Inpaint360GS 并没有解决它。** 它的完整链路是：

```
HQSAM 2D mask → GS-IoU 关联 → 人工指定 target_id → 删除 → 修复
                                  ↑
                          人工先验在这一步
```

对本项目的含义（回到 [00-requirements.md](00-requirements.md) 第四节约束 1）：

- **GS-IoU 可以作为"候选 mask 传播方案"使用** —— 它让一个已经拿到的 2D mask 在 3D 里保持一致、可跨视角操作
- **但不能作为"候选生成方案"** —— 它不产生候选，候选仍来自 HQSAM（或你自研的检测器）
- 开放集合的候选生成仍需自己解决，路线见 [01-landscape.md](01-landscape.md) 第 2 节的识别线

### 实现细节（论文未描述的启发式）

源码 `seg/mask_associate.py` 的实际逻辑比论文 Fig. 3 描述的具体得多，是**有效的工程启发式，不是普适的 3D 物体分解算法**：

| 步骤 | 实现 |
|---|---|
| 1 | 把图像切成 **16×16 网格单元（不是 16×16 像素 patch）**（`--patch`，默认 16；注释：higher value, smaller object segmented, but slower） |
| 2 | 只在 **mask 覆盖非空的 patch** 内工作（`patch_coverage > 0`） |
| 3 | 每个 patch 内对**深度**做 `KMeans(n_clusters=2)`，取 `cluster_centers_.argmin()` 为前景簇 |
| 4 | **只取前景簇中最近的 30%**（`num_to_select = int(len(sorted_indices) * 0.3)`） |
| 5 | 若 patch 内 Gaussian 少于 2 个，退化为取**最近 50%** |

**第 1、4 步是论文没写的关键启发式**，也是"这不是通用算法"的直接证据——30% 这个截断比例是经验值，换场景未必成立。

### ⚠ 论文式 2 与代码实现不一致（复现陷阱）

**这是源码验证时发现的重要落差，论文和代码对 GS-IoU 的定义不同。**

论文式 2 写的是标准 IoU：

$$\text{GS-IoU}_{ij} = \frac{|P_i \cap P_j|}{|P_i \cup P_j|}$$

但代码 `seg/mask_associate.py:126-133` 实际执行的是另一种度量：

```python
associate_mode = "IOU_heighlight"
if associate_mode == "IOU":
    gs_iou = [num_intersection[i] / (num_union[i] + 1e-8) ...]              # 论文式(2)，走不到
elif associate_mode == "IOU_heighlight":
    gs_iou = [num_intersection[i] / (num_cur + num_intersection[i] + 1e-8) ...]  # 实际执行
...
if gs_iou[selected_mask] < 0.1:      # 论文写 σ = 0.2，代码是 0.1
```

| 项 | 论文 | 代码 |
|---|---|---|
| 分母 | $\|P_i \cup P_j\|$（标准 IoU） | $\|P_{\text{cur}}\|$ 与交集之和（**当前视角覆盖率的非线性变换**） |
| `"IOU"` 分支 | 论文式 2 | **死代码**，`associate_mode` 硬编码为 `IOU_heighlight` |
| 阈值 | σ = 0.2 | **0.1** |

**为什么重要**：标准 IoU 对集合大小差异敏感，覆盖率不敏感。当某物体在不同视角下可见比例差异大时（前视看到全部、侧视只看到一条边），两种度量的**排序会不同**，进而改变关联结果。

**对复现的影响**：按论文式 2 实现会与官方代码行为不一致，且论文中**没有 `IOU_heighlight` 模式的任何描述**，调试时会反复怀疑自己的实现。

> **建议先固定官方默认实现，再将标准 IoU 作为单变量对照。** 只有在同一删除集合、同一评测协议下比较后，才决定项目默认度量。

#### 价值 2：式 8 的深度定义 = 消除单目深度的尺度对齐依赖

> **更正（源码验证后）**：早期版本写"零额外依赖"。**不准确，已更正。**

**它消除的是什么**：单目深度模型（Marigold 等）输出深度与 3DGS 场景**尺度不一致**的问题。式 8 与 RGB 用同一套 α-blending 权重，因而**天然同尺度**，不需要尺度对齐。

**它没有消除的是什么**（`run_inpaint.sh` 的完整链路）：

```bash
python tools/prepare_lama_data.py ... --inpaint2lama
conda activate lama
python bin/predict_color.py --data_name 360_${scene}_virtual   # 补颜色
python bin/predict_depth.py --data_name 360_${scene}_virtual   # 补深度 ← 仍需 LaMa
python tools/prepare_lama_data.py ...
python edit_object_removal_plyfusion.py ...                     # 反投影融合
python edit_object_inpaint.py ...                               # 3DGS 微调
```

因此仍需要：

| 仍需组件 | 位置 |
|---|---|
| **LaMa 对缺失深度做补全** | `bin/predict_depth.py` |
| RGB 与深度的 mask | `prepare_lama_data.py` |
| 深度反投影 | `edit_object_removal_plyfusion.py` |
| 后续 3DGS 微调 | `edit_object_inpaint.py` |

**准确表述**：

> 式 8 消除的是**单目深度尺度对齐依赖**，不是**完全不需要深度补全模型**。

**对本项目的价值判断不变**：它省掉了 Marigold 这类大模型依赖和尺度对齐的调试成本，实现简单（一个加权求和），且**顺便给出了算 NBS 区域的工具**。这些仍然是真实的收益——只是别指望它让整个修复阶段变成零依赖。

#### 价值 3：虚拟相机圆周轨迹 = NBS 识别的通用工具

- 去掉了时间依赖（不依赖 DEVA 的时序假设）
- 不依赖可学习 mask 的迭代投影
- 若场景是固定相机绕物体一圈采集的，**这个机制几乎可以直接套**

#### 价值 4：3D 混合监督的 SSIM 全图技巧 = 立即可用的训练配方

哪怕自研 3D 重优化，`全图 SSIM + 带 mask 的 L1/LPIPS` 这个组合是现成的、有消融支撑的配方。

**注意那个反直觉点：mask 区域很小的时候，在小区域内算 SSIM 会不稳定。**

### 5.2 四个硬限制（对本项目而言）

#### 限制 1：作者自己承认的——阴影残留 + 复杂不规则纹理

结论段原话：**"our approach occasionally exhibits residual shadow artifacts cast by the removed objects and struggles with inpainting irregular complex textures, which remain to be explored in future work."**

**第一条正好命中 [00-requirements.md](00-requirements.md) 第 4 层（烘焙交互残留）。**

> **意味着这篇论文没有解决阴影问题，它只是把物体删干净了。**
>
> 如果拿它当 pipeline 的修复层，还需自己接一层阴影清理，或走逆渲染路线（GS-ID / ComGS）。**这是一个明确的、需要自己填的洞。**

#### 限制 2：需要图像 + COLMAP + 训练，不是"作用于现成 PLY 的工具"

整条 pipeline 从 RGB 图像开始重建 GRF。

- 仅 PLY 不满足官方原生流程的输入依赖；ExternalSelection 仍待适配验证。
- 有官方兼容图像、标定、训练资产与对象特征时，可尝试 NativeBaseline；存在相机文件本身不保证可运行。

> **它不是一个能"拿来处理客户给的 PLY"的组件，而是自建重建流程时才用得上的一环。**

#### 限制 3：训练时间远超 inpainting 时间

见 Table 2 分析。**规划吞吐量时用总时间（24 / 15 分钟），不要用 inpainting 时间。**

#### 限制 4：有独立复现困难

一项后续基准工作报告**无法复现**论文声称的 inpainting 时间，**实测 202 秒**；并且因为它自己的输入模态没有 per-Gaussian 实例身份，**它没有使用 Inpaint360GS 的物体移除机制，只评了 inpainting/refinement 部分**。

更有意思的是，该工作报告 **FID/PSNR 与人类偏好相关性很差**，在人工对战中人类有 **37.5%** 的情况下更偏好 Inpaint360GS 的结果。

> **不确定**：该基准工作复现时改变了渲染分辨率（512 而非论文规定的 1/4 分辨率），所以 202 秒与论文口径不完全可比；且它只评了部分模块。
>
> **但"指标与人类偏好脱节"这个结论值得警惕**——自研方案完成后，**目视评估不能省**，尤其中间还要过人工确认环节。

---

## 六、可复用机制清单（按优先级）

| 优先级 | 机制 | 依赖 | 对应需求层 | 复用成本 |
|---|---|---|---|---|
| **P0** | GS-IoU 关联 + K-Means 前景二分 | 仅需投影和集合运算 | 第 1 层 | **低** |
| **P0** | 式 8 深度定义 | 仅需渲染深度通道 | 第 3 层 | **极低** |
| **P1** | 虚拟相机圆周轨迹 + PCA 位姿对齐 | 需渲染能力 | 第 3 层 | 低 |
| **P1** | $\mathcal{L}_{space}$ KNN 空间正则 | 需训练 | 第 1 层 | 低 |
| **P2** | 3D 混合监督（全图 SSIM + masked L1/LPIPS） | 需训练 | 修复层 | 低 |
| **P2** | 递归条件 inpainting | 需 LaMa + 位姿 | 修复层 | 中 |
| **P3** | 遮挡物临时移除/恢复流程 | 需 YOLOv8 + 完整 pipeline | 第 3 层 | 中 |

**首轮优先复现完整官方默认流程**，以人工复核的删除集合对比“仅删除”和“官方补全”。以上机制是后续模块实验清单，不构成已经验证的低成本承诺：GS-IoU 依赖已有 2D mask，不负责候选生成；式 8 深度也不单独解决 NBS 判定。

---

## 七、对项目的三点具体建议

1. **先冻结官方 commit 与默认参数，验证选择、删除和补全的完整闭环。** 选择与修复分别评测；关联度量和递归补全随后做单变量消融。不要在首轮同时改写关联器、训练器和修复器。

2. **用它的数据集做起点，但别只依赖它。** 自建数据集解决了 SPIn-NeRF 的三个已知问题，是目前最合适的 360° benchmark。但注意它的场景是"单物体/多物体遮挡"，**与机器人现场的干扰物分布可能差别很大**（现场更多是低矮、贴地、成堆的物体）。**自建 paired GT 仍然要做——自己的数据才是最终裁判。**

3. **阴影问题要单独立项。** 作者自己把它列为 future work。结合 [01-landscape.md](01-landscape.md) 第四节：**务实的局部颜色压平 vs 完整逆渲染，前者性价比高得多，建议先做前者。**

---

## 参考来源

- [Inpaint360GS 项目页](https://dfki-av.github.io/inpaint360gs/)
- [arXiv:2511.06457（**全文，本文所有公式与数据的来源**）](https://arxiv.org/abs/2511.06457)
- [ar5iv HTML 全文](https://ar5iv.labs.arxiv.org/html/2511.06457)
- [WACV 2026 开放获取版（CVF）](https://www.openaccess.thecvf.com/content/WACV2026/html/Wang_Inpaint360GS_Efficient_Object-Aware_3D_Inpainting_via_Gaussian_Splatting_for_360deg_WACV_2026_paper.html)
- [WACV 2026 补充材料](https://openaccess.thecvf.com/content/WACV2026/supplemental/Wang_Inpaint360GS_Efficient_Object-Aware_WACV_2026_supplemental.pdf)
- [IEEE Xplore 收录](https://ieeexplore.ieee.org/document/11492045)
- [DFKI 出版物页](https://www.dfki.de/en/web/research/projects-and-publications/publication/16581)
- [Semantic Scholar 页面（含引用列表）](https://www.semanticscholar.org/paper/Inpaint360GS%3A-Efficient-Object-Aware-3D-Inpainting-Wang-Zhang/bcc79d38e0a7b4c35ed0d66d7975f0b15400a0ac)

**抓取渠道说明**：`dfki-av.github.io`、`arxiv.org`、`github.com` 的网页抓取在本次会话被网络策略拦截，改用命令行 curl 取得页面（下载内容按未信任数据处理，隔离在 `/tmp/ext_inpaint360/`）。
