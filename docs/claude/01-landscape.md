# 01 · 方法调研

> 调研范围：3DGS 场景中的干扰物识别与移除、空洞修复、光照/阴影一致性三条线。
>
> 整理日期：2026-10-12 ｜ 状态：调研完成
>
> **信息强度标记约定**：本文所有方法条目若无特别说明，均来自论文原文或官方页面（**已核实**）。标注 **不确定** 的条目表示无法确认发布时间或来源互相矛盾，不应作为决策依据。

---

## 一、问题分解与三条技术线的对应关系

| 需求层（见 [00-requirements.md](00-requirements.md)） | 对应技术线 | 成熟度 |
|---|---|---|
| 第 1 层 归属判定 | **识别线**：2D→3D mask lifting | 较成熟，有系统性弱点 |
| 第 2 层 被压制背景 | 识别线 + 修复线交界 | 依赖训练期干预 |
| 第 3 层 真空洞 | **修复线**：inpainting | 较成熟 |
| 第 4 层 阴影残留 | **光照线**：逆渲染 / 务实压平 | 不成熟，本项目的主要缺口 |
| 第 5 层 表示与导出 | 工具链（见 [02-tools.md](02-tools.md)） | 成熟 |

**总体判断**：识别线和修复线有可复用工作，光照线仍不成熟。首版瓶颈是用户指定目标后的三维归属、背景保护和修复分区；自动发现开放集合目标属于后续问题。

---

## 二、识别线：2D→3D Mask Lifting

### 2.1 共同机制

所有 lifting 方法共享同一个算子：渲染时每个像素是沿光线 alpha 加权的混合，因此每个 Gaussian 对某像素的贡献约为 $\alpha_i T_i$。**这个软权重就是从 2D mask 反投到单个 Gaussian 的映射。**

各方法的差异只在**如何解决由此产生的歧义**——一个 Gaussian 在多个视角下被看到、遮挡关系不同、边界暴露程度不同。

### 2.2 Gaussian Grouping（ECCV 2024）

**机制**：给每个 Gaussian 挂一个可学习的 identity encoding，用 SAM 派生的 2D mask loss 训练，**外加一个 3D KNN 空间一致性正则**。

**定位**：训练期一次完成，支持开放词汇编辑（移除+修补、重着色、风格迁移、新视角分割）。

**权衡**：与外观联合优化，**质量最好但最慢**。

**对本项目**：它的 ${\mathcal{L}}_{space}$ 思想是最有价值的可复用点，Inpaint360GS 正是沿用了这一项。若不追求最强编辑质量，可只借用这个损失项而不整套继承。

### 2.3 FlashSplat（ECCV 2024）

**机制**：将固定 3DGS 下的标签分配写成整数线性优化，利用 alpha 混合贡献的结构推导闭式最优标签分配；它不是一般意义上的 min-cut / graph-cut 求解器。

**定位**：比迭代法快约 **50×**（数十秒每场景量级），精度相当或更好。**"分割提升变成一次求解"**是它的卖点。

**对本项目**：值得作为 2D→3D 归属基线，重点验证其输入 mask、索引输出、边界误删和运行成本；不能据此承诺交互式产品性能。

### 2.4 SAGA / Segment Any 3D Gaussians（AAAI 2025）

**机制（AAAI 2025 / 官方 v2）**：学习 Scale-Gated Affinity Features，以 scale-aware contrastive training 建立不同尺度下的对象亲和关系；用 2D SAM mask 及其尺度监督 3D 特征，交互时通过 prompt、尺度门控和相似度阈值得到 Gaussian 集合。

**版本边界**：官方 main README 仍描述早期的 SAM 特征引导与对应关系损失，且明确要求参考 v2。实验应固定分支与 commit，不能把早期 main 描述替代正式版机制。

**定位**：需要预训练 3D 特征；毫秒级是特征训练完成后的交互推理速度，不包括数据准备、SAM 特征/mask 提取和 3D 特征训练。

**对本项目**：适合作为已有完整数据和特征场景下的交互式 3D 分割候选；不是仅凭 PLY 即可直接使用的零准备分割器。

### 2.5 三个方法共同的失败模式

必须预期到并主动规避：

| 失败模式 | 成因 |
|---|---|
| **细薄结构破碎**（线缆、栏杆） | 边界处在多视角下可见度低 |
| **透明/高反光物体粘连** | alpha 权重语义模糊，SAM 本身也难以处理 |
| **边界大面积遮挡时误判** | 投票机制对被遮挡 Gaussian 无票可投 |

**第三条是根本性的**，与 [00-requirements.md](00-requirements.md) 第 1 层的"低可见度 Gaussian"问题同源。**仅靠 lifting 无法解决，必须引入 3D 空间先验**（KNN 标签扩散或几何包络判定）。

**重要设计约束**：不要对低可见度 Gaussian 做颜色式的 inpainting 推断，容易产生浮空幽灵 Gaussian。应走标签扩散路线。

### 2.6 开放词汇路线

若干扰物种类无法预先枚举，需要把 CLIP embedding 挂到 Gaussian 上再文本查询（如 OVR-GS）。

**代价**：多一层 embedding 训练 + 一次文本检索。

**已知弱点**：**小物体召回不稳定**。

**本项目定位建议**：把它当**排序信号**，而非**决策信号**。这与 [00-requirements.md](00-requirements.md) 第四节"候选生成 + 排序 + 确认"的设计一致。

> **补充说明（源码验证后）**：Inpaint360GS 的 GS-IoU **不是**开放集合识别方案——它依赖 HQSAM 产生 2D mask，且需要人工指定 `target_id`（见 [03-inpaint360gs.md](03-inpaint360gs.md) 价值 1 的更正）。首版由用户指定对象，因此不需要先解决自动发现干扰物。 上节的识别线（FlashSplat / SAGA / Gaussian Grouping）主要解决“给定提示或 2D mask 后如何得到 3D 归属”，不自动决定哪些对象应删除。自动判断对象是否该删除属于后续研究，不阻塞用户提示驱动的编辑。

### 2.7 几何启发式（兜底路线）

若干扰物有明确空间分布特征（集中在某区域、或"浮在地面上方"），可用 RANSAC 平面拟合 + 连通域 + 高度阈值做无监督删除。

**优点**：零训练成本，粗糙但对"一大坨东西"很有效。

**建议用法**：**作为 debug 对照基线先跑一遍。** 它给出的结果能帮你判断 2D lifting 是真的在起作用，还是只是碰巧。

---

## 三、修复线：Inpainting

### 3.1 五种策略对比

| 策略 | 机制 | 优点 | 缺点 |
|---|---|---|---|
| **1. Occlusion-aware 逐视角 2D inpaint + 回投** | 先用渲染深度把其他视角的背景像素投影过来填充，**只剩真正看不见的像素**才交给生成模型 | 最稳；真观测不引入幻觉 | 需位姿；实现环节多 |
| **2. 3D 原生优化** | 删除后直接在空洞区域新增 Gaussian，用 inpaint 后的视角监督 | 无跨视角一致性问题（本就在 3D 里优化） | **空洞大且真未观测时，3D 里无任何约束来源，退化成噪声** |
| **3. 生成式 3D Inpainting / Diffusion** | 扩散模型做 3D 补全 | 大空洞下视觉丰富 | **跨视角一致性和幻觉是本质缺陷**——同一区域在不同视角编出不同内容，3DGS 重优化被迫在矛盾信号里折中，结果是糊的 |
| **4. 删 + 重优化（保守基线）** | 用 mask 限制采样视角，加长训练，在被暴露区域加密 densification | 实现成本低一个量级 | 接受质量损失 |
| **5. 阴影/光照残留处理** | 见第五节 | —— | 与上述正交，但决定成败 |

### 3.2 策略 1 的代表工作

#### 3DGIC · 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency（CVPR 2025）

**核心思路**：用渲染深度把其他视角的背景像素投影到当前视角，**精修 SAM 掩码**，使只有真正被遮挡的像素才需要 inpaint。

**报告数据**：
- mask 精修使 **inpaint 区域缩小 30–60%**，耗时 < 1s
- SPIn-NeRF 上 FID 36.4、m-FID 96.3、LPIPS 0.26、m-LPIPS 0.028（配 LDM inpainter）
- 单场景 5000 步在 **RTX 3090 上约 15 分钟**

**对本项目**：**算力门槛完全可行**（用户有 4090）。"先跨视角填充真观测，再生成剩余"这个流程顺序是关键，直接对应 [00-requirements.md](00-requirements.md) 第 3 层"避免破坏完好背景"的要求。

#### AuraFusion360（CVPR 2025）

**定位**：reference-based 的 360° 无界场景 inpainting on GS。

**三个组件**：depth-aware unseen-mask 生成；Adaptive Guided Depth Diffusion 做零样本点放置；SDEdit 增细节。

**附带贡献**：**360-USID** —— 首个 360° 无界场景 inpainting 的带真值数据集。

**对本项目**：360° 无界场景与你"真实场景重建"的形态接近，数据集可直接用于评测。

#### Inpaint360GS（WACV 2026）

**这是与本项目最接近的工作**，单独立文档解剖：[03-inpaint360gs.md](03-inpaint360gs.md)。

**先给结论**：它的**识别机制**（GS-IoU 关联 + K-Means 前景二分）和**深度定义技巧**（从 Gaussian 场直接算深度，免对齐）是可直接复用的最高价值部分；它的**修复质量**是当前 SOTA 但**不解决阴影残留**。

### 3.3 策略 4 的证据：不要盲目上扩散模型

一项针对**多物体 3DGS 场景单步 inpainting 方法**的基准测试得到两个反直觉结论（**已核实为论文报告，但该工作本身信息标注不清，见下方说明**）：

1. **基于重建的 inpainter 在 3D 一致性上常常胜过生成式扩散模型**
2. **从头重新初始化场景比微调现有场景质量更高**

**含义**：若场景多视角覆盖好，策略 4 可能打平甚至更好，且实现成本低一个量级。**不要默认上扩散模型。**

> **不确定**：上述基准工作的发布渠道与时间我未能确认，来源之间存在矛盾（有标注为 2026 年的会议链接）。**其结论方向性地合理，但不应作为定量依据。**

### 3.4 修复线的核心取舍：幻觉 vs 一致性

Inpaint360GS 作者在论文中明确写出选型理由："To avoid hallucination artifacts, we adopt Fourier convolutions LaMa"——作者希望降低幻觉风险，采用 LaMa 而非扩散模型；LaMa 仍会推断未观测内容，不能保证无幻觉或真实还原。

**这是一个值得记住的工程取舍**：扩散模型画得漂亮，但在需要跨视角一致的任务上，**幻觉是负债**。

---

## 四、光照线：烘焙残留与场景合成

这是本项目的**主要技术缺口**，因为 Inpaint360GS 作者自己承认未解决（"residual shadow artifacts cast by the removed objects"）。

### 4.1 问题定义

**ComGS 论文**直接点名了这个问题的本质：

> GS 里**烘焙的外观与阴影信息**会在合成物体时导致不一致。

### 4.2 完整逆渲染路线

#### GS-ID（ICCV 2025）

**机制**：一个 deshadowing 模块学习 **per-splat 可见性向量**，从而把多个未知光源产生的阴影解耦；配合可学习环境贴图 + 各向异性球面高斯混合表示局部光源；用扩散模型提供的 **normal / albedo / roughness 先验**打破光照-材质歧义。

**对本项目**：这是"阴影清理"最完整的解法，但引入了大量假设和优化复杂度。

#### ComGS（ICLR 2026, arXiv:2510.07729）

**机制**：**Surface Octahedral Probes (SOPs)** —— 用八面体贴图存储光照与遮挡，**插值查询而非光线追踪**，因而比光线追踪方案快 ≥ 2×。光照只在**物体放置位置**估计（一个 360° radiance field + 微调扩散模型做补全），而非全场景光传输。

**性能**：约 **28 FPS**，单次编辑约 **36 秒**。

**对本项目**：它的使用场景是"往场景里放东西"，与本项目的下游（机器人仿真）逻辑一致。但**它的目标不是删除阴影，而是插入时重算阴影**——需要判断是否满足你的需求。

#### 其他

- **3D Gaussian Inverse Rendering with Approximated Global Illumination**（arXiv:2504.01358）：用**屏幕空间光线追踪**在光栅化的 G-buffer 上做一发间接光，避免探针结构在编辑后失效。定位为对 GS-IR（光探针缓存）和 Relightable3DGS（BVH 光线追踪）的替代。
- **LightBridge**（见下方不确定说明）：训练一个前馈模型，一次前向预测整个 relit 3DGS——视频扩散模型提供 2D relighting 线索，transformer 传播到 SH 系数。训练于合成 OLAT 数据集（300+ Infinigen 房间，9000+ 场景-光照对）。

> **不确定**：LightBridge 的 arXiv 编号（2609.02543）指向的时间点无法核实，可能超出本次调研可确认的范围。**内容方向可参考，但不应作为已确认事实引用。**

### 4.3 务实替代（推荐先做）

**关键判断**：如果新物体在仿真器里本来就会有一套自己的光照，**你不需要把背景做到物理正确，只需要把背景的烘焙阴影抹掉**。

**做法**：
1. 提取地面平面
2. 检测地面上的暗斑（与周围地板的亮度/色度差异 + 与干扰物足迹的几何重合）
3. 做局部颜色归一化压平

**权衡**：粗糙，但对"地面还留着旧影子"这个具体穿帮点性价比高得多，且不需要逆渲染那套脆弱假设。

**建议**：**先做务实版。** 逆渲染是研究级投入，其收益只有在仿真器真的要复用场景光照时才体现——这一点取决于 [README.md](README.md) 待决问题第 4 条的答案。

### 4.4 调研缺口

**未找到以"接触阴影清除"命名的任务定义。** 现有工作处理的是**投射阴影分解**（GS-ID）或**插入后阴影重合成**（ComGS），而非**接触 AO 清理**。这可能意味着：
- 该子问题尚未被单独形式化，需要自己定义
- 或它被隐含在更一般的"场景重光照"问题里

**这是一个需要自己填的洞，不存在拿来即用的方案。**

---

## 五、评测体系

### 5.1 优先级最高的动作：自建 paired GT（先做数据集，再写算法）

对同一场景，**带干扰物拍一遍、搬走后拍一遍**，同一相机轨迹。这样就有了 **real GT**，而非合成出来的假空洞。

**为什么必须自建**：SPIn-NeRF 数据集之所以成为领域标准基准，恰恰因为它是真实场景"有/无目标物"两次捕获的。**只要有 4090 和可控场景，两三天就能积累一批。**

**注意**：Inpaint360GS 的作者也采用了这个做法，并额外修正了 SPIn-NeRF 的三个已知缺陷——正面视角限制、光度不一致（他们固定曝光/白平衡/ISO）、测试视角点云泄漏（他们剔除测试视角的 COLMAP 点）。

### 5.2 指标

| 指标 | 说明 |
|---|---|
| **masked PSNR / SSIM / LPIPS** | **在 mask 区域内计算**，不要算全图 |
| **FID / masked FID** | 分布层面的质量 |
| **3D 一致性**（如 TSED） | 跨视角深度/特征一致性 |

**为什么必须看 masked 指标**：Inpaint360GS 的 Table 1 里，SPIn-NeRF 的 masked SSIM 高达 0.9854 而全图 SSIM 只有 0.5000。这不能直接证明全图指标被稀释：mask 极性、置黑区域、裁剪与 padding 都可能影响数值。应分别报告修复区域质量和未编辑区域保持度，并冻结指标实现。

### 5.3 评测协议的坑（重要）

**这个领域的评测协议极不统一，不同论文的 LPIPS/FID 数字不可直接横向比较。**

已核实的具体证据：

- 有文献明确指出 **OR-NeRF 报告的数字和 SPIn-NeRF 原始数字对不上**，因为指标计算方式不同（前者按 inpainting mask 的包围盒裁剪、四边各 pad 10%）
- 有论文报告在**统一协议下**重排后，某些方法的 LPIPS/FID 从 `0.4210 / 92.62` 变成 `0.6692 / 244.19`

**应对**：
1. 要么严格复现原协议
2. 要么只在自己的 paired 数据集上做内部对比
3. **绝不跨论文引用数字做横向比较**

### 5.4 指标与人类偏好可能脱节

一项基准工作报告称 **FID/PSNR 与人类偏好相关性很差**；在其人工对战中，人类有 **37.5%** 的情况下更偏好 Inpaint360GS 的结果，尽管聚合指标给出的排序相反。

> **不确定**：该基准工作的来源与时间无法确认，且它复现时改变了渲染分辨率（512 而非论文规定的 1/4 分辨率），**实测 202 秒 vs 论文报告的约 3 分钟口径不完全可比**。

**仍然值得警惕的结论**：**目视评估不能省**，尤其中间还有人工确认环节。

### 5.5 待补充：masked 指标的正式定义

**未找到 mPSNR / mLPIPS 作为 SPIn-NeRF 官方术语的确证。** 现有文献显示的是 ad-hoc 的"区域裁剪 LPIPS/FID"用法。

**建议**：在自建评测代码中**显式定义并固定**自己的 masked 计算规则（掩码、裁剪、padding 策略），写进文档，避免后续版本之间失去可比性。

---

## 六、方法调研结论

1. **用户指定目标后的归属链路可复用**：FlashSplat 做 2D→3D 基线，SAGA 用于已有训练特征的交互分割；自动候选生成与低可见度补偿另行验证。

2. **修复线走策略 1**：occlusion-aware 跨视角填充优先，生成式只用于剩余区域。3DGIC 的流程顺序是关键。

3. **光照线是缺口**：先做务实版局部压平，逆渲染（GS-ID / ComGS）作为第二阶段选项。

4. **评测必须先自建 paired GT**：否则整个项目没有可验证的进展度量。

5. **最接近本项目的具体方案是 Inpaint360GS**，其可复用机制见 [03-inpaint360gs.md](03-inpaint360gs.md)。

---

## 参考来源

### 识别 / Mask Lifting
- [SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields (CVPR 2023)](https://tisl.cs.toronto.edu/publication/202306-cvpr-spinnerf/cvpr23-spinnerf.pdf)
- [SBR-GS: Enhancing Scene Editability based on Static Background Reconstruction (IJCNN 2025)](https://www.semanticscholar.org/paper/SBR-GS%3AEnhancing-Scene-Editability-based-on-Static-Wu-Wang/907eceb906e4714ada3a233dce30c14706e154d0/figure/4)

### 修复 / Inpainting
- [3D Gaussian Inpainting with Depth-Guided Cross-View Consistency (CVPR 2025)](https://bytez.com/docs/cvpr/34786/paper)
- [AuraFusion360 (CVPR 2025)](https://www.computer.org/csdl/proceedings-article/cvpr/2025/436400q366/299cN7caFDq)
- [Inpaint360GS (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/html/Wang_Inpaint360GS_Efficient_Object-Aware_3D_Inpainting_via_Gaussian_Splatting_for_360deg_WACV_2026_paper.html)
- [GPGS: Consistent 3D Object Removal via Geometry-Aware 3D Inpainting (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/37515/41477)
- [Object Removal in 3D Gaussian Splatting via Multi-View Geometry Guidance and Feature Fusion (IEEE)](https://ieeexplore.ieee.org/abstract/document/11406268)
- [Geometry-Aware Diffusion Models for Multiview Scene Inpainting](https://ar5iv.labs.arxiv.org/html/2502.13335)

### 光照 / 阴影 / 合成一致性
- [ComGS: Efficient 3D Object-Scene Composition via Surface Octahedral Probes (arXiv:2510.07729)](https://arxiv.org/abs/2510.07729)
- [ComGS 项目页](https://nju-3dv.github.io/projects/ComGS/)
- [GS-ID: Illumination Decomposition on Gaussian Splatting (ICCV 2025)](https://cislab.hkust-gz.edu.cn/publications/gs-id/)
- [3D Gaussian Inverse Rendering with Approximated Global Illumination (arXiv:2504.01358)](http://export-test.arxiv.org/pdf/2504.01358)

### 其他被提及的工作（未逐一核实）
GOR-IS、OVR-GS、Ref2Inpaint、3DSceneEditor、SpotLessSplats、InFusion、GScream、Gaussian Grouping、MALD-NeRF、RefFusion、NeRFiller、GauGroup、Inpaint360GS 数据集

### 本轮技术纠错来源

- [SAGA 官方 v2 README](https://github.com/Jumpat/SegAnyGAussians/tree/v2)：数据准备、尺度提取、亲和特征训练和交互 GUI；与早期 main 区分。
- [SAGA 论文](https://arxiv.org/abs/2312.00860)：尺度门控亲和特征与多粒度分割。
- [FlashSplat 官方仓库](https://github.com/florinshen/FlashSplat)：ECCV 2024，固定 3DGS 的闭式标签分配；本轮未执行算法数值复现。
