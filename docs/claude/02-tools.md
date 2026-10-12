# 02 · 工具评估：SuperSplat 与 LichtFeld Studio

> 评估目的：判断现成工具能覆盖本项目的哪一层，以及哪些必须自研。
>
> 整理日期：2026-10-12 ｜ 状态：调研完成，**含较多待实测验证项**
>
> **信息强度警告**：这份文档的 **待验证 / 不确定** 条目是全部文档中最多的。两个工具迭代都快，且本次调研受网络条件限制（多域名抓取被策略拦截）。具体到某个功能在哪个版本引入，**请以官方仓库和文档为准**。

---

## 一、核心结论

**这两个工具不是同类替代品，而是互补的两端：**

| | **SuperSplat** | **LichtFeld Studio** |
|---|---|---|
| 本质 | 浏览器里的**编辑器 + 查看器** | 原生**训练工作站** |
| 技术栈 | TypeScript / PlayCanvas Engine / WebGPU | C++23 / CUDA（训练）+ Vulkan（渲染） |
| 覆盖环节 | 编辑、优化、发布 | 训练 → 检视 → 编辑 → 导出 |
| 作者 | PlayCanvas（商业引擎公司，工具开源作生态） | MrNeRF（awesome-3DGS 作者） |
| License | **MIT** | **GPLv3** |
| 安装 | Node.js，浏览器打开 | 编译 30–90 分钟或买 Windows 二进制 |

**一句话**：SuperSplat 是"拿到 PLY 之后怎么办"，LichtFeld 是"从图像开始到 PLY 怎么办"。

**关键差异**：**LichtFeld 提供 MCP + Python 插件系统，SuperSplat 没有脚本接口。** 这一条决定了谁适合当自动化后端。

---

## 二、SuperSplat

### 2.1 定位与形态

- 2023 年 11 月首版 → 逐步开源（编辑器起初即开源，**查看器后来也以 MIT 开源**）→ 现均在 `playcanvas/supersplat` 下，MIT 许可
- 当前为 v2.x 系列
- **纯浏览器应用**，WebGPU 渲染，无原生后端

### 2.2 选择工具（8 种，核心资产）

| 类别 | 工具 |
|---|---|
| **2D 屏幕空间** | Rectangle、Brush、**Flood Fill**（连通域）、Lasso、Polygon |
| **3D 空间** | **Sphere**、**Box** |
| **颜色** | Eyedropper（按颜色相似度选） |

配套操作：Delete / Hide / Lock、Move / Rotate / Scale、**按选择集导出**。

内部用 **8-bit state 位标记**（selected / deleted / locked）。

### 2.3 格式支持

- **导入**：PLY、压缩 PLY（Draco）、SPLAT、SOG、LCC、**PLY 序列（4DGS 回放）**
- **导出**：PLY、压缩 PLY、SPLAT、**独立 HTML viewer（zip 打包）**

### 2.4 自动化程度（关键限制）

- **编辑器本身没有任何脚本接口。** 全部选择都是鼠标操作。
- 有一个 REST API，但**只负责上传和列出**场景（resumable multipart upload 到 superspl.at），**不能驱动编辑动作**。
- **内部架构是干净的 Command Pattern**（`EditOp` / `EditHistory`，带 `do()` / `undo()` / `destroy()`）。

> **待验证**：理论上可以 fork 后自己包一层脚本化的"选择 + 删除"，但这是自研工作量，不是现成能力。是否值得，取决于 [05-roadmap.md](05-roadmap.md) 的 阶段 1 验证结果。

### 2.5 对本项目的价值

| 能做 | 不能做 |
|---|---|
| 干扰物的**人工标注工具** | **自动识别干扰物** |
| 候选结果的**目视复核 + 手工修正** | **任何 inpainting / 空洞修复** |
| 快速预览"删完长什么样" | headless / CLI 批处理 |
| 作为交付时的分享查看器 | 语义分割 |

**最后一项不是问题，是边界**：SuperSplat 的设计哲学是"你告诉它删什么，它删"，**删掉就是删掉，它不会补**。所有修复工作都在它之外。

### 2.6 SuperSplat 作为标注器：结论需要附加条件

> **重要更正（验证后）**：早期版本直接写"用 SuperSplat 圈选后导出 Gaussian 索引，作为监督真值"。**在当前版本不能直接成立，已更正。**

**问题所在：SuperSplat 导出会打乱索引。**

| 事实 | 说明 |
|---|---|
| **1.0.0 起加载时做 Morton / Z-order 重排** | 为渲染做缓存友好的排序优化，大场景最多快约 4× |
| **后果** | 编辑后导出的索引**与原文件 Gaussian 索引不再一一对应** |
| **官方提供的开关** | 加载时传 `reorder: false`（新代码库中为 `skipReorder`）可保留原始顺序，代价是渲染性能下降 |
| **陷阱** | **压缩 PLY 和 `.splat` 永远按 Morton 序写**，不受该开关影响 |
| **导出过滤逻辑** | 按 deleted 状态、selection、opacity 阈值、NaN/Inf 有效性过滤后序列化 |

**未压缩导出与关闭重排不足以保证身份对应**：删除/过滤会使数组位置重新编号。标注需绑定源资产哈希、版本，并在过滤前保存选择集与源索引或稳定 ID；导出资产另存映射。具体重排开关和格式行为来自外部资料，仍需按固定版本实测。

**因此正确的结论是**：

> SuperSplat 可以很好地做**人工选择和视觉预览**，但**不能直接当作训练标签来源**。

**若要用作标注器，需要以下之一**：

| 方案 | 成本 |
|---|---|
| **fork SuperSplat，给每个 Gaussian 保存稳定 ID** | 中；需改 `EditOp` 层与导出逻辑 |
| **将选择集作为独立的 mask 文件导出**（而非依赖 PLY 里的索引顺序） | 中；同上 |
| **仅用作人工复核 / 视觉标注工具**，训练标签另行生成 | **低，推荐起点** |

**在未改造前，SuperSplat 的定位应限定为"人工复核前端"**，这仍是有价值的定位，只是不能承担"产生监督真值"的职责。

> 相关细节待实测确认，已列入 [05-roadmap.md](05-roadmap.md) 阶段 1/3 可选验证清单。

---

## 三、LichtFeld Studio

### 3.1 定位

MrNeRF 的作品（即维护 awesome-3D-Gaussian-Splatting 的人）。

宣传语是 "train, inspect, edit, automate, and export from a single native application" —— **以训练为核心，编辑和自动化是配套**。

### 3.2 硬能力

**训练**
- C++23 + CUDA 原生训练
- **MCMC 优化策略**
- bilateral grid 外观建模
- **3DGUT**（畸变相机模型）
- pose optimization
- save-and-resume

**CLI / headless**
- `LichtFeld-Studio -d <data> -o <out>` 跑训练
- `--headless --export ply` 走纯命令行

**编辑**
- brush / lasso / polygon / crop / align / compose
- 场景图分组
- 按到世界中心距离选择

**导出**
- PLY、SOG、SPZ、**USD**、**RAD**（SparkJS 的 LOD 流式格式）、独立 HTML viewer

**渲染后端演进**：早期 OpenGL → 曾加 CUDA renderer → **后来完全切到 Vulkan**（训练仍是 CUDA，viewer 的 Vulkan 与 trainer 共享 buffer）。

### 3.3 对本项目最关键的三个能力

#### (1) MCP + Python 插件 —— 两个工具间最大的差异

LichtFeld 明确提供 **MCP resources、MCP tools、embedded Python**，以及 **Python plugin system**（自定义面板、operator、tool、依赖）。

**含义**：**可以写程序驱动它的选择、删除、导出。**

这直接对应 [04-architecture.md](04-architecture.md) 里 `OperationPlan` 的"执行层"——**LichtFeld 是可选的编辑与训练适配器**。SuperSplat 完全没有这个面。

**⚠ 但不要把项目绑死在它上面。**

LichtFeld 的 API 能力比早期文档里"待验证"的判断更有把握——**选择 mask、soft delete、渲染 RGB/深度、Python 插件、导出这些确实已有**。但即使如此：

> **GSCleaner 的核心数据模型和算法应独立于 LichtFeld。**
>
> LichtFeld 作为**插件宿主 / 导入导出适配器**接入，作为**可替换的运行时后端**，而不是核心依赖。
>
> 理由：不能把项目绑定在它的**版本节奏、GPLv3 许可证、CUDA 环境要求**上。

**仍需实测的项**（列入 [05-roadmap.md](05-roadmap.md) 阶段 1/3）：

- [ ] Python API 是否能在 **headless** 下运行
- [ ] 删除和导出后**是否保持稳定 ID**
- [ ] 是否能读取原始相机数据
- [ ] 是否能对渲染图点击分割
- [ ] 是否能输出与源资产身份对应的 Gaussian mask
- [ ] mask 是否可作为训练或修复后端输入
- [ ] 训练期 mask 是否适用于本项目的**数据格式**
- [ ] GPLv3 对最终产品的影响（**商用前需法务确认**）

#### (2) 已内置 SAM 2.1 权重

模型资产列表中有 **SAM 2.1 Hiera base+ 的 `.lfw` 权重**，另有 **MoGe**（单目几何/深度/法向）、**RoMa v1**（稠密匹配）、**LPIPS**（评测用）。

changelog 中有 "New mask mode combining Ignore and Segment"、"Support in-memory masks and float-dtype colors for direct-scene plugins"、"Fix masks not loading or applying"。

**含义**：模型资产可减少部署准备，但是否提供可调用的图像点击分割、跨视角 Gaussian mask 和后端输入仍需实测；权重存在不等于能力链路已集成。

#### (3) 训练期 mask 支持

因为它是**训练器**，mask 可以参与训练本身。

**这直接对应 [00-requirements.md](00-requirements.md) 第 2 层"被压制背景"** —— 可排除干扰物像素，降低前景污染风险；这不会自动为隐藏背景提供监督。只有在其他视角可见且确认属于背景的原始像素才能用于背景恢复。

**这是 SuperSplat 结构上做不到的事。**

### 3.4 对本项目的价值

| 能做 | 不能做 |
|---|---|
| 可编程的 3DGS 后端 | mask lifting 逻辑（要自己接） |
| 训练期 mask 监督（改善被压制背景） | 空洞判定（要自己算） |
| SAM 2.1 权重/分割接口（待验证） | inpainting / 修复 |
| CLI headless 批处理 | 阴影清理 |
| Python 插件 / MCP 驱动编辑 | 碰撞 mesh 导出（有 USD，但需确认内容） |

---

## 四、两个工具合力能覆盖的层（对照五层模型）

| 需求层 | SuperSplat | LichtFeld Studio | 需自研 |
|---|---|---|---|
| **1. 归属判定** | ✗（仅手工） | ~ 有 SAM2 + mask 基础设施，但 lifting 逻辑要自己接 | **是** |
| **2. 被压制背景** | ✗ | ✓ 训练期 mask 可改善，但要设计策略 | 部分 |
| **3. 真空洞识别** | ✗ | ~ 可导出渲染深度自己算 | **是** |
| **4. 烘焙阴影残留** | ✗ | ✗ | **是** |
| **5. 表示与导出** | ✓ 格式全，含 HTML viewer | ✓ 格式全，含 USD / RAD | 碰撞 mesh |
| — | 人工修正、复核、分享 | 训练、编程式编辑、CLI | 生成式修复 |

**结论：两个工具能覆盖约 30–40% 的工程量，而且恰好是"不该自己写"的那 30–40%**（训练器、编辑器、格式转换、查看器）。

**剩下的核心研究部分——mask lifting、空洞判定、修复、阴影清理——它们都没有，也不打算有。那部分才是本项目的实质内容。**

---

## 五、两个必须先决策的约束

### 5.1 GPLv3 是硬约束

| | SuperSplat | LichtFeld Studio |
|---|---|---|
| License | **MIT** | **GPLv3** |
| 内部工具使用 | 随便用 | 随便用 |
| **交付给客户的产品** | 无顾虑 | **传染性**：若链接其代码发布，需开源整个作品 |

**务实的规避方式**：让 LichtFeld 作为**独立进程**通过 CLI 或 MCP 调用——**子进程调用 + 文件交换通常构成 aggregate 而非衍生作品**。

> **这属于法律判断，商用前请走法务确认，不要依据本文档做决策。**

相对地，SuperSplat 是 MIT，**没有任何这类顾虑**。

### 5.2 Linux 上只能从源码编译

LichtFeld 的预编译二进制**只发 Windows**（且有付费门槛：注册 + 捐款获取，GitHub Releases 不放二进制）。**Linux 必须自己编**：

**依赖要求**
- CUDA Toolkit 12.8+
- CMake 3.30+
- GCC 14+
- vcpkg

**已知坑**
- vcpkg 路径要下载编译约 **40 个依赖，首次 30–90 分钟**
- **SDL3 必须带 X11 或 Wayland 后端**，否则报 `No available video device`
- **WSL2 下 GUI 要走 WSLg**——需确认 X11/Wayland 能通。headless 模式可绕开（`--headless`），但那样就没了实时检视
- 有人反馈在滚动更新的发行版上编译失败（Arch AUR 上有 build 不通过的记录）

**硬件**：4090 + 驱动应满足（要求 compute capability 7.5+、驱动 570+）。

> **注意本项目的实际环境是 WSL2**（Linux 6.6.87.2-microsoft-standard-WSL2）。**GUI 可用性需实测**，headless 是稳妥起点。

**编译这一步要有心理预期，不是 `pip install` 那种体验。**

---

## 六、推荐架构分工

结合 [04-architecture.md](04-architecture.md) 的 Capability-Tiered Pipeline：

```
┌─ 主控（Python，自研）──────────────────────────────┐
│  CapabilityDescriptor → OperationPlan → 执行/报告   │
└──────┬─────────────────────────────────────────────┘
       │
       ├─ 独立算法后端 ──→ Inpaint360GS 官方基线 / 兼容训练器
       │
       ├─ 可选编辑与训练适配器 ──→ LichtFeld Studio
       │     · COLMAP 数据集训练（Tier A）
       │     · 训练期 mask 监督（改善第 2 层）
       │     · 验证内置 SAM2.1 的点击分割与 mask 输出
       │     · Python 插件 / MCP 驱动选择与删除
       │     · CLI headless 批处理
       │
       ├─ 自研研究组件（实质工作）
       │     · mask lifting + 低可见度补偿
       │     · 真空洞判定（跨视角可见性统计）
       │     · 修复（3DGIC / Inpaint360GS 路线）
       │     · 阴影残留清理
       │
       └─ 人工复核与交付 ──→ SuperSplat
             · Box / Flood Fill 手工修正
             · mask 标注（反哺评测集）
             · 按选择集导出
             · HTML viewer 交付
```

**分工理由**：LichtFeld 可承担部分可编程的 3DGS 后端能力；算法主线不应依赖它，SuperSplat 承担"人在环的复核前端"，两者**通过 PLY 文件交换，不产生代码耦合**。文件或进程边界本身不是许可证兼容性结论；适配器集成方式另行核验。

---

## 七、信息可信度说明

### 已核实（来自官方页面或仓库）

- SuperSplat：MIT 许可、8 种选择工具、仅上传型 REST API、Command Pattern 架构（`EditOp` / `EditHistory`）、格式支持列表
- LichtFeld：GPLv3、C++23/CUDA、MCP + embedded Python + Python 插件系统、内置 SAM 2.1 权重（含 MoGe / RoMa / LPIPS）、Vulkan 渲染器、仅 Windows 预编译包、Linux 编译依赖与已知坑

### 待验证（可选适配器清单，不阻塞官方基线）

1. LichtFeld 能否从源码编译通过（WSL2 环境）
2. headless 训练 → 导出 PLY 全链路
3. Python 插件 / MCP 的实际 API 粒度：**按索引批量删除？读回选择集？headless 下可驱动？**
4. WSL2 下 GUI 可用性（WSLg）
5. SuperSplat 的 `EditOp` 架构 fork 后包脚本化编辑的可行性

### 不确定（不应作为决策依据）

- **两个工具的具体版本号与功能引入版本**。检索到的资料指向的版本和发布时间无法核实，有来源互相矛盾，甚至有标注为未来日期的条目。
- LichtFeld 编辑能力的实际效果**未实测**。"MCP 能驱动选择删除"是从其定位和文档描述推出的。

---

## 参考来源

### SuperSplat
- [SuperSplat 官方产品页](https://playcanvas.com/products/supersplat)
- [SuperSplat Editor README（GitHub）](https://github.com/playcanvas/supersplat)
- [SuperSplat 3DGS Viewer 开源公告](https://blog.playcanvas.com/supersplat-3dgs-viewer-is-now-open-source/)
- [SuperSplat API & Integrations](https://developer.playcanvas.com/user-manual/supersplat/api-integrations/)
- [SuperSplat API Reference](https://developer.playcanvas.com/user-manual/api/supersplat/)
- [SuperSplat 编辑工具与选择工具（DeepWiki）](https://deepwiki.com/playcanvas/supersplat/9-editing-tools)
- [SuperSplat Edit History / EditOp 架构（DeepWiki）](https://deepwiki.com/playcanvas/supersplat/4.3-edit-history-and-undo-system)
- [SuperSplat for LichtFeld Studio 插件](https://github.com/playcanvas/supersplat-lichtfeld-plugin)

### LichtFeld Studio
- [LichtFeld Studio 官网](https://lichtfeld.io/)
- [LichtFeld-Studio（GitHub）](https://github.com/MrNeRF/LichtFeld-Studio)
- [LichtFeld Studio Linux 编译说明（Wiki）](https://github.com/MrNeRF/LichtFeld-Studio/wiki/Build-Instructions-%E2%80%90-Linux)
- [LichtFeld Studio 构建与发行文档](https://raw.githubusercontent.com/MrNeRF/LichtFeld-Studio/master/docs/building_and_distribution.md)
- [LichtFeld Studio Releases（含 SAM 2.1 等模型资产）](https://github.com/MrNeRF/LichtFeld-Studio/releases)
- [LichtFeld Studio v0.5.3 发布说明（radiancefields.com）](https://radiancefields.com/lichtfeld-studio-v0.5.3-drops-its-cuda-renderer-for-a-full-vulkan-viewer)
