# 叉车多视角 SAM 分部件标注 revision3

8 个 COLMAP PINHOLE 800×600 目标视角：left 000401/000644/001001，right 001371/001401/001737/001801/002401。

本轮先逐张真实去畸变 RGB 确认像素坐标，再用 SAM 正点/负点及部件 box 分割。每个部件输出 component mask，并用人工 polygon/box 硬裁剪以防墙面、地面吞并；并集输出 `_mask.png`、绿色 overlay `_preview.jpg`。可复用脚本 `scripts/wanwei/segment_forklift.py`，本轮全部输入提示在 `prompts.json`，完整提示也嵌入 `manifest.json`。

8 张目标 mask 均经过目视审查，接受为 **局部正支持**，不是完整轮廓或零误删证明。主要车身、门架、传感器和可见叉齿获得覆盖，但横杆、按钮、链条边缘仍可能漏选；大部分视图有画面裁断。因此 `complete_target_annotation=false`，禁止把 target mask 外像素自动当作负证据。left001001 的底部长叉已另用独立提示覆盖，避免首轮漏掉整条叉齿。

另外 9 张 `_negative_mask.png` 只覆盖人工明确识别的墙、地面、货架等保留区域，不是 target mask 补集。蓝色汇总见 `negative-contact.jpg`。left002501/003048 是不同轨迹中的背景证据，不是清场真值。

revision2 的大地面/墙误选已全部拒绝，不能用于删除；revision3 才是当前接受的候选。这些 mask 只供后续 3D 正负支持审计，尚未编辑 PLY 或形成批准删除集合。

