# SDF-Angelo 方法评估与论文实验规划

本文档基于当前 `projects/sdf_angelo` 代码实现，总结方法定位、论文贡献、实验设计、消融实验、评估指标，以及该工作的 B 类期刊/会议适配性。

---

## 1. 方法定位

当前方法可以概括为：

> 在 Neuralangelo/NeuS-style SDF 重建框架上，引入稀疏 COLMAP 点云 SDF warmup 约束以加速并稳定几何训练；训练完成后提取 SDF mesh，并通过可见性过滤、UV 展开、神经 RGB 烘焙和视角相关 UV residual student 蒸馏，将隐式神经场转化为可实时渲染、可压缩部署的显式纹理网格表示。

建议论文题目方向：

- **Fast Neural SDF Reconstruction and View-dependent UV Texture Distillation**
- **Visibility-aware Neural SDF Mesh Extraction with UV Residual Distillation**
- **面向实时渲染的神经 SDF 显式化与 UV 视角相关纹理蒸馏方法**

建议论文主线：

> 从 Neural SDF 到可实时/可部署显式 UV 资产的转换与蒸馏框架。

不建议将论文主线只放在“点云 SDF 约束加速训练”上，因为该方向已有相近思想，单独作为创新点可能偏弱。更建议将其作为完整 pipeline 中的几何增强模块。

---

## 2. 当前方法模块与论文价值

### 2.1 稀疏点云 SDF warmup

相关代码：

- `projects/sdf_angelo/trainer.py`

当前实现包含：

- 从 COLMAP `points3D.bin/txt` 加载稀疏点云；
- 使用 `transforms.json` 中的 `sphere_center/sphere_radius` 进行归一化；
- 根据 COLMAP 重投影误差 `max_error` 过滤噪声点；
- 在训练早期加入点云 SDF 零水平集约束；
- 使用 Huber 风格损失降低离群点影响。

可写成的贡献：

> 利用 SfM 稀疏点云作为低成本几何先验，通过误差过滤和鲁棒 SDF warmup 约束提升 Neural SDF 早期几何收敛稳定性。

价值：

1. 利用 COLMAP 已有输出，额外成本低；
2. 有助于减少早期漂浮几何和错误表面；
3. 对稀疏视角、纹理弱区域、mask 数据可能有效；
4. 适合作为训练加速和稳定化模块。

风险：

1. 单独创新强度有限；
2. 点云噪声可能损害几何，需要误差过滤和 warmup 截止；
3. 需要用收敛曲线和几何指标证明有效性。

---

### 2.2 确定性表面追踪与表面细化

相关代码：

- `projects/sdf_angelo/model.py`

当前实现包含：

- sphere tracing 表面定位；
- sign flip 后二分 refinement；
- 可选 surface cache；
- 与 NeuS-style SDF volume rendering 结合。

论文中建议定位：

> 作为稳定获取 SDF 表面点和支持后续 UV/mesh 显式化的技术细节，而不是主创新点。

建议消融：

- refinement steps: `0 / 1 / 3 / 5`；
- surface cache on/off；
- 比较 depth error、normal consistency、单次渲染/采样耗时。

---

### 2.3 Visibility-aware mesh extraction

相关代码：

- `projects/sdf_angelo/scripts/extract_mesh.py`

当前实现包含：

- marching cubes SDF mesh 提取；
- largest connected component 过滤；
- alpha mask 可见性过滤；
- nvdiffrast depth visibility 过滤；
- 支持 `depth_percentile`、`depth_trim_low/high`、`depth_margin_ratio` 等鲁棒参数；
- 支持 mesh simplification、remesh、UV unwrap 前处理。

可写成的贡献：

> 设计可见性驱动的 mesh 清理与 UV 前处理流程，去除 SDF 提取中由不可见区域和背景区域产生的浮动几何，使隐式重建结果更适合导出为显式 textured mesh。

价值：

1. 对实际数据很有用；
2. 能减少浮动面和背景壳；
3. 降低 mesh 面数，提升后续 UV 处理和渲染效率；
4. 适合通过可视化展示论文效果。

风险：

1. 偏工程后处理；
2. 如果过滤过强，可能删除真实几何；
3. 需要通过参数消融展示鲁棒性。

---

### 2.4 UV 神经纹理烘焙

相关代码：

- `projects/sdf_angelo/scripts/extract_mesh.py`
- `projects/sdf_angelo/utils/mesh.py`
- `projects/sdf_angelo/uv_viewer/uv_cache.py`
- `projects/sdf_angelo/uv_viewer/texture_generator.py`

当前实现包含：

- 对 mesh 进行 UV 展开；
- 在 UV atlas 上 rasterize texel 到 3D surface point；
- 缓存 texel 对应的 point、normal、SDF feature；
- 根据相机位置生成 view direction；
- 查询 `neural_rgb(points, normals, rays_unit, feats, app)` 生成视角相关纹理；
- 支持 padding/dilation；
- 支持保存 UV bundle。

论文价值：

1. 将隐式神经场转为显式 mesh + UV 表示；
2. 支持传统图形管线和 Web/实时 viewer；
3. 为后续 student 蒸馏提供 teacher 数据；
4. 是连接 Neural SDF 与可部署资产的关键步骤。

需要重点区分三种外观表示：

| 表示 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| Static UV texture | 固定视角或平均视角烘焙 | 快、简单、兼容性好 | 难以表示视角相关效果 |
| Dynamic neural UV | 每帧根据相机查询 neural RGB 生成 UV texture | 质量接近 teacher | 仍依赖较重神经网络 |
| UV residual student | 用轻量 student 预测视角相关 residual | 质量/速度/大小折中较好 | 需要蒸馏训练 |

---

### 2.5 UV residual student 蒸馏

相关代码：

- `projects/sdf_angelo/scripts/export_uv_teacher_dataset.py`
- `projects/sdf_angelo/scripts/train_uv_student.py`
- `projects/sdf_angelo/uv_distill/model.py`
- `projects/sdf_angelo/uv_distill/render.py`

当前实现包含：

- 导出多视角 teacher UV texture；
- 使用 depth visibility/front-facing mask 筛选可见 texel；
- 支持原始视角、jitter 视角、interpolation 视角；
- 训练轻量 `UVResidualStudent`：

```python
input = [base_rgb, local_view_direction, latent_uv_feature]
output = rgb_delta
pred_rgb = base_rgb + delta
```

其中 local view direction 使用 tangent frame 表达：

```python
local_x = dot(ray, tangent)
local_y = dot(ray, bitangent)
local_z = dot(ray, normal)
```

这是当前最值得主打的论文贡献。

可写成的贡献：

> 将高成本视角相关神经颜色解码器蒸馏为 UV 空间轻量 residual student。该 student 以基础纹理、局部视角方向和可学习 UV latent grid 为条件，预测视角相关颜色残差，在保持较好外观质量的同时降低存储和推理成本。

优势：

1. 结构清晰，便于消融；
2. 与显式 mesh/UV 资产天然结合；
3. 可以从质量、速度、模型大小三个维度体现优势；
4. 相比单纯 static texture，更适合表示高光、非朗伯外观和视角相关颜色变化。

---

## 3. 建议论文贡献点

建议凝练为 3 个贡献：

1. **Sparse point guided SDF warmup**  
   提出基于 SfM 稀疏点云的 SDF 零水平集 warmup 约束，通过误差过滤和鲁棒损失提升早期几何收敛稳定性。

2. **Visibility-aware mesh extraction and UV baking**  
   设计可见性驱动的 mesh 清理与 UV 神经纹理烘焙流程，将隐式 SDF 表示转为紧凑、干净、可部署的显式 textured mesh。

3. **View-dependent UV residual distillation**  
   将高成本神经颜色解码器蒸馏为轻量 UV residual student，以 base texture、局部视角方向和 latent atlas 为条件，实现高效视角相关外观渲染。

---

## 4. 主实验设计

### 4.1 主对比表

建议使用 DTU、Tanks and Temples、Mip-NeRF 360 或自采真实数据进行评估。

| Method | Geometry prior | Texture/appearance | PSNR↑ | SSIM↑ | LPIPS↓ | Chamfer↓ | FPS↑ | Size↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Neuralangelo | 无 | implicit neural |  |  |  |  |  |  |
| SDF-Angelo baseline | 无 | implicit neural |  |  |  |  |  |  |
| Ours-Geo | sparse pc warmup | implicit neural |  |  |  |  |  |  |
| Ours-UV Static | sparse pc warmup | static UV texture |  |  |  |  |  |  |
| Ours-UV Dynamic | sparse pc warmup | neural UV cache |  |  |  |  |  |  |
| Ours-Student | sparse pc warmup | residual UV student |  |  |  |  |  |  |

预期结论：

- `Ours-Geo` 在早期训练和几何质量上优于 baseline；
- `Ours-UV Static` 速度最快但视角相关效果弱；
- `Ours-UV Dynamic` 质量接近 teacher 但仍有计算成本；
- `Ours-Student` 在质量、速度和大小之间取得更好的平衡。

---

### 4.2 几何训练实验

| 实验 | 设置 | 指标 | 目的 |
|---|---|---|---|
| Baseline | 无 pc_sdf | PSNR、Chamfer、F-score、训练曲线 | 对比原始训练 |
| +pc_sdf | 开启点云 SDF warmup | 同上 | 证明收敛更快/几何更准 |
| weight ablation | `0.5 / 1 / 2 / 5` | 几何指标、PSNR | 找最佳权重 |
| end_iter ablation | `10k / 50k / 100k / all` | 几何/纹理 | 证明 early stop 优于全程约束 |
| max_error ablation | `1 / 2 / 5 / no filter` | 几何质量 | 证明误差过滤有效 |

建议重点画：

- training PSNR curve；
- Chamfer curve；
- 20k/50k/100k iteration 的中间 mesh 可视化。

---

### 4.3 Mesh 导出与可见性过滤实验

| Variant | visible filter | depth filter | remesh/simplify | Faces↓ | Chamfer↓ | render PSNR↑ |
|---|---|---|---|---:|---:|---:|
| raw MC | ✗ | ✗ | ✗ |  |  |  |
| alpha visible | ✓ | ✗ | ✗ |  |  |  |
| depth visible | ✓ | ✓ | ✗ |  |  |  |
| full preprocess | ✓ | ✓ | ✓ |  |  |  |

必要可视化：

1. 原始 marching cubes mesh；
2. alpha visible 过滤后 mesh；
3. depth visible 过滤后 mesh；
4. UV textured mesh 渲染结果。

建议说明：

- 可见性过滤减少浮动面和背景壳；
- face 数减少；
- 几何完整性基本保持；
- 后续 UV 展开更稳定。

---

### 4.4 UV/student 外观实验

| 方法 | 描述 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Size↓ |
|---|---|---:|---:|---:|---:|---:|
| Teacher implicit | 原 SDF/neural RGB 渲染 |  |  |  |  |  |
| Static UV | 固定/平均视角 UV texture |  |  |  |  |  |
| Dynamic neural UV | 每视角查询 neural RGB 生成 texture |  |  |  |  |  |
| UV residual student | 轻量 student 预测 residual |  |  |  |  |  |

该表是论文核心表之一。需要强调：

- Teacher 是质量上限，不一定最快；
- Static UV 是传统显式资产 baseline；
- Student 应明显优于 static UV，并显著快于 heavy teacher/dynamic neural UV。

---

## 5. 消融实验设计

### 5.1 点云 SDF warmup 消融

| Variant | pc_sdf | Huber | error filter | early stop | Chamfer↓ | F-score↑ | 50k PSNR↑ | final PSNR↑ |
|---|---|---|---|---|---:|---:|---:|---:|
| baseline | ✗ | - | - | - |  |  |  |  |
| naive point SDF | ✓ | ✗ | ✗ | ✗ |  |  |  |  |
| filtered Huber | ✓ | ✓ | ✓ | ✗ |  |  |  |  |
| ours | ✓ | ✓ | ✓ | ✓ |  |  |  |  |

---

### 5.2 UV student 结构消融

| Variant | base RGB | local view dir | latent grid | view aug | PSNR↑ | LPIPS↓ | FPS↑ | Size↓ |
|---|---|---|---|---|---:|---:|---:|---:|
| static UV | ✓ | ✗ | ✗ | ✗ |  |  |  |  |
| MLP no latent | ✓ | ✓ | ✗ | ✓ |  |  |  |  |
| latent no view | ✓ | ✗ | ✓ | ✓ |  |  |  |  |
| full student | ✓ | ✓ | ✓ | ✓ |  |  |  |  |

该消融用于证明：

- local view direction 对视角相关外观有用；
- latent grid 提供空间相关的 residual 表达能力；
- view augmentation 提升 novel view 泛化；
- residual learning 优于直接预测 RGB。

---

### 5.3 Student 容量消融

| 实验 | 设置 | 指标 |
|---|---|---|
| latent_dim | `0 / 4 / 8 / 16 / 32` | PSNR、LPIPS、模型大小 |
| latent_scale | `2 / 4 / 8 / 16` | PSNR、显存、速度 |
| hidden_dim | `32 / 64 / 128` | PSNR、FPS |
| num_layers | `2 / 3 / 4` | PSNR、FPS |
| texture_size | `512 / 1024 / 2048 / 4096` | PSNR、FPS、资产大小 |

---

### 5.4 View augmentation 消融

| Variant | original | jitter | interpolation | PSNR↑ | LPIPS↓ |
|---|---|---|---|---:|---:|
| original only | ✓ | ✗ | ✗ |  |  |
| original + jitter | ✓ | ✓ | ✗ |  |  |
| original + interp | ✓ | ✗ | ✓ |  |  |
| full | ✓ | ✓ | ✓ |  |  |

---

## 6. 推荐评估指标

### 6.1 图像质量

- PSNR；
- masked PSNR；
- SSIM；
- LPIPS；
- novel view rendering 可视化。

已有基础：

- `projects/sdf_angelo/scripts/uv_mesh_psnr.py`

建议补充：

- SSIM；
- LPIPS；
- final average 指标保存到 JSON/CSV。

---

### 6.2 几何质量

- Chamfer Distance；
- F-score；
- Normal consistency；
- vertex/face count；
- mesh clean ratio，即过滤前后 face 数比例。

已有基础：

- `projects/sdf_angelo/scripts/eval_DTU_mesh.py`

---

### 6.3 效率与部署指标

必须报告：

- training time to target quality；
- mesh extraction time；
- UV baking time；
- teacher dataset export time；
- student training time；
- rendering FPS；
- GPU memory；
- model/asset size。

对本方法而言，**FPS 和资产大小** 与 PSNR 同等重要。因为显式化和 student 蒸馏的核心意义在于部署和实时渲染，而不是单纯超过 teacher 质量。

---

## 7. 推荐实验命令流程

### 7.1 Baseline 训练

```bash
python train.py \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --logdir logs/dtu_baseline \
  trainer.loss_weight.pc_sdf=0.0 \
  trainer.pointcloud_sdf.enabled=False
```

### 7.2 Ours 几何训练

```bash
python train.py \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --logdir logs/dtu_ours_pc_sdf \
  trainer.pointcloud_sdf.enabled=True \
  trainer.loss_weight.pc_sdf=2.0
```

### 7.3 提取原始 mesh

```bash
python projects/sdf_angelo/scripts/extract_mesh.py \
  --single_gpu \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --checkpoint logs/dtu_ours_pc_sdf/latest_checkpoint.pt \
  --resolution 1024 \
  --output_file meshout/raw_mesh.ply
```

### 7.4 提取可见性过滤 mesh

```bash
python projects/sdf_angelo/scripts/extract_mesh.py \
  --single_gpu \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --checkpoint logs/dtu_ours_pc_sdf/latest_checkpoint.pt \
  --resolution 1024 \
  --depth_visible \
  --alpha_threshold 0.5 \
  --output_file meshout/visible_mesh.ply
```

### 7.5 生成 UV textured mesh

```bash
python projects/sdf_angelo/scripts/extract_mesh.py \
  --single_gpu \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --checkpoint logs/dtu_ours_pc_sdf/latest_checkpoint.pt \
  --input_mesh meshout/visible_mesh.ply \
  --uv_textured \
  --texture_size 2048 \
  --uv_target_faces 200000 \
  --output_file meshout/ours_uv.obj
```

### 7.6 导出 UV teacher dataset

```bash
python projects/sdf_angelo/scripts/export_uv_teacher_dataset.py \
  --single_gpu \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --checkpoint logs/dtu_ours_pc_sdf/latest_checkpoint.pt \
  --mesh meshout/ours_uv.obj \
  --output_dir datasets/uv_teacher/dtu_scanXX \
  --texture_size 2048 \
  --jitter_per_view 2 \
  --interp_steps 2 \
  --pad_iters 2
```

### 7.7 训练 UV residual student

```bash
python projects/sdf_angelo/scripts/train_uv_student.py \
  --dataset_dir datasets/uv_teacher/dtu_scanXX \
  --output_dir logs/uv_student/dtu_scanXX \
  --steps 20000 \
  --batch_views 2 \
  --texels_per_view 32768 \
  --latent_dim 8 \
  --latent_scale 4 \
  --hidden_dim 64 \
  --num_layers 3 \
  --view_kinds original,jitter,interp
```

### 7.8 评估 UV mesh PSNR

```bash
python projects/sdf_angelo/scripts/uv_mesh_psnr.py \
  --single_gpu \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --checkpoint logs/dtu_ours_pc_sdf/latest_checkpoint.pt \
  --mesh meshout/ours_uv.obj \
  --split val \
  --texture_size 2048 \
  --mask_psnr \
  --debug_dir eval_debug/ours_uv
```

---

## 8. B 类期刊/会议适配性评估

### 8.1 总体判断

综合当前代码和方法设计，我的判断是：

> 该方法具备冲击 B 类期刊/会议的潜力，但需要将论文主线从“工程 pipeline”凝练为“面向实时显式化的 Neural SDF 到 UV residual student 蒸馏框架”，并补齐系统实验，尤其是质量、速度、模型大小、几何指标和充分消融。

如果实验完整、写作聚焦、可视化充分，适合投稿应用型/工程型 B 类会议或期刊。若只展示代码 pipeline 或少量定性结果，可能只能作为 workshop、中文核心或一般应用类期刊工作。

---

### 8.2 为什么有 B 类潜力

优势如下：

1. **问题有实际价值**  
   Neural SDF/Neuralangelo 质量高但训练慢、渲染慢、部署难。将其转为 mesh + UV/student 是实际应用中的重要问题。

2. **pipeline 完整**  
   已覆盖训练、mesh 提取、可见性过滤、UV 烘焙、teacher dataset 导出、student 蒸馏、PSNR 评估等环节。

3. **UV residual student 有明确创新空间**  
   该模块具备结构设计、消融实验和速度/大小优势，不只是简单后处理。

4. **实验故事容易讲清楚**  
   可围绕“implicit teacher 质量高但慢，static UV 快但质量低，ours student 兼顾质量和效率”构建论文主线。

5. **可视化效果直观**  
   mesh 清理、UV texture、novel view rendering、FPS 对比都容易做成直观图表。

---

### 8.3 目前距离 B 类的主要差距

1. **创新点还偏分散**  
   当前包含 pc_sdf、surface tracing、mesh filtering、UV baking、student distillation 多个模块。论文必须突出一个主贡献，否则容易被评为“系统拼接”。

2. **需要和强 baseline 对比**  
   至少需要比较：
   - Neuralangelo；
   - NeuS/VolSDF 或同类 SDF 方法；
   - static textured mesh；
   - 可能的话加入 3D Gaussian Splatting 或 BakedSDF/MobileNeRF 类方法讨论。

3. **必须有充分消融**  
   B 类审稿通常会要求证明每个模块的必要性。尤其是：
   - pc_sdf 是否真正提升几何；
   - visibility filter 是否提升 mesh 质量；
   - UV student 的 view direction、latent grid、augmentation 是否必要。

4. **效率指标不能缺**  
   如果目标是实时/部署，必须报告 FPS、模型大小、显存、导出耗时。只有 PSNR 不足以支撑论文结论。

5. **student 的实时性需要确认**  
   如果每帧全 atlas 预测 2048/4096 纹理，速度可能不足。需要实测，并考虑可见区域更新、降采样或 tile update 等策略。

---

### 8.4 推荐投稿定位

#### 更适合的方向

- 计算机图形学应用；
- 三维重建与可视化；
- 虚拟现实/增强现实资产生成；
- 多媒体系统与图像图形结合；
- 工程型视觉会议或期刊。

#### 不太建议直接冲击的方向

- 顶级计算机视觉会议主会，如果没有非常强的理论/算法创新和大规模对比；
- 纯理论方向期刊；
- 只看 benchmark SOTA 的三维重建论文路线。

---

### 8.5 B 类命中概率评估

以下是主观评估，假设实验结果正常且写作质量合格：

| 完成度 | 描述 | B 类适配性 |
|---|---|---|
| 低 | 只有 pipeline 和少量可视化 | 较低 |
| 中 | 有 PSNR、mesh 可视化、部分消融 | 有机会，但风险较大 |
| 高 | 有完整主表、几何指标、效率指标、充分消融、多个数据集 | 较合适 |
| 很高 | 在质量-速度-大小上明显优于强 baseline，并有高质量 demo | 有较强竞争力 |

当前代码基础属于“中到高”的起点，但论文结果还取决于实验完整度。

---

## 9. 建议优先完成的实验清单

### 必须完成

1. Baseline Neuralangelo/SDF-Angelo vs Ours pc_sdf 训练曲线；
2. DTU 几何 Chamfer/F-score；
3. raw mesh vs alpha/depth visible mesh 可视化；
4. Teacher implicit vs static UV vs dynamic UV vs UV student 的 PSNR/SSIM/LPIPS；
5. FPS、显存和资产大小对比；
6. UV student 消融：
   - no view direction；
   - no latent grid；
   - no view augmentation；
   - full model。

### 建议完成

1. texture size: `512 / 1024 / 2048 / 4096`；
2. latent_dim: `0 / 4 / 8 / 16 / 32`；
3. latent_scale: `2 / 4 / 8 / 16`；
4. depth visibility 参数消融；
5. 与 3DGS 或其他实时表示做质量/速度定性对比。

---

## 10. 最终评价

该工作最适合定位为：

> 面向实时渲染和可部署三维资产生成的 Neural SDF 显式化与 UV 视角相关蒸馏方法。

核心卖点不是单纯提高 PSNR，而是：

1. 用 sparse point SDF warmup 提升几何训练稳定性；
2. 用 visibility-aware mesh extraction 得到更干净的显式 mesh；
3. 用 UV residual student 将重型 neural RGB 解码器蒸馏为轻量视角相关纹理模型；
4. 在质量、速度、资产大小和部署兼容性之间取得较好折中。

对于 B 类期刊/会议：

> 如果补齐多数据集实验、强 baseline、完整消融和效率对比，该方法是有希望的。建议避免把论文写成“若干工程技巧集合”，而要聚焦在“Neural SDF 到可实时显式 UV asset 的蒸馏框架”这一清晰主线上。


## 11. DTU 全量实验脚本（新增）

为在 multi-GPU 服务器上批量跑完 15 个 DTU scan 的 baseline/ours 对比，已新增以下文件：

- 通用配置：
  - `projects/sdf_angelo/configs/dtu_base.yaml`
  - `projects/sdf_angelo/configs/dtu_baseline.yaml`（关闭 pc_sdf）
  - `projects/sdf_angelo/configs/dtu_ours.yaml`（开启 pc_sdf）
  - `projects/neuralangelo/configs/dtu_generic.yaml`（原始 Neuralangelo，未改动）
- 主控脚本：
  - `tools/run_dtu_full.py`

典型使用流程：

```bash
# 1. 训练：在 4 张 GPU 上并行跑所有 scan/method（含原始 Neuralangelo）
python tools/run_dtu_full.py train --gpus 0,1,2,3 --methods baseline,ours,neuralangelo

# 若只想跑 baseline + ours
python tools/run_dtu_full.py train --gpus 0,1,2,3 --methods baseline,ours

# 2. 评估：对每个 checkpoint 算 train PSNR 与 Chamfer
python tools/run_dtu_full.py eval --gpus 0,1,2,3 --methods baseline,ours,neuralangelo

# 3. 汇总：生成 CSV 表格、markdown 报告与收敛曲线
python tools/run_dtu_full.py aggregate
```

输出位于 `meshout/dtu_full/`，包含：

- `all_eval_records.csv`：每个 checkpoint 的详细指标
- `final_summary.csv`：每个 scan/method 在最终 checkpoint 的结果
- `method_averages.csv`：方法级别的平均指标
- `RESULTS.md`：可直接贴入论文的 markdown 表格
- `convergence_curves.png`：按 method 平均的 PSNR/Chamfer 收敛曲线

脚本支持把原始 `projects.neuralangelo` 作为第三个方法一起训练/评估，
从而直接得到同数据集、同迭代数下的 Neuralangelo 复现结果，不再需要引用论文值。
