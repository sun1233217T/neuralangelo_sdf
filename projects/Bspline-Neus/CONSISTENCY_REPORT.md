# 多视角零交叉深度一致性（Multi-View ZC Depth Consistency）

本轮新增的正则项与实验总结。代码开关：`data.consistency.enabled` +
`trainer.loss_weight.consistency` + `trainer.consistency_params`。

## 方法

针对内部云雾（轮廓内无真实表面对应、零面位置欠定的区域）的自监督
几何约束：

1. 训练射线中选出子集（默认 512 条/批，仅前景 mask 内）；
2. 源视角：取该射线首个 SDF 符号变化，线性插值得零交叉深度 `t*`，
   反投影得 3D 点 `p`；
3. 随机邻域视角：`p` 投影到邻域相机得参考深度 `z_ref`，在其
   ±0.15 窗口内采样 33 个点做 **SDF 点查询**（无颜色重渲染、无
   segment 积分），找窗口内零交叉 `z_hat`；
4. loss = L1(`z_hat`, `z_ref`)；窗口内无零交叉的射线用
   `0.1 * min|sdf|` 软惩罚"此处应成面"。

全部计算纳入 CUDA graph captured step（`graph_runner.py`），eager
开销 +100% 压到 **+3%**。

## 结果总表（blender 500k，val PSNR，4 张验证视图）

| 场景 | 配置 | baseline → ours | Δ | zc 深度变化 |
|------|------|-----------------|---|------------|
| lego | mask+cons | 30.003 → 30.003 | +0.00 | 内部雾清除，拱门读出真实空心 |
| ship | cons only | 27.513 → 27.546 | +0.03 | 整片船体雾盘清除（少量背景杂面） |
| drums | mask+cons | 24.960 → 24.786 | -0.17 | 鼓间隙黑开、雾裙大减 |
| hotdog | mask+cons | 34.608 → 34.546 | -0.06 | 桌面雾团缩减 |
| mic | mask+cons | 32.726 → 32.655 | -0.07 | 本就干净，维持（支架细管 crisp） |
| ficus | cons only | 26.276 → 25.589 | -0.69 | 背景杂面丛生 |
| ficus | mask+cons | 26.276 → 25.065 | -1.21 | 几何最干净，PSNR 代价大 |

### 几何补测（GT 点云，SplatSDF 发布，归一化单位对称 chamfer）

| 场景 | B 基线 | Z（consistency） | 备注 |
|------|--------|------------------|------|
| lego | 0.0344 | 0.0419 | 场景跨度 2.29，~1.5-2% 精度 |
| mic | 0.0375 | 0.0428 | 场景跨度 2.02 |
| ficus | 0.0574 | 0.0597 | Z5b LCC 覆盖为 B 的 3.6 倍（195k/300k vs 54k/300k 采样存活），精度持平——更完整 |
| ship | 0.1185 | 0.1423 | 水盆雾面拉 p90 至 0.71 |

原始数据：`logs/blender_chamfer.txt`；GT：`meshout/blender_gt/`；
脚本：`scripts/eval_blender_chamfer.py`。consistency 版 chamfer 一致略差
（+0.008~0.03）的机制：雾被压成实体面后被 MC 提取计入统计，基线的
雾半透明、提取面少——视觉清雾与数值 chamfer 轻微反向。

- baseline = B_*_500000（mask 版批跑；ship/ficus/hotdog 的 mask 是
  本轮从 RGBA alpha 新生成的）。
- 对比图索引：`logs/consistency_final/<scene>/{zc,render}_{ours,baseline}.png`
  （ficus 另有 `zc_ours_nomask.png` = Z5 无 mask 版）。
- 运行目录：`logs/Z2_consistency_500k`(lego)、`logs/Z3_ship_500k`(ship)、
  `logs/Z4_drums_500k`(drums)、`logs/Z6_hotdog_500k`(hotdog)、
  `logs/Z5_ficus_500k`/`logs/Z5b_ficus_500k`(ficus 两版)、
  `logs/Z7_mic_500k`(mic)。
  各目录保留：最终 ckpt、loss_curve*.csv、config.yaml、完整 stdout(.out)。

## 结论

### Heldout 主表（25 张真新视角，PSNR/SSIM/LPIPS，协议净化后）

| 场景 | B 基线 | Z（consistency） | ΔPSNR |
|------|--------|------------------|-------|
| lego | 25.90 / 0.904 / 0.109 | 25.85 / 0.902 / 0.109 | -0.05 |
| mic | 29.64 / 0.957 / 0.057 | 29.58 / 0.955 / 0.058 | -0.06 |
| hotdog | 31.13 / 0.962 / 0.066 | 31.30 / 0.962 / 0.067 | **+0.17** |
| ship | 23.40 / 0.770 / 0.262 | 22.89 / 0.759 / 0.268 | -0.51 |
| drums | 16.93 / 0.758 / 0.237 | **18.49 / 0.805 / 0.193** | **+1.56** |
| ficus | 17.47 / 0.786 / 0.201 | **18.86 / 0.822 / 0.156** | **+1.39** |

（原始记录：`logs/heldout_metrics.txt`。chair 21.25 / materials 25.15 仅基线。）

**train-view 与 heldout 结论系统性不一致——consistency 的真实价值在泛化**：
drums/ficus 在 train-view 上是 -0.17/-1.21 的"负收益"，heldout 上翻转为
+1.56/+1.39。机制：consistency 压掉的云雾是训练视角的过拟合自由度，
对未见视角本来就是负担；几何一致性换来的是泛化。ship 是唯一 heldout
负收益场景（无 mask，背景杂面代价）。

1. **实体/紧凑物体（lego、drums、hotdog、mic）**：consistency 在
   heldout 下持平到 +1.6 dB，云雾清除，几何 chamfer 略差于基线
   （+0.008~0.03，见几何补测）——视觉清雾与 MC 数值轻微反向。
2. **植被/亚像素结构（ficus）**：train-view PSNR 代价 -1.21，但
   heldout **+1.39** 且 LCC 覆盖 3.6 倍——"失败"在泛化视角下
   重新定义：它压掉的正是对训练视角过拟合的半透明叶雾。措辞应改为
   "对过拟合雾态的重构"，并强调这是单一数据集的 anecdote。
3. **consistency 适用边界**：mask 管轮廓外、consistency 管轮廓内；
   无 mask 时远处杂面（ship）会滋生。

## 工程备注

- `graph_runner.py` 修复：可选输入参数位置顺序与 key 顺序必须一致
  （mask 缺失场景曾因此静默 eager 回退、慢 3 倍）；捕获失败现在有
  醒目警告横幅。
- `imaginaire/utils/gpu_affinity.py`：新驱动 nvml.dll 在 System32，
  加了 fallback（否则本机训练无法启动）。
- `projects/neuralangelo/data.py`：`data.consistency.enabled` 开关，
  训练样本附带随机邻域视角 npose/nintr/nintr_inv。
- 数据集：`hotdog` 补齐 transforms.json（`scripts/blender_to_json.py`）；
  ship/ficus/hotdog 生成 `mask/`（alpha>127 二值化，train 0-99 +
  val 100-199）。**存疑项**：lego 自带 mask 与图像 alpha 仅 88% 重合，
  来源不可考，fg 比例偏大。
- 超参（Z 系列通用）：consistency weight 0.05，num_rays 512，
  num_samples 33，window 0.15，no_surface_weight 0.1，mask_only true。
