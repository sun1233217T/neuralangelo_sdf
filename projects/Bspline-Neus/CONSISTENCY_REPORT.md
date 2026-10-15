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

1. **实体/紧凑物体（lego、drums、hotdog、ship、及本就干净的
   mic/materials/chair）**：consistency 以 ≤0.2 dB 的 PSNR 代价稳定
   清除内部云雾，是几何质量的大幅净收益。**mask 管轮廓外、
   consistency 管轮廓内，两者互补**——ship 无 mask 时轮廓外会滋生
   远处杂面。
2. **植被/亚像素结构（ficus）不适用**：叶片小于像素、跨视角严重
   遮挡，零交叉跨视角一致的前提被系统性违反；且雾态本身就是该
   场景的光度最优解，强行实体化必然损失 PSNR。这是清晰的适用
   边界，建议作为 limitation 报告或在植被类场景降权/关闭。

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
