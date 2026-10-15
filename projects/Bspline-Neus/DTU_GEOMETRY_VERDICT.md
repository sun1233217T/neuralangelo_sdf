# DTU scan24 几何质量裁决报告

2026-10-15。本报告记录 W2（当前 DTU 基线）的几何质量复测、根因诊断、
以及 Z8"几何栈全开"抢救实验的裁决结果。

## 1. 结论（TL;DR）

**抢救失败，论文策略转向。**

- W2 基线 chamfer（官方 DTU 协议）：d2s 7.61 / s2d 9.40 / **overall 8.51**
  （raw eval units，clip 20；neuralangelo 同级别 ~0.6）
- Z8（mask + consistency 0.05 + thin_shell 0.1 + prune 全开，500k）：
  d2s 7.32 / s2d 10.29 / **overall 8.81** —— 无改善
- 机制诊断：重建出的不是物体表面，而是**弥漫评估盒的碎片云雾**；
  NeuS 只需"每条射线第一个交叉点正确"，云雾完全不影响 31 dB 的渲染，
  但 Marching Cubes 提取的零面与 GT 中位距离 ~100mm

## 2. 复测与排除性验证（每一步都有证据）

| 怀疑 | 验证 | 结果 |
|---|---|---|
| 评估协议/单位错位 | 相机中心相似变换对齐：49 视角，rmse=9.7e-5，chamfer 不变（8.5092→8.5092） | 排除 |
| 顶部分块假象 | MC 未焊接顶点造成"8418 连通分量"假象；merge_vertices 后最大分量 99.8% | 拓扑实际连贯 |
| 边界大壳 | 最大连通分量是 ±700 范围的场边界壳（99.8% 面积），被 ObsMask 协议裁掉 | 不影响 chamfer，但解释了视觉 |
| s_var 过钝（NeuS 糊） | W2 inv_std=95.8（密度过渡 ~3.4mm），足够锐利 | 排除 |
| 剪枝日程 | W2 无 prune，E22b 同程度碎裂（4869 分量/焊接后同类结构） | 与剪枝无关 |

**评估盒内的真实分布**（ObsMask 盒内、GT 单位）：

- W2：in-box d2s p50=106mm，仅 7.8% 的点在 20mm 内；盒内表面是单一
  99.8% 连片——"雾壳"（云雾的外包络面），物体表面被埋没
- Z8：p50=98mm，frac<20mm = 8.2% —— 完全一致

叠合图：`meshout/scan24_W2_overlay.png`、`meshout/scan24_Z8_overlay.png`
（绿=GT 点云，红=盒内重建表面；红云弥漫全盒）

## 3. 根因分析

**结构性困难，与表示容量无关**（W2 只用了 dense 4.6% 的控制点也失败）：

1. NeuS 体渲染的监督信号只约束"射线积分出来的颜色"，零面位置在
   弱观测区（相机看不到、被遮挡、纹理弱）完全欠定；
2. 真实数据（DTU 49 视角 + 噪声）下欠定区域巨大，SDF 零面在这些
   区域自由游荡成碎片云；
3. blender 合成数据（100 视角、完美相机、物体紧凑）上同样的雾被
   consistency+mask 清除——因为那里"多视角都看见、只是零面欠定"；
   DTU 上是"**根本看不见**"，consistency 连惩罚目标都找不到；
4. 这解释了项目历史中全部几何异常（泡泡、z=-1.2 死 L4、视角外
   超细化碎片、云雾）的同源性。

## 4. 对论文的含义

**从"重建质量竞争"转向"表示设计竞争"**：

- 主张：局部显式层次 B 样条场——效率（4.6% 参数、3.5GB 显存）、
  局部编辑零漂移、区域裁剪分发、解析任意阶导数、白盒诊断、
  自适应细化引擎（THB 细分 + 重 banding + 剪枝）
- 证据：blender 8 场景 PSNR 持平 neuralangelo 类水平 + 内部云雾
  清除（见 CONSISTENCY_REPORT.md）+ 上述效率/可编辑性实验
- 诚实报告：DTU 几何作为 limitation，附本节的机制分析——
  "何时失效"的分析（blender 有效 vs DTU 无效的边界条件）本身就是
  有价值的科研内容

## 5. 相关文件

- mesh：`meshout/scan24_W2_500k.ply`、`meshout/scan24_Z8_500k.ply`
- chamfer 结果：`meshout/eval_W2/results_024.json`、`meshout/eval_Z8/results_024.json`
- 叠合图：`meshout/scan24_W2_overlay.png`、`meshout/scan24_Z8_overlay.png`
- Z8 训练：`logs/Z8_scan24_500k/`（config、loss 曲线、最终 ckpt、渲染图）
- 评估协议：`projects/sdf_angelo/scripts/eval_DTU_mesh.py`，
  `scale_mat_0` + ObsMask + max_dist=20 官方口径
