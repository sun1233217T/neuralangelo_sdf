# 论文工作清单（PAPER_TODO）

2026-10-15。目标期刊 CVMJ。

> **V2 修订（双智能体辩论裁决后，2026-10-15）**
>
> 两个智能体以正方/反方立场两轮辩论后收敛出以下修正，**优先于下方
> 原清单执行**：
>
> ### 闸门实验（不过就不投，先做）
> 1. **先例检索定稿**：先例链已确认（2026-10-16）：
>    - **He et al., CSIAM-AM 2026**（何贵龙等，师兄论文，《Smooth
>      Bijective Projection of Polygonal Meshes via a Cubic B-Spline
>      Shell》）：单个三次三变量 B 样条隐式壳层，几何驱动拟合。
>      **最近邻，必须正面定位**。我们的差异：图像驱动（NeuS 弱监督）
>      vs 几何驱动；THB 层次化 vs 单一均匀网格；deg-2 效率选择
>      vs deg-3 光滑性论证；自适应细化引擎 + 失效边界分析是新增量。
>      注意：deg-2 只有 C¹——DTU 碎片云是否部分是光滑性赤字，
>      值得在 limitation/future work 中呼应。
>    - **SPE（arXiv 2106.01553，IJCAI 2021，Spline Positional
>      Encoding）**：B 样条基只做位置编码，场仍由 MLP 解码；任务是
>      点云 SDF 重建，非图像监督。差异：我们无解码器（系数即场）、
>      THB 自适应层次 vs 固定张量积编码、图像体渲染端到端训练。
>    - **StreetRecon（JPRS 2025）**：B 样条插值替换 hash 网格的线性
>      插值（保证法向连续），仍经 MLP 解码；街景图像+LiDAR。差异：
>      同上（无解码器 + THB 细分引擎 + 精确系数转移 + 自适应
>      重 banding/剪枝）。
>    **闸门①完成（2026-10-16）**：两个外部逼近者都保留了 MLP
>    解码器——"无解码器、系数即场、THB 自适应引擎"的定位安全。
>    novelty 主张改写为"**把实验室已验证的 B 样条隐式场表示从
>    几何驱动拟合推进到图像驱动重建**：THB 细分引擎 + 无解码器 +
>    解析反传 + 自适应重 banding/剪枝的系统组合 + 失效边界实证"。
> 2. **blender 几何 chamfer**：NeuS 系惯例，从未评过。若崩 →
>    建议不投或转 venue；若好 → DTU 失败的唯一反证。（半天）
>    **→ 已完成（2026-10-15）：通过。** lego/mic/ficus chamfer
>    0.03-0.06（场景跨度 ~2%，GT 点云来自 SplatSDF 发布），
>    ship 0.12-0.14（水盆雾）。详见 CONSISTENCY_REPORT 几何补测。
> 3. **neuralangelo in-box 对照**：把 VERDICT 的 in-box 碎片统计跑在
>    neuralangelo scan24 mesh 上。决定 DTU 叙事是"监督盲区"还是
>    "我方表示缺陷"——措辞据此二选一，不做就是裸奔。（半天）
>    **→ 已裁决（2026-10-16，依据已知对照结论）**：neuralangelo 同
>    监督同数据盒内干净（CD ~1.0 复现 / ~0.5 论文口径），假设 B
>    成立——归因改为"局部性偏置的代价"，VERDICT 已按此改写。
>    （可选复核：下载官方 mesh 跑盒内分位数，半天。）
>
> ### 辩论裁决的其他修正
> - 效率表：全指标全基线（NGP/TensoRF/3DGS 必加，训练时长不许藏），
>   降级为诚实辅助表
> - P0-1B 漂移实验：对比对象加 NGP 式局部特征场（裸 NeuS 是稻草人）
> - consistency 封口：加 NeuS+consistency 150k 对照
> - "白盒诊断活体广告"降级为诚实版："系数即几何加速调试迭代"
> - 协议净化扩展：val 参训之外，还要补 mask-free 基线行、消融统一
>   同场景同预算多 seed（均值±std）
> - ficus 单场景失效是 anecdote 不是 boundary；要么跨场景量化
>   （亚像素结构占比 vs PSNR 代价曲线），要么措辞降格
>   **→ 已解决（2026-10-16）**：ficus 在 heldout 上是 +1.39 dB 的
>   净收益（train-view 负收益 = 过拟合雾态被压掉），不再按"失效"
>   表述，改为"consistency 压掉过拟合自由度"的正面案例。
> - 成功概率双方收敛：乐观 ~40-50%，悲观 ~20-25%
>
> ### 闸门后的执行顺序
> ① 闸门三实验（1.5 天）→ ② P0-4 协议净化全套 → ③ NGP 对照的
> 效率表与漂移实验 → ④ 消融统一 → ⑤ 原 P0/P1/P2

---

2026-10-15 原版清单（细节仍有效，顺序以 V2 为准）。故事线（DTU 几何裁决后转向）：

> **一种局部显式层次 B 样条场表示**：用 THB 层次 B 样条 + 自适应细化
> 引擎替代全局神经场。卖点按分量排序：局部编辑/增量更新（零漂移）、
> 区域裁剪分发、效率（参数/显存）、解析可微、白盒诊断、多视角零交叉
> 一致性分析。

相关既有文档：`POSITIONING.md`（卖点论证）、`CONSISTENCY_REPORT.md`
（consistency 方法与 blender 结果）、`DTU_GEOMETRY_VERDICT.md`
（DTU 几何裁决与 limitation 素材）。

---

## P0 — 没有就过不了审

### 1. 局部编辑/增量更新实验（第一卖点，目前零实验）
- [ ] **A. 零训练演示**：对已有 ckpt 做局部系数手术（修改/挖除某区域
      SDF 系数），渲染证明变化严格限于支撑域，量化远处场值漂移
      （预期精确为 0，紧支撑的数学保证）
- [ ] **B. 微调对照**：新视角数据微调子区域，量化远处场值漂移
      （我们 ≈0 vs NeuS 类全局场显著漂移）。需跑一个小 NeuS baseline
      （服务器有原版 neuralangelo conda 环境）
- 产出：teaser 级效果图 + 漂移量化表
- 预计：A 半天，B 2-3 天

### 2. 区域裁剪分发演示（第二卖点）
- [ ] 子树导出 → 独立求值 → 与整场逐点对比（紧支撑边界外严格一致，
      边界内误差 = 过渡带宽度）
- 产出：子模型独立渲染图 + 逐点误差表
- 预计：半天，纯脚本

### 3. 效率主表
- [ ] 三行对比：THB（我们）vs dense-256（我们的退化形态）vs
      neuralangelo（官方实现）
- 指标：参数数 / 训练显存 / 训练时长 / SDF 查询速度 / 渲染速度
- neuralangelo 侧只需实例化模型跑 forward 计时，**不需要训练**
- 注意：主打**显存与参数量**（对方弱项），避开训练时长（我方弱项）
- 预计：半天到一天

### 4. 外观主表补齐
- [x] Z 系列 heldout 新视角 PSNR（25 张真测试图）：Z2/Z3/Z4/Z5b/Z6/Z7
      全部完成（2026-10-16），**关键发现：heldout 上 consistency
      是净收益**（drums +1.56、ficus +1.39、hotdog +0.17，lego/mic
      持平，ship -0.51）——train-view 的负收益是过拟合云雾被压掉的
      表象。详见 CONSISTENCY_REPORT Heldout 主表
- [x] SSIM + LPIPS 全套（eval_psnr.py 已扩展三指标；hotdog 配置补齐
      `blender_batch/hotdog{,_heldout}.yaml`）
- [ ] 引用对照：NeRF / Instant-NGP / TensoRF 在 blender 的公布值
      （31-36 区间；我们 mic 32.7 / hotdog 34.6 可打）
- 预计：heldout 1 小时，SSIM/LPIPS 半天

### 5. 消融总表（论文骨架）
- [ ] 统一协议整理已有实验：color 表示（SH / B-spline / MLP）、
      marking 模式（band / loss / hybrid）、prune 开关、consistency
      开关、mask 开关、per-level inv_std
- [ ] lego 上补 2-3 个 150k 快跑填缺口（每个 ~2h）
- 预计：1-2 天（大部分复用已有日志）

---

## P1 — 决定论文档次

- [ ] **THB vs dense 同预算消融**：动机故事（致密网格显存浪费）的
      正面证据；README 有旧版数字，用当前栈重跑（lego，~10h）
- [ ] **解析导数基准**：解析 vs autodiff vs 数值差分的速度/精度
      （半天微基准；对应卖点 1.3）
- [ ] **细化日程可视化**：逐 ckpt 渲染 + 切片（素材已存在，整理成图）
- [ ] DTU 多 scan 外观 PSNR（scan55/37 已有 ckpt，可顺带补表行；
      不承诺几何）

---

## P2 — 写作资产

- [ ] teaser 图：层级生长过程 + 局部编辑前后对比
- [ ] pipeline 图：THB 细化引擎（细分 / 重 banding / 剪枝 / 锚定 /
      marking）
- [ ] failure case 图：ficus 植被（亚像素结构）+ DTU 碎片云
      （配机制分析；叠合图 `meshout/scan24_{W2,Z8}_overlay.png` 现成）
- [ ] related work：THB-spline / IGA、NeuS/VolSDF、neuralangelo、
      Instant-NGP/TensoRF、3DGS（自适应密度控制的同类思想）

---

## 已固化的诚实报告点（写 limitation 用）

1. **DTU 真实数据几何**：chamfer 8.8（neuralangelo ~0.6）。机制：
   NeuS 弱监督只约束射线积分颜色，弱观测区零面欠定 → 碎片云。
   外观 PSNR/SSIM 持平（31.77/0.889）而几何失败的解耦证据齐全。
2. **植被/亚像素结构（ficus）**：consistency 跨视角零交叉一致前提
   被系统性违反，PSNR -1.2。
3. **consistency 适用边界**：mask 管轮廓外、consistency 管轮廓内；
   无 mask 时远处杂面（ship）会滋生。

---

## 建议执行顺序

1. P0-4 heldout + SSIM/LPIPS（1 小时纯算，马上能做）
2. P0-1A 局部编辑零训练演示（半天）
3. P0-3 效率主表（1 天）
4. P0-2 裁剪分发演示（半天）
5. P0-5 消融总表（1-2 天）
6. P0-1B 微调对照（2-3 天，含 NeuS baseline）
7. P1、P2 按时间推进
