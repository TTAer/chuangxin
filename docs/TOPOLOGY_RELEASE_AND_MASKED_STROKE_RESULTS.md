# 拓扑发布审计与多来源 Masked Stroke Transformer 训练

2026-10-06。本轮先执行用户评估中的错误审计、同候选对照和受保护回放，再开发并完成独立新架构的 GPU 训练。RC0 未修改；真实双人审核仍为 0/25 + 0/25，独立 verified design=0，CoordinateConventionFrozen=false。

## PortPartitionNet 发布审计：仍为 NO_GO

固定 expanded-seed0，不选择错误更少的其他种子替代。原133例、自动668例均为开发队列，标签来自原始 wire-group 弱监督，不能解释为独立整图准确率。

| 同候选模块级对照 | 原133跨方向F1 | 原133错误合并 | 自动668跨方向F1 | 自动668错误合并 |
|---|---:|---:|---:|---:|
| 旧 Topology 保守策略 | 83.08% | 0 | 84.62% | 7 |
| PPN expanded-seed0 MAP | 85.90% | 26 | 96.95% | 48 |

旧模块使用相同候选 crop、原/重建 geometry、原图未遮罩 dot/T 上下文；这是模块级条件对照，不是完整 RC0 遮罩推理精度。旧 uncertain 只表示禁止合并，并非一个有监督的 split 类。

在旧模块 uncertain 子集中，PPN 的已知跨方向合并 precision：原队列 28/(28+26)=51.85%，自动队列 305/(305+46)=86.89%。因此不可直接把 uncertain 全交给 PPN 接管。

权重≥0.95 的 FP：原队列三种子为 8/1/0，交集0；自动队列为16/6/8，交集4。seed0原队列8条来自0053页两个无明显独立圆点的四臂交叉，具有明显种子敏感性。自动队列稳定错误中，0167的带红色圆点T交汇、0197的T交汇与 VDDA 文字可能含 source-group 冲突；只登记怀疑，不擅自改标签。FN示例同时包含无明显圆点绿色交叉和虚线交汇。像素观察来自代理，human_reviewed=false。

错误 crop、逐条位置、三种子权重及 FN 示例见 `output/port-partition-release-audit/same-candidate-audit.json`。示例按队列顺序选取，不作为错误类型占比统计。

## 65页受保护回放

基于缓存的 RC0 detector/OCR 遮罩，从原图构造所有候选；完成预测后才用 PCB 标签登记 known/unknown。unknown 不计准确或错误。GT10 缺乏本地 wire-group 真值，局部关系全部按 unknown 登记。

固定诊断门槛：marginal≥0.99、最优/次优 logit margin≥log(99)、entropy≤0.15；仅 old-uncertain；最多8条入射线段；禁止桥接两条已有输出 net。没有独立校准集，所以这只是未校准消融，不是可部署门槛。

| 队列 | 全部候选 | old-uncertain | 高margin合并提议 | 禁止已有net桥接 | 孤立wire补入既有net | ΔNetF1 | ΔNetLineF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT10 | 298 | 32 | 11 | 10 | 1 | 0 | −0.000006348 |
| PCB55 | 5387 | 1838 | 257 | 31 | 86 | 0 | −0.000358971 |

GT10改善0页/退化1页；PCB55改善1页/退化21页。高权重 unknown 合并关系分别491/6398，不把它们记为 FP。split-only 为 no-op：旧 uncertain 已禁止合并，旧 connected 受保护。merge-only 与 both 一致。

这个窄回放只补孤立 wire，保持 pin membership 不变，因此 NetHypergraphF1 不可能提升；不能据此推断全部拓扑学习无效。但 NetLineF1 已下降，且退化页多于改善页，本次回放不满足晋级要求。没有训练更多 PPN，没有修改冻结阈值。

## 新视角：Masked Stroke Transformer

不沿用四端口 CNN + pair 能量建模。将48×48图像分为36个8×8 patch，使用64维、2层、4头 Transformer，先遮盖60%的 token 重建灰度笔画，再均值池化直接预测15种合法电气分区。共86,927参数，输出天然满足等价关系传递性。

外部图像只提供视觉自监督，不从颜色或墨迹强行制造 net 标签。拓扑微调仍用已有3732训练样本；原133开发验证只用于选择 checkpoint，自动668只用于额外诊断。

| 实际预训练来源 | 去重后真实图像 | 图像块 |
|---|---:|---:|
| PCBnet 训练划分 | 237 | 3780 |
| AITEE 作者训练划分 | 662 | 10571 |
| CGHD 已有训练划分 | 17 | 79 |
| OpenSchematics 新增许可筛选样本 | 17 | 272 |
| 自生成102个训练布局族 | — | 1632 |
| 合计 | 933 | 16334 |

保留263个留出图像像素hash，排除跨来源重复；official-train200没有贡献新的独立像素，未重复计数。GT10和Freeze-A没有参加预训练。像素去重不等于设计家族独立。

OpenSchematics 检查了6个分散offset共60行，取得17张许可可追溯图像；没有下载或使用整个公开库。新来源：[数据卡](https://huggingface.co/datasets/bshada/open-schematics)，保留原项目license和出处于 intake.json。AITEE：[作者库](https://github.com/CKnievel/aitee-dataset)；CGHD：[作者库](https://github.com/DFKI/cghd)。CGHD归档CC BY-SA 3.0与元数据CC BY4.0差异保留在来源记录。本轮权重仅用于本地研究，不宣布具备比赛分发授权。AMSNet、CircuitOCR、OmniSch未因数量大就绕过许可/来源门禁；SchGen代码数据未渲染成图像。

## 已完成 GPU 训练及对照

现有 Windows Python3.13 / Torch2.6 CUDA11.8，RTX4060 Laptop。没有中断用户已有训练；未安装 Ubuntu Torch。预训练6轮，每轮来源均衡抽样4096块，共24576次抽样，允许重复，不是每轮遍历全部16334块。CGHD小样本会被重复抽样，不能因此宣称更多独立数据。

scratch 与 multisource-pretrained 固定同一 seed=20261006、同一训练数据/24轮微调，原133选择最佳checkpoint。完整训练约145秒，峰值allocated26.39MiB/reserved32MiB。已保存预训练checkpoint、两组最佳checkpoint、逐轮日志和两组ONNX。

| 模型 | 原133跨方向F1 | 原133MAP FP/FN | 自动668跨方向F1 | 自动668MAP FP/FN |
|---|---:|---:|---:|---:|
| Transformer scratch | 80.54% | 26 / 32 | 97.07% | 32 / 42 |
| Transformer 多来源预训练 | **83.69%** | **12 / 34** | **97.19%** | **29 / 42** |
| 冻结 PPN expanded-seed0 | 85.90% | 26 / 18 | 96.95% | 48 / 30 |

多来源预训练相对 scratch 原队列提升3.15个百分点，错误合并26→12，但略增加 FN；相对 PPN 并未提升原队列F1。仅一组种子，不能声称稳定泛化或比赛成绩提升。

固定描述性权重0.95时：原队列 scratch 94TP/0FP，预训练99TP/0FP，PPN107TP/8FP；自动队列为1192TP/3FP、1180TP/8FP、1201TP/16FP。预训练在自动队列高权重错误反而比 scratch 更多，所以不是所有工作点都更好。没有据此选择部署 threshold，也没有接入RC0或运行Freeze-A。

结论：新架构证明多来源笔画预训练有条件性收益，保留为独立研究候选。下一阶段应使用独立校准集和真实人工证据检验错误合并；当前主提交仍为冻结 RC0。

## 产物与复现

- 数据：`data/processed/masked-stroke/pretrain.npz`、`preparation.json`、`records.json`。
- 新来源：`data/raw/OpenSchematics-pretrain-sample/intake.json`。
- 训练：`scripts/train_masked_stroke_transformer.py`，设置 `.deps` 为 PYTHONPATH 后用现有 D:/Python313/python.exe 执行。
- 评估：`.venv/Scripts/python.exe scripts/evaluate_masked_stroke_transformer.py`。
- 结果：`output/training/masked-stroke-transformer/evaluation.json`。
- 模型：`models/candidates/masked-stroke-transformer/{scratch,multisource-pretrained}/model.onnx`。
- 164既有 unittest 通过；冻结清单53/89/215/102/39共498个条目均未改动。ONNX数值和遮罩隔离检查另存于训练目录。
