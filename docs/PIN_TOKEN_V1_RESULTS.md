# PinTokenNet V1 实际训练结果（2026-10-05）

结论：V1 已完成训练与 CPU ONNX 导出，但不接入 RC0。两个学习模型都低于最近端点基线，不能据此宣称整图 Pin/Net 涨分。冻结 RC0 的源码、选定权重、配置及镜像未修改。

## 数据与监督

使用官方 200 训练页中的固定 64 页：48 页训练、16 页开发验证。原图逐像素核对与 PCBnet CVJson 文字框一致，实际执行 RapidOCR，得到 2,252 个可唯一对齐且有明确文本引用的 token，训练 1,669 / 验证 583。没有替换 OCR，没有生成虚构 top-k 或字符置信度。

角色来源为 CVJson 原有文本和引脚引用，属于弱/含噪监督。多角色冲突、多引脚共用、OCR 对齐不唯一及未标注角色均排除；5,200 个 OCR token 因无唯一可监督对齐被排除。几何 shortlist 固定取最近八个端点，只有一个明确关联因不在 shortlist 而排除。没有使用顺序补号或模型伪标签。

现有角色样本为 designator 528、pin number 505、pin name 1,100、component model 119。net label 与 other 没有监督样本，因此虽然输出头预留六类，当前只能报告验证集中出现的四类。component model 的 F1 为零，是明确未解决的失败项。

训练/验证沿用已有训练重合分区，并额外按保守结构指纹分组，所有带原验证页的结构组都归验证。组间不交叉，但指纹不是原始设计同构证明，也不能把该验证集当 Freeze-A。未读取 10GT、Freeze-A 或 Freeze-B 进行训练和选模。

## 两组真实训练

共用 token 文本语法特征、几何候选与 null 类；使用相同种子、15 个 epoch、开发验证选择 checkpoint。V1 的视觉输入为 token 的 32×96 局部字符裁剪；不是完整四边条带模型。

| 指标 | 几何+文字 MLP | 视觉+文字联合模型 |
|---|---:|---:|
| 验证角色 macro F1（出现的四类） | 0.655861 | 0.653812 |
| 全 token 关联准确率（含 null） | 0.897084 | 0.895369 |
| pin token 关联准确率 | 0.881215 | 0.878453 |
| null 准确率 | 0.923077 | 0.923077 |
| 最近端点错的 24 个难例恢复 | 2 | 1 |
| 最近端点对的 338 个易例破坏 | 21 | 21 |
| 221 个 null token 被错误绑定 | 17 | 17 |

最近端点在同一 362 个 pin token 上命中 338 个，准确率 **0.933702**，优于两组模型。该基线只评价已知 pin token 的关联，不能拿它与包含 null 的全 token 准确率直接比较。

两个模型的身体框与端点集合来自标注，仅 token 来自实际 OCR。因此这些数字是条件局部验证，不能当作预测端点上的身份恢复率、正式角色 GT 盲测或比赛 PinF1。身份恢复率及整图 Pin/Net F1 均保持未评，不写成零或虚构收益。

## 导出与决定

两个 ONNX 均在 CPU 上测试动态 batch 1 和 7，argmax 与 PyTorch 相同，最大 logit 误差分别为 9.54e-7 与 4.29e-6。模型位于 `models/candidates/pin-token-v1/`，没有写入 `pipeline.json`。

当前决策为 `STOP_V1_integration_pending_better_local_association_and_reviewed_labels`。没有必要把已低于局部基线的 V1 接进整图，再扩大模型或反复扫描超参。下次可研究保留可靠几何锚点的残差关联、包含 IC 边上下文的 token 编码，以及经审核的 model/net/other 难负例；先解决本轮破坏易例和型号角色失败的问题，再做用户规定的整图 PinF1 +0.01 与 NetF1 同步提升门槛验证。

## 复核文件

- 数据统计与固定页清单：`data/processed/pin-token-v1/preparation.json`
- 两组 checkpoint、历史和验证预测：`output/training/pin-token-v1/`
- 易/难例审计：`output/ablation/pin-token-v1.json`
- CPU 导出一致性：`output/ablation/pin-token-v1-export.json`
- 数据准备、训练、评估、导出：`scripts/*pin_token_v1.py`
- 实验模型定义：`experiments/pin_token_v1.py`，不由主线导入。

同域独立 Freeze-A 的实际采集状态另见 `docs/FREEZE_A_PROGRESS.md`。在设计排除与真值审核完成之前，不改变 RC0 的 NO_GO 晋级决定。
