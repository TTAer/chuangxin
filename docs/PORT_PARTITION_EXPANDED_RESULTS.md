# PortPartitionNet数据扩充与复训（2026-10-06）

本轮按用户明确要求扩充开发数据并完成实际训练。网络架构不变；原PortPartitionNet实验文件和RC0/V1/R冻结文件均保持不变。没有使用Freeze-A、没有训练pin坐标定义、没有修改比赛入口或提交包。

## 扩充数据

在现有PCBnet图像SHA划分下，扫描296张去重开发图像，保留0237的既有排除。候选先由原图灰度笔画、线段和交点生成，裁剪像素也在读取注释前创建。没有用GT器件/文字遮罩生成新增候选；每页最多512个候选，确定性hash排序，不按GT挑候选。随后才读取wire-group paths产生部分关系监督。

本轮扩充的是现有真实图像中的自动候选样本及覆盖页数，不是下载新外部数据集。没有将缺失、歧义端口强行标为断开，也没有用合成增强数量充当新增真实来源。

| 数据 | 扩充前 | 扩充后 |
|---|---:|---:|
| 训练样本 | 953 | 3,732 |
| 含训练样本的图像SHA数 | 138 | 234 |
| 固定原开发验证样本 | 133 | 133 |
| 新自动候选开发诊断样本 | 0 | 668（53张图像） |

新增2,779个训练样本，训练量约3.92倍；总计4,533个训练/验证/诊断样本。107,310个扫描候选中，102,801个因监督不足或歧义跳过，781个因接近已有训练中心跳过，281个重复crop/target跳过，新增可监督样本3,447个。跳过候选不算识别失败或正确负例，不能由这个比例推算候选precision/recall。

完全相同训练裁剪的标签检查发现一组冲突（两个新增样本、两条关系）。将这些新增关系的监督mask置为未知，保留原始快照 `relations-before-conflict-resolution.npz` 和处理记录。原始1,086例的四个数组保持一致，训练/评估来源图像SHA交集为0，新增训练池与GT10精确图像SHA交集为0。尚未证明结构家族独立性或弱标注正确性，不能称为独立测试集。

## 实际复训与比较

沿用约34k参数的PortPartitionNet、partial-label likelihood和灰度增强，固定24轮、CPU训练，种子20261006/20261007/20261008。没有扩大网络、改变损失或扫描阈值；按原133例开发cross-axis F1选首次最佳checkpoint。扩充使每轮更新步数增加，因此这是正常扩充训练效果比较，不是等更新步数的纯数据量消融。668例不参与checkpoint选择，但结果查看后仍是开发诊断数据，不具备blind资格。

| 同一评估队列 | 原模型三种子F1均值 | 扩充训练三种子F1均值 |
|---|---:|---:|
| 原固定133例 | 0.8170 | **0.8640** |
| 新自动候选668例 | 0.9195 | **0.9732** |

| 种子 | 原133例新F1 | 原133例错误合并：旧→新 | 自动668例新F1 | 自动668例错误合并：旧→新 |
|---|---:|---:|---:|---:|
| 20261006 | 0.8590 | 36→26 | 0.9695 | 67→48 |
| 20261007 | 0.8593 | 47→2 | 0.9740 | 89→14 |
| 20261008 | 0.8737 | 26→13 | 0.9762 | 54→22 |

错误合并按弱标注的跨方向FP关系计数，不是错误net数或受污染页面数。原133例有424条已监督跨方向关系（152正/272负）；自动668例有1,552条（1,268正/284负）。自动诊断连接正例约81.70%，明显偏多，并且过滤了监督不足候选，所以不能将97.32%解释为自动整图准确率。新增候选尚未采用完整检测/OCR遮罩，与比赛真实pipeline还有差距。

## 保守权重诊断，不授予Commit

附加报告固定0.9/0.95关系边缘权重的合并precision、错误合并比例、召回和coverage，不以结果搜索提交操作点，权重也没有概率校准。默认候选仍固定seed0，不按结果挑最有利种子。

seed0在0.95权重下：

| 队列 | 合并precision | 已选合并TP/FP | 连接关系召回 | 已监督关系coverage |
|---|---:|---:|---:|---:|
| 原133例 | **93.04%** | 107/8 | 70.39% | 27.12% |
| 自动668例 | 98.69% | 1,201/16 | 94.72% | 78.41% |

默认seed0在原验证集仍未达到附件建议的>95%合并precision；不同队列表现差异明显，不能仅凭自动队列高分晋级。seed1/seed2的0.95权重原队列precision为99.10%/100%，但样本少且未经校准，也不能挑种子后当成可靠准入证明。尚未进行真实人工FP裁决、旧uncertain拓扑的整图回放或Freeze-A验证，状态仍为research/evidence-only。

## 产物与验证

- 扩充数据与来源记录：`data/processed/port-partition-expanded/`
- 三种子模型：`models/candidates/port-partition-expanded/expanded-seed{0,1,2}/port-graph/`
- 训练历史、checkpoint和新旧对比：`output/training/port-partition-expanded/`
- 完整评估：`output/training/port-partition-expanded/evaluation.json`
- 原图候选冒烟输出：`output/port-partition-expanded-proposals/pcb55-0005.json`

三个新模型的8样本ONNX与PyTorch核验最大logit绝对差≤5.73e-6。seed0原图入口完成118条几何线段、127个端口分区候选，commit=0，不生成比赛网表。164项既有测试通过；RC0/V1/R的53/89/215文件及上一轮PortPartitionNet研究清单的102文件SHA全部不变。

复现入口（已有数据和模型不覆盖，训练须使用新run-id）：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_port_partition_expanded.py
$env:PYTHONPATH=(Resolve-Path .deps).Path
& D:/Python313/python.exe scripts/train_port_partition_expanded.py --epochs 24 --device cpu --variants port-graph --seed 20261006 --run-id new-expanded-run
.\.venv\Scripts\python.exe scripts/evaluate_port_partition_expanded.py
```

准备脚本检测已存在数据时停止，以保留本轮扩充快照。评估入口固定比较本轮expanded-seed0/1/2和原matched-seed0/1/2；额外复跑的run-id需另立报告，不能覆盖此次结果。

本輪已完成数据扩充与三种子复训；未宣称比赛NetF1/PinF1提升。下一项关键工作是自动候选中的假合并审计和保守整图回放，而不是继续增大模型。人工Convention Pilot仍未完成，verified designs仍0，CoordinateConventionFrozen仍false。
