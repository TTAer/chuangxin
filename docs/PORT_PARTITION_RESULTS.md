# PortPartitionNet新架构实验（2026-10-06）

按用户本轮新增要求，重新启动隔离算法研究和实际模型训练。RC0运行基线及Convention Pilot审核协议继续冻结；算法暂停的旧约束在这个独立实验分支解除，双人审核、Freeze-A和坐标门禁没有解除。未读取Freeze-A进行训练或推理，未反推L1，未改比赛入口。

## 新的任务定义

先学习局部线段的连通分区，再考虑接入几何图/电气超图，不直接预测pin坐标或重排OCR token。该定义暂时避开未解决的body/tip/contact标注语义。每个局部交点有left/right/top/bottom四个端口，输出四端口的15种合法等价分区之一。分区天然保证连接关系对称和传递；“A连B、B连C、A不连C”的非法组合不能成为最终输出。

视觉部分用同一个CNN编码四条规范化方向的arm strip，另一个CNN编码局部整体图像；四个arm token通过两轮共享注意力消息传递，再由共享pair head给六条关系打分。对15个分区累计关系能量，用partial-label likelihood训练，MAP分区解码。训练阶段50%灰度增强，颜色仅作视觉特征。理论约束只保证输出关系形式合法，不保证电气事实正确。

这是新的局部拓扑系统原型，不是已经完成的第二套比赛解析器。当前仍依赖现有检测/OCR观察作为原图候选生成的输入；没有重新训练器件检测器、没有解析pin身份、没有替换飞线合并或产生比赛网表。它不会自动补pin或移动锚点。

## 实际数据与训练

从已有PCBnet开发数据的1,114个交点裁剪重新派生四端口关系监督。沿注释wire paths在距交点6px的四个方向查询wire-group归属，容差1.5px；缺失或多重归属端口的关系不监督，不能当作断开负例。少于3条可监督关系的28例排除，保留1,086例、5,706条可监督关系。

| 分组 | 局部样本 | 含样本的图像SHA数量 |
|---|---:|---:|
| 训练 | 953 | 138 |
| 开发验证 | 133 | 24 |

保留既有图像SHA分组，已检查精确重复不跨train/validation；结构设计家族独立性尚未证明。数据仍是弱wire-group注释，且裁剪候选来自旧拓扑训练流程（候选构建使用了注释器件/文字遮罩），因此是条件局部任务，不是全图自动候选召回率。GT10、Freeze-A和真实人工审核均未参与。

数据来源背景：[PCBnet作者论文](https://arxiv.org/abs/2608.27923)。本轮没有下载新数据；使用现有数据的新关系监督，避免污染待审核独立来源。没有把面向拥塞/DRC/IR-drop的CircuitNet当作原理图连通监督，见[其官方项目](https://github.com/circuitnet/CircuitNet)。拓扑保持损失是已有研究方向，见[clDice作者论文](https://campar.in.tum.de/pub/paetzold2021cldice/paetzold2021cldice.pdf)；本轮采用有限分区约束，不声称该思路为首次提出。

初始16轮普通整体CNN对照出现多数类退化，单次图模型跨方向F1为0.7613。因此补充更强对照：相同arm视觉编码器与pair head、去掉图消息传递/分区约束；另训练仅分区约束的消融。三组种子20261006/20261007/20261008，每组固定24轮，无阈值扫描。逐轮开发验证cross-axis F1选首次最佳checkpoint，所以以下是开发选择结果，不是独立测试。由于全局GPU显存已被占用约7.8GB，本轮实际使用CPU PyTorch训练，模型规模约34k存储参数，未中断其他GPU任务。独立pair和partition-only控制中保留但未使用的注意力参数不计为有效模型能力。

## 开发验证结果

| 模型 | seed0 F1 | seed1 F1 | seed2 F1 | 均值 |
|---|---:|---:|---:|---:|
| Arm视觉＋独立pair预测 | 0.4048 | 0.4075 | 0.3934 | 0.4019 |
| Arm视觉＋合法分区，无消息传递 | 0.6557 | 0.6344 | 0.6824 | 0.6575 |
| Arm视觉＋图消息传递＋合法分区 | 0.8025 | 0.8119 | 0.8366 | 0.8170 |

F1只统计已监督的跨方向关系（排除容易的left-right和top-bottom），开发分母424条，正例152条。默认原型固定使用seed0，并未挑F1最高的seed2。seed0图模型TP126、FP36、FN26；错误合并仍存在，不能提交或宣告部署准入。

独立pair对照在133例四端口潜在关系中分别有33/40/33例传递性冲突；分区模型均为0，这是结构性保证，不是识别准确率。该冲突计数包括潜在未知端口，仅衡量完整四端口输出形式，不能解释为真实电路错误数。部分监督关系全部正确的案例比例，图模型分别为84.21%、84.96%、77.44%；最高跨方向F1与最高完整案例准确率并不一致。

分区约束和消息传递两项都有开发收益信号，但数据少、来源弱、checkpoint由开发集选择，尚不能证明泛化优势，也没有与RC0整图PinF1/NetF1作等价比较。

## 自动原图入口与验证

```powershell
.\.venv\Scripts\python.exe scripts/prepare_port_partition.py
$env:PYTHONPATH=(Resolve-Path .deps).Path
& D:/Python313/python.exe scripts/train_port_partition.py --epochs 24 --device cpu --variants arm-pairs,port-graph --seed 20261006 --run-id new-run
.\.venv\Scripts\python.exe scripts/run_port_partition.py --image data/raw/PCBnet/Dataset300/0005/0005-kicad.png --observations output/designator-stable-catalog/pcb55/0005/observations.json --output output/port-partition-proposals/pcb55-0005.json
```

训练入口拒绝覆盖已有best.pt，复跑须换run-id。默认推理模型为matched-seed0/port-graph。原图入口只读取图像、已缓存的检测/OCR观察和模型，不读取GT；坐标明确从bottom-left观察转换为top-left一次。PCB55-0005冒烟测试生成118条几何线段、127个端口分区候选，全部evidence-only、commit=0。这些数量不是有效连接数或准确率；模型softmax权重没有概率校准。

11个实际训练ONNX导出均完成8样本batch的PyTorch/ONNX Runtime数值核验，最大绝对logit差≤1.15e-5。新增测试只覆盖非法传递关系拒绝和歧义wire-group不误作负例，全套164测试通过。RC0/V1/R冻结文件校验见本轮research-manifest.json。

实际权重：`models/candidates/port-partition-v1/`；训练记录：`output/training/port-partition-v1/`；数据：`data/processed/port-partition-v1/`；原图原型输出：`output/port-partition-proposals/pcb55-0005.json`。没有更新models/pipeline.json或比赛ZIP。

## 后续准入条件

下一步应检查FP36对应的实际像素失败、弱注释矛盾与自动候选生成误差，再决定是否扩大开发训练数据。只有自动候选条件下的局部拓扑收益、保守合并操作点及整图回放都得到证据，才考虑RC1 proposal。必须保留现有飞线语义合并与器件/pin先验约束，不能将局部物理wire-group预测当作完整电气超图。

独立比赛精度仍需实际双人审核和正式Freeze-A；本轮新模型训练不消除该缺口。审核文档按用户要求补充severity/scope和未来evaluator_hash，但当前A/B仍0/25，verified仍0，CoordinateConventionFrozen仍false。
