# NetSetQuery Oracle Ladder：完整监督第一关结果

日期：2026-10-06。结论：**当前 NetSetQuery v1 未通过完整监督的可学习性门槛，停止后续覆盖率、真实 observation 和弱标签训练。保持研究归档，不接入 RC0。**

这轮按用户评估先检查完整网络分区，而没有训练旧版17.14%兼容标签。完成了完整 pin-set 与 pin+wire 联合两组 GPU 实验，各24轮。第一关失败后执行条件停止；四项建议实验并未全部执行。

## 数据及 Oracle 边界

从既有官方 TRAIN200 的原始标注筛选，保持原图像训练/开发划分，不用 GT10 或 Freeze-A 训练。实际49训练页、14开发验证页；111页结构校验失败、25页 pin 归网不完整、1页超过128 query 被排除。没有自动修复、补网或把几何推测写成真值。

准入条件是：结构可解析，每个 pin 恰有一个 GT net owner，net 数≤128，pin+wire token≤512。每条 wire 的 owner 直接取原始 GT 的 net.edges 归属。63页源标注 hash 核查通过，GT10像素重合和准入页内像素重复过滤未发现额外排除。设计家族独立性未证明，原始标注也没有获得真实双人核验。

**这里是完整监督的几何 Oracle，不是完整语义信息 Oracle。** GT component/pin identity 用于对齐目标，GT pin/body/wire 坐标用于构造输入；GT net ID、membership、wire owner 不进入特征。输入仍是冻结 v1 的12维几何/粗类型 token，无文字、label token 或唯一身份语义编码，也不读取原图。pin/wire 输入没有顺序位置编码。由此不能把失败归结为“任何完整语义 Oracle 都无法学习”。

512 token 是本轮显式资源边界，不是比赛容量承诺；本次没有页面因该边界被排除。被筛除的页面不能在最终系统中静默遗漏。

## 容量审计

| 范围 | 页数 | net 最大值 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: |
| 所有源标注（包含无效结构） | 200 | 206 | 129.10 | 196.10 |
| 仅通过结构校验 | 89 | 206 | 91.20 | 141.76 |
| 本轮最终准入 | 63 | 92 | — | — |

因此128 query不是完整数据覆盖保证。本轮维持128，不盲目扩容；0106-jlc 显式因 query overflow 退出。结构有效不等于人工核验正确。

## 固定训练协议

冻结 v1 架构：189,186参数，128 query，width64，2层编码器+2层解码器。两组从相同新种子20261007独立初始化，AdamW学习率0.001，weight decay0.001，梯度裁剪1，整页batch1，每组24轮。共2,352次 optimizer step。

监督采用 Hungarian 对齐后的 pin CE+Dice；联合组增加 wire CE+Dice，两头共同匹配 GT net。完整 GT 为匹配 query 提供正存在监督，并对其余 query 提供负存在监督，BCE权重0.2。没有追加防坍缩正则或调 query 数。

训练前保存 `output/training/netset-oracle/protocol.json`：开发页均值 pin membership micro F1≥0.80、pairwise F1≥0.70，并超过两种简单参考的较高均值0.05，才允许后续梯度覆盖率/真实观察训练。这是本地可行性门槛，不是官方验收阈值。最佳检查点按开发 pin micro F1 选择，所以14页不是独立盲测。

RTX4060 Laptop CUDA 两组训练共312.94秒；Torch实际峰值分配38.81 MiB，峰值缓存56 MiB。没有停止其他用户进程。只保存研究检查点，没有导出比赛模型或更新提交包。

## 实测结果

所有指标为逐页计算再取平均，表中为百分数。最佳开发检查点分别是第16、23轮。

| 指标 | pin-set Oracle | pin+wire Oracle |
| --- | ---: | ---: |
| 训练 pin membership micro F1 | 39.16 | 37.41 |
| 开发 pin membership micro F1 | 33.11 | 32.22 |
| 开发 Hungarian net Dice（以所有真实 net 为分母） | 18.97 | 17.99 |
| 开发 exact net-set F1 | 0.75 | 0.77 |
| 开发 pairwise connected F1 | 17.27 | 16.97 |
| 开发 wire ownership | 未监督，不作结论 | 21.44 |
| 第一关 | FAIL | FAIL |

成员 micro F1 的 net 匹配先最大化逐 net Dice，再统计匹配集合的 pin 交集。exact 指完全相同成员集合。Hungarian net Dice 将未匹配真实 net 计0。wire ownership 是 GT 线段在 pin 匹配网络下的 owner 分类正确率，**不是端点匹配 NetLineF1**。它不衡量生成线段的几何正确性。

每个 pin 各成一网的抽象参考 micro F1 均值44.83%；全部 pin 成一网参考12.62%。其中3个开发页 pin 数超过128，所以 singleton 是不受 query 容量约束的参考，不能称为同容量模型公平对照。不论该参考，本轮两个模型也远低于预注册0.80/0.70门槛。

## 欠分区与重复 query 诊断

| 开发页均值 | pin-set Oracle | pin+wire Oracle |
| --- | ---: | ---: |
| 真实 net 数 | 34.07 | 34.07 |
| 活跃硬 pin query | 9.07 | 9.07 |
| existence≥0.5 的 query | 21.21 | 31.79 |
| 完全匹配 net 数 | 0.071 | 0.143 |
| 部分交集匹配 net 数（不完全相等） | 9.000 | 8.786 |
| 未匹配或零交集真实 net 数 | 25.000 | 25.143 |
| membership probability≥0.1 的活跃软集合 | 12.86 | 11.21 |
| 阈值软集合重复数 | 0.071 | 0.357 |
| 全 query 成员概率向量平均余弦 | 0.480 | 0.613 |

硬集合重复为0是每个 pin 独占一个 argmax 类的构造结果，不能用来证明不存在 query 冗余。0.1软集合阈值只作固定诊断，没有用来筛选比赛输出。联合监督使 existence 数更接近真实网络数，却没有增加实际 pin 分区数或提高成员恢复。

训练集本身也未充分拟合。因此当前结果更支持“此输入表示/解码/优化组合尚未学会分区”，而非单纯“开发域泛化失败”或“17.14%标签覆盖率不足”。尚不能在本轮区分表示不足、优化困难及标注噪声各自贡献；不能用失败直接否定所有集合预测架构。

## 物理连接与逻辑网络的证据限制

准入标注中1,623个带线段 GT net，有155个按**精确线段端点相等**计算出多个分组。该诊断不处理线段中途接入、交叉电气语义、符号端口或 label。因此它不是可靠物理连通真值，也不能证明这些网都是飞线/标签网。

这提醒下一版必须先明确 physical group 与 logical net 的监督和可观测语义，而不能把纯几何分组当 wire owner。当前输入没有 label/text，遇到依赖隐式标签的网络可能缺乏辨识信息。本轮只记录限制，没有使用这些诊断修改原 GT、生成伪 owner 或并网。

## 条件停止与归档

| 建议实验 | 实际状态 |
| --- | --- |
| Full pin-set Oracle | 已完成，FAIL |
| Pin+wire joint Oracle | 已完成，FAIL |
| 100/75/50/30/20/10%覆盖率曲线 | 第一关失败，未执行；不伪造曲线 |
| 冻结真实 observation+GT对齐 Oracle | 第一关失败，未执行 |
| 原弱标签训练 | 未执行 |

不进行 GT10/PCB55 接入回放；本轮未产生 ΔNetF1 或 ΔNetLine，不能声称比赛收益。当前分支冻结为 `NO_GO_FULL_ORACLE`。

后续若重新授权研发，先补足可区分的物理/标签输入并做小规模可拟合性检查，再重新登记独立实验协议；不要直接扩数据、延长弱标签训练或调阈值。本轮不启动这些新变量。

## 核验与产物

标签重编号、query置换、unknown token扰动的 loss差均0；unknown token梯度0；容量超限显式拒绝。这些仅验证损失实现性质，不是部分监督训练结果。训练全过程 loss 有限。

原 RC0 53条、NetSetQuery原型203条、Agreement351条 SHA 全部不变；RC0 ZIP SHA仍为 `04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4`。Reviewer A/B仍0/25，verified designs仍0，坐标 convention未冻结，官方 evaluator equivalence仍 unresolved。

复现入口：

- `scripts/prepare_netset_oracle.py`：原数据准入和训练张量；已有产物时拒绝覆盖。
- `scripts/train_netset_oracle.py`：固定GPU训练；已有协议时拒绝覆盖。
- `scripts/evaluate_netset_oracle.py`：最佳检查点 CPU 训练/开发复核与逐页 logits缓存。
- `scripts/audit_netset_oracle.py`：容量、源标注/冻结hash核验和 exact/partial/net Dice。
- `scripts/check_netset_oracle_loss.py`：损失性质检查，不更新模型。

主要结果位于 `output/training/netset-oracle/`；训练数据位于 `data/processed/netset-oracle/`。归档清单 `output/netset-oracle-research-manifest.json` 保存代码、数据、检查点和结果 SHA。evaluator及评分协议hash仅作为本轮开发实验的溯源，不填作首次独立评分证据。
