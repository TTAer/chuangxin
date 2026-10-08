# PPN + MST Agreement Gate：局部互补成立，当前自动接入 NO_GO

2026-10-06。按最新用户评估完成错误交集、三模型分歧矩阵、固定工作点、unknown暴露和三个整图消融。本轮没有训练、扩数据、改变模型/预训练配置、扫描阈值或选择新默认seed。RC0、PPN、MST及人工审核协议保持冻结。

## 1. 错误交集

固定 PPN expanded-seed0 + MST multisource-pretrained。每个候选只统计4条跨方向关系；valid=false不进入准确率。15类分区MAP与marginal高权重工作点分别分析。

| 开发队列 | PPN MAP FP | MST MAP FP | FP交集/并集 | PPN/MST FN | FN交集/并集 |
|---|---:|---:|---:|---:|---:|
| 固定133 | 26 | 12 | **0 / 38** | 18 / 34 | 16 / 36 |
| 自动668 | 48 | 29 | **9 / 68** | 30 / 42 | 24 / 48 |

原队列FP互补明显；FN互补有限。自动队列9条共有MAP错误集中在5个候选：0081的无独立圆点黑色四臂交叉、0132的相邻绿色折线、0159局部交汇、0167带红点T。0167仍是 source-wire-group 与像素证据冲突疑点，不能据此自动改弱标签。像素笔记均为代理观察，human_reviewed=false。

## 2. 固定0.95工作点：一致只能获得Proposal

门槛来自上一轮报告，不扫描、不校准。逐关系要求 PPN marginal≥0.95 且 MST marginal≥0.95；两者不一致则abstain。旧RC0 confident保留；旧uncertain才有新增Proposal机会。

| 队列/适用域 | PPN TP/FP | MST TP/FP | Joint TP/FP | Joint precision | Joint recall |
|---|---:|---:|---:|---:|---:|
| 固定133全部已知跨方向关系 | 107 / 8 | 99 / 0 | **91 / 0** | **100%** | 59.87% |
| 自动668全部已知跨方向关系 | 1201 / 16 | 1180 / 8 | **1165 / 4** | **99.66%** | 91.88% |
| 固定133旧uncertain子集 | — | — | **10 / 0** | 100% | 22.73% |
| 自动668旧uncertain子集 | — | — | **272 / 2** | 99.27% | 81.68% |

高权重FP交集：固定133为0、自动668为4。Joint precision优于单独PPN，但不是“两个模型独立”。自动队列仍有共享错误。

样本量限制：91/91的一侧95%二项下界约96.76%；实际旧uncertain适用域10/10的下界仅74.11%。关系在同一crop/页面内相关，因此这些二项诊断还不能作为设计级独立置信保证。0 FP不等于已证明99%真实精度。

冻结的另外两个PPN种子也参与读取性稳定性检查，没有新增训练或改变默认seed：固定133的Joint均0FP，TP为93/85；自动668为1164TP/3FP与1092TP/2FP。MST只有一个seed，不能宣称MST多种子稳定。两个开发队列都主要来自PCBnet，不构成跨来源独立验证。

## 3. 三模型完整分歧矩阵

`output/topology-agreement/disagreement-matrix.md` 保存固定133与自动668的27种 C/S/U组合；JSON另含MAP矩阵。每行记录关系总数、known-positive、known-negative、unknown。RC0的U不是有监督split类；不能把“旧模块禁止合并”和“确认不连接”混为一谈。

模块级旧Topology的同候选对照沿用上一轮缓存geometry/未遮罩dot/T上下文，与完整RC0遮罩入口有区别。自动整图暴露及回放采用实际RC0缓存候选和遮罩，分别报告，不能把模块级数值直接当整图准确率。

## 4. Unknown Agreement Exposure

从原图候选构造，不按GT过滤；读取前一轮固定候选预测后对相同图像位置运行冻结MST。GT10局部wire-group真值缺失，全部按unknown；PCB55在推理后登记可观察关系。unknown既不是TP也不是FP。

| 队列 | 候选数 | unknown关系总数 | PPN high unknown | MST high unknown | Joint high unknown | Joint unknown候选 | 旧uncertain中的Joint unknown关系 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT10 | 298 | 1192 | 956 | 1144 | **946** | 279 | 92 |
| PCB55 | 5387 | 18251 | 9463 | 10349 | **7226** | 2764 | 1178 |

Agreement减少了暴露，但仍保留PCB55中76.36%的PPN高权重unknown关系。不能把“一致”当作安全证明。没有预登记可接受暴露上限，也没有独立校准集；不事后挑选上限来宣布Go。仅局部描述性precision门槛通过。

## 5. 隔离整图回放：重新构建电气图

三个模式都在独立output目录执行，不写回任何RC0预测。使用冻结parse_graph与缓存observations重建；恢复运行时在图构建后调整的component显示bbox，并比较排序后的hyperGraph与全部物理线段。65/65页准确复现原网络/线段，pin保持不变；net/edge标识符不参与等价比较。没有更换pin/OCR模型。

保留labels和flying-label语义合并；禁止新增连接桥接两条已有net。候选最多8条入射wire。

- **split-only**：旧connected的probability低于冻结神经connect门槛；两模型强烈支持合法LR/TB两轴分开；无可见dot；整个32×32证据crop与已检测symbol/OCR区域分离。撤销局部junction/contact，重建网络；检查输出只能细化旧分区，不能引入其他合并或丢失pin覆盖。
- **merge-only**：旧uncertain；两模型在全部6条关系上均≥0.95，支持完整合法连接分区。局部和全局禁止桥接已有net。逐关系Proposal不能随意扩成四臂全连接，因此此回放权限比局部统计更严格。
- **both**：同时应用上述动作。

只有0208页[669.5,194.5]被旧模块connected而两模型强烈支持canonical split。旧probability=0.5129，不是高置信神经connected；但crop与OCR区域重叠，原图包含连接器边线/编号，因此安全门禁拒绝。这是unknown关系，没有人工真值。split-only没有实际拆网，不能宣称验证了拆分收益或拆分算法无效。

GT10 merge应用2次，21次已有net桥接被拒绝；PCB55应用90次、157次重建无实际变化、32次桥接被拒绝、1次超过入射线段限制。所有accepted操作commit=false。

| 队列 | 模式 | ΔNetHypergraphF1 | ΔNetLineF1 | 任一拓扑指标改善/退化页 | 系统门槛 |
|---|---|---:|---:|---:|---|
| GT10 | split-only | 0 | 0 | 0 / 0 | 未通过 |
| GT10 | merge-only | 0 | **−0.000121039** | 0 / 2 | 未通过 |
| GT10 | both | 0 | −0.000121039 | 0 / 2 | 未通过 |
| PCB55 | split-only | 0 | 0 | 0 / 0 | 未通过 |
| PCB55 | merge-only | 0 | **+0.001645927** | **4 / 19** | 未通过 |
| PCB55 | both | 0 | +0.001645927 | 4 / 19 | 未通过 |

这是0–1 F1单位，+0.001646约为+0.1646个百分点，不能写成比赛综合分提升。PCB55均值上涨由少数改善页贡献，退化页更多；不满足 ΔNetF1>0、ΔNetLineF1≥0、regressPages<improvePages。GT10/PCB55继续分别使用冻结的Q&A proxy与PCB开发scorer。

此次代码允许重建成员关系，但最终没有通过门禁的split，merge又保护已有net，实际pin membership仍未改变。结论只覆盖当前受保护动作的发布分支，不能否定所有拓扑表示或未来经过人类验证的拆网动作。

## 6. 当前决定及项目状态

**Agreement自动接入保持NO_GO；冻结本条训练/接入研发路线。** PPN用于结构消融基线；MST用于研究Evidence/审核参考；不赋予自主Commit权限。RC0继续作为工程提交基线，没有独立比赛成绩声明。

真实Reviewer A=0/25、B=0/25，VerifiedDesigns=0，CoordinateConventionFrozen=false。算法不能填补真实双人审核。后续主线回到已有Convention Pilot、Directional Pad来源/内容/几何/家族门禁及最终提交稳定性；不增加新模型或扩大训练数据来绕过这些缺口。

167项unittest通过，其中新增3项检验已有net桥接拦截、拆分不丢pin/不合并无关net以及忽略net标识符的等价比较。RC0及所有历史研究冻结清单另做SHA校验。

## 7. 复现与证据

```powershell
.venv/Scripts/python.exe scripts/audit_topology_agreement.py
.venv/Scripts/python.exe scripts/summarize_topology_agreement.py
.venv/Scripts/python.exe scripts/replay_topology_agreement.py
.venv/Scripts/python.exe -m unittest discover -s tests -q
```

`local-audit.json`/`local-predictions.npz`为关系级可追溯输出；`unknown-exposure.json`为全部候选暴露；`existing-seed-sensitivity.json`为冻结种子诊断；`output/topology-agreement-replay/summary.json`保存逐页及三个模式结果；每页另存输出JSON与拒绝/接受原因。模型、训练数据及旧报告SHA不变。
