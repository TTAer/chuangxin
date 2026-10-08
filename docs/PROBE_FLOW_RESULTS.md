# ProbeFlowNet：视觉通导率与主动探针响应

2026-10-06。状态：**完成GPU训练；受控合成可拟合性成立；真实整页有效性未验证；不接入RC0**。

用户授权从新视角重新研发。本轮将网络归属问题改写为“在已观察线段图上注入探针信号，哪些节点产生同组响应”。它借鉴图上的扩散与电阻网络直觉，避免让固定数量的net query直接分摊整页成员。此处是物理连通模型，通导率是学习得到的无量纲参数，不代表元器件真实电阻；不能把通过电阻相连的不同电气net合并。

## 架构与训练目标

灰度交点crop → CNN → 候选连接通导率 → 对称邻接A → L=diag(A1)-A → G=(L+0.003I)^(-1)。归一化R_ij=G_ij/sqrt(G_ii G_jj)，以0.5阈值和连通分量输出物理线段组。损失为分别归一化的正/负全局可达关系BCE，加0.2倍可观察交点BCE。训练不需要net槽位或Hungarian；报告的成员匹配仍使用Hungarian，这是评测口径而非学习机制。

模型16,137参数（包括图像对比预训练投影头）。原型固定最多12个已观察wire run、36个候选交点，每个crop为24×24。输入提供精确线段与交点索引，属于**几何Oracle**。尚无自动检测、OCR、pin身份/角色、逻辑飞线标签或跨页语义。与既有双图方向相容的后续设计是先得到physical groups，再用实际可观察label/电源类连接logical nets；当前未实现该层，RC0原有飞线逻辑保持冻结。

相关图学习思想已有先例：[End-To-End Learned Random Walker](https://arxiv.org/abs/1905.09045)。本轮是探索性工程架构，未证明学术首创。

## 第一版失败与R2修正

V1用sigmoid(logit)^4，所有有效候选边始终有非零通导率。4页tiny训练首次在step120通过门槛，最终完全拟合；然而两组主训练模型在旋转加模糊的128布局stress上都误合为一组。开发集100%不能掩盖这一失败。泄漏累积是结构假说，没有做单因素因果确认。

R2平均4个90°方向的CNN特征，再令g=ReLU(logit)^2/(1+ReLU(logit)^2)，允许精确零通导率。训练加入亮度、噪声及50%概率3×3平均池化模糊。**protocol文本把训练模糊写成Gaussian sigma0.7，但实际代码使用平均池化；Gaussian sigma0.7用于两个扰动评估集。以归档代码为准，不修改事前protocol掩盖差异。** R2同时改变对称性、通导率和增强，收益不能归因于其中一项。

## 数据与真实执行

| 数据 | 实际取得/使用 | 边界 |
| --- | --- | --- |
| 自建可控编译数据 | tiny4、train1024、validation128、旧stress128，另新留出128布局 | 连接点/桥跨由编译器直接生成真值；不是独立真实设计；随机seed分离不证明图同构或家族分离 |
| [CI2N作者数据](https://github.com/LAD021/ci2n_datasets) | 作者train目录64张不同像素图，抽取1024灰度patch；实际6轮对比预训练 | 图像自监督，未用OBB/几何规则制造电气监督；裁剪数量不等于独立设计数 |
| [Circuit Digitization作者资料](https://github.com/boscochanam/circuit-digitization) | 实际下载31条作者人工核验net记录、257条派生参考及许可；隔离，未训练 | 257不等于README中的164 heldout子集；作者核验不等于本项目verified design |
| CircuitSense作者HF | README请求502、API请求超时 | 未取得，未训练，不能计入数据规模 |

CI2N固定commit `2ffd41b2255726eb1be215fa7ba1d9808c80c808`；Git树19,060项是文件项数，不是训练图数。库Apache-2.0；原图来源权益仍需核对，当前不对外重分发。已排除已知GT10/PCB开发图的精确像素重复，但不能声称排除家族重叠。

Circuit Digitization固定commit `4e5c2ec800601c621823a1f90afe8bd24ee3067c`。作者人工net标注MIT，CGHD1152派生数据CC BY 4.0，CGHDv12派生参考CC BY-SA 4.0，细则与原作者归属见本地ground_truth/LICENSE。v12参考由stroke map、symbol擦除和连通规则生成，不能当人工真值。现有本地图像版本不同，原图对齐、角色映射与家族排除完成前不进入训练或独立评分。

执行硬件RTX4060 Laptop，Torch2.6 CUDA。V1 tiny400步、CI2N预训练6轮、两组主训练各16轮，共2544个optimizer steps，用时183.08秒，峰值分配193.92MiB。R2从本轮CI2N图像预训练权重初始化，未载入V1拓扑微调权重；12轮×128步=1536 optimizer steps，用时316.39秒，峰值272.23MiB。完整日志与checkpoint已保留。

## 结果与对照

| 模型/对照 | 队列 | 成员F1 | 完整分区正确率 |
| --- | --- | --- | --- |
| V1 scratch | 旧stress128 | 28.06% | 0% |
| V1 CI2N预训练 | 旧stress128 | 28.06% | 0% |
| R2 | train1024 / validation128 | 100% / 100% | 100% / 100% |
| R2 | 已观察旧stress128，仅开发诊断 | 100% | 100% |
| R2 | 新留出128 | 100% | 100% |
| 全部交点相连 | 同一新留出128 | 31.67% | 0% |
| 全部交点断开 | 同一新留出128 | 72.84% | 10.16% |
| R2同一学习交点head + 简单连通分量 | 同一新留出128 | 100% | 100% |

新留出seeds40000:40128，数据hash在R2训练前登记；checkpoint仅按validation成员F1、平局按可达关系BCE选取，选择epoch12。新留出在训练完成后评分，后续CPU核验重复计算用于实现一致性，不参与选模。旧stress已看过，明确降为开发诊断。新留出仍与训练同一生成器，不是外部盲测。

**简单连通分量同样满分：本轮尚不能证明全局传播提供额外收益。** 该对照使用同一全局训练head，不是单独训练的局部模型，也不是global-loss消融；它只能限定解码机制结论。CI2N预训练在V1扰动集没有收益；R2未做scratch对照，不能将其满分归功于真实图像预训练。

旧日志macro pair F1在没有正配对的页面记0，即使分区完全正确。新留出13页全singleton，导致macro为89.84375%；正确解释是 pooled raw pair TP=560、FP=0、FN=0，micro F1=100%，128/128完整分区正确。原始日志不改写，audit单独解释这个口径。

旋转响应最大差0；节点置换后响应最大差2.98e-7；候选重排最大差0；反向梯度有限。RC0 53项、NetSet Oracle219项、Agreement351项SHA及原25例审核协议锁通过。没有产生人工审核或verified design。

## 复现与交付

- 编译：`scripts/prepare_probe_flow.py`；V1训练：`scripts/train_probe_flow.py`；R2：`scripts/train_probe_flow_sparse.py`。已有protocol时训练脚本拒绝覆盖；新复训应另登记实验输出。
- 核验：`scripts/audit_probe_flow.py`、`scripts/control_probe_flow.py`。
- 独立研究推理：`scripts/predict_probe_flow.py INPUT.npz --output OUTPUT.json`。只读patches/edges/edge_mask/valid，不读truth；拒绝超容量或错误图输入。输出是物理wire-run组和探针响应，不是比赛pin/net JSON。
- R2权重：`output/training/probe-flow-sparse/best.pt`；报告：`summary.json`、`audit.json`、`controls.json`；数据来源：两个raw目录下intake.json。
- 完整冻结清单：`output/probe-flow-research-manifest.json`。

下一证据门槛是用来源配对的真实可见连通监督检验是否跨域，以及与单独训练的局部CNN+CC比较；在此之前只保留研究候选，不扩大RC0变更或声称比赛精度。人工A/B仍0/25、verified designs=0、官方evaluator等价性unresolved。

## 2026-10-07：真实资料准入核查

运行`audit_connectivity_intake.py`：本地CGHD-v16有160条图像记录，31条作者网络标注中仅4条存在同名候选，全部仍未通过像素/实例身份对应。作者说明704×704副本含非等比例缩放、翻转旋转和二值stroke map，不能仅凭文件名或尺寸配对。

更关键的是31/31记录均有同一`(component_id, "e")`出现在多个net。这是**组件参与网络的关联结构**，并非唯一可观察pin的分区标签；例如电阻的两个端口分别属于不同网络。将这些组件ID直接union会短路电阻并错误合并网络。本资料可在正确配对后用于component-pair指标或组件—网络超图任务，尚不能直接监督ProbeFlow的物理wire-run分区。保持training_admissible=0；没有伪造端口编号或训练标签。逐页结果见`output/probeflow-data-intake/connectivity-admission-audit.json`。
