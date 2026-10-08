# 真实网络标注的语义与坐标准入

2026-10-07。本轮完成固定来源读取、监督口径修正和4页自动对应诊断；**没有新增模型训练、独立评分或人工verified design**。

## 新的实质证据

取得[作者固定commit](https://github.com/boscochanam/circuit-digitization/tree/4e5c2ec800601c621823a1f90afe8bd24ee3067c)的完整数据说明、31页上游路径映射、组件框元信息、笔画/照片模态表、网络评分实现及许可。只读取资料，不执行作者代码。源仓库保存于`data/raw/CircuitDigitization-source`；此前ProbeFlow归档保持不变。

作者明确其评分为“组件对是否共享至少一个net”，并非pin-net完整分区。[评分实现](https://github.com/boscochanam/circuit-digitization/blob/4e5c2ec800601c621823a1f90afe8bd24ee3067c/wire_detection/benchmark/join_eval_real_f1.py)逐net取组件两两组合；没有把组件作为导线节点做传递闭包。

31页转为**484个正组件对、410个负组件对**，保留原net关联，不虚构pin序号。25/31页存在非传递三元组：A与B共享某net、B与C共享另一net，A与C却不共享net。B可以是有两个独立端口的电阻，直接union这些关系会短路器件。一个明确fixture验证保留(0,1)/(1,2)时不制造(0,2)。

这里的“负组件对”是该作者electrical_idxs评分范围内没有共同net的关系，不代表实际电流无法流经器件，也不保证所有隐藏端口都已覆盖。

## 本地原图候选

在160条现有CGHD-v16图像记录中找到4个同名候选。对作者组件中心与本地实例多边形中心做8种坐标轴交换/翻转的指派诊断，未用网络真值选变换或映射；本轮探索结果用于数据对应，不是盲测分数。

| 页面 | 电气组件对应 | 平均归一化中心距离 | 类别对应 |
| --- | --- | --- | --- |
| C15_D2_P2 | 7/7 | 0.003612 | 7/7 |
| C242_D1_P1 | 10/10 | 0.000661 | 10/10 |
| C2_D2_P1 | 10/10 | 0.001311 | 10/10 |
| C37_D2_P4 | 11/11 | 0.000827 | 11/11 |

四页最佳坐标提案均为不交换、不翻转的归一化坐标；类别用显式别名表检查。位置距离单位为归一化坐标欧氏距离，**不是官方pin像素误差指标**。此项合计38/38电气组件对应，但仍是自动提案，没有晋级verified。

作者模态表表明这四页基准输入都为二值stroke map；因此观察候选选用本地`segmentation`图，而不是拍摄照片。使用源数据笔画图不是模型输入真值net，但有较强预处理Oracle性质，未来必须单独报告照片队列和stroke-map队列，不能混称真实照片效果。

![本地笔画图上的组件中心对应提案](../output/component-incidence-20261007/coordinate-proposals.png)

已实际查看生成图：标记位于对应器件附近，未见明显全页错位。机器查看不计为人类审核；图中叠加ID仅用于对应核查，原始输入路径另存，训练不能读这张提案图。作者net_gt_ui_overlays带真值导线，不作训练输入。

## 准入与下一实验定义

当前training_admitted=0。剩余缺口是原始704基准副本的像素/预处理身份、实例一一对应证据、source许可与家族重叠，尤其尚无端口身份。作者上游路径映射仅证明文件名和drafter归属，作者也明确它不能证明预处理副本字节一致。

这个来源未来适合监督“组件对共享net”或组件—网络关联超图，**不能直接接到ProbeFlow的wire-run分区loss**。正确路径有两个：取得可靠端口/导线对应后做物理分区监督；或另登记允许组件参与多个net的关联预测任务。二者应分开，避免通过电阻等器件的连接路径被误认为同一net。

复现脚本`scripts/prepare_component_incidence.py`；逐页目标`output/component-incidence-20261007/targets.json`；自动对应与类别检查`audit.json`；本轮独立冻结清单`manifest.json`。新资料不进入RC0、Convention Pilot、Freeze-A，不改变原始review_hash。人工A/B仍0/25，独立verified designs仍0，官方evaluator等价性仍unresolved。
