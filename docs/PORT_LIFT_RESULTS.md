# PortLiftNet V1：真实训练结果

2026-10-07。状态：GPU训练与控制评估完成，研究原型；未接RC0，未宣称比赛分数或独立真实准确率。

新架构将物理导线路径和多端口组件关联分开，用最大路径乘积代替拉普拉斯求逆，用端口汇总代替组件关系传递闭包。详见[架构与事前协议](PORT_LIFT_ARCHITECTURE.md)。组件可以同时参与多个net，器件内部不添加导线边；外部短接仍允许同器件不同端口同网。

4页tiny240步，第200步首次同时达到组件与导线F1/完整关系100%。随后重置随机初始权重，1024训练、128开发、128扰动布局，12轮每批16，共768主训练optimizer steps，加tiny共1008步；不使用真实31页标注、GT10、Freeze-A或既有训练模型权重。结构沿用可控视觉编译器，但端口归属与监督是本轮新编译。种子和NPZ hash训练前保存。

## 分开报告两项任务

| 队列 | 组件对micro F1 | 整页组件关系正确率 | 导线对micro F1 | 整页导线关系正确率 |
| --- | --- | --- | --- | --- |
| train | 100.00% | 100.00% | 100.00% | 100.00% |
| validation | 100.00% | 100.00% | 100.00% | 100.00% |
| stress | 100.00% | 100.00% | 100.00% | 100.00% |

checkpoint为第12轮，选择仅使用开发组件micro F1，同分取较低loss。第1轮组件F1已80.86%而导线F1为0%，说明端口直接附着在同一观察wire会贡献高组件分；不能仅靠组件分数断言学会拓扑。这里“整页关系正确率”要求该页评分范围内所有配对同时正确，不是官方exact-net F1。

参数7,477，运行949.05秒，CUDA实际峰值分配336.16MiB。灰度交点CNN从零训练，C4方向平均；12个wire节点、36候选交点、10组件×2端口，几何/端口附着Oracle已提供。

## 同一扰动队列的对照

| 方法 | 组件对micro F1 | 整页关系正确率 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| direct_component_pair | 100.00% | 100.00% | 3595 | 0 | 0 |
| incorrect_component_transitive_closure | 77.35% | 4.69% | 3595 | 2106 | 0 |
| all_component_pairs_positive | 76.86% | 4.69% | 3595 | 2165 | 0 |
| all_component_pairs_negative | 0.00% | 0.00% | 0 | 0 | 3595 |
| observed_same_wire_only_no_image | 81.70% | 23.44% | 2483 | 0 | 1112 |
| frozen_ProbeFlow_R2_plus_same_port_pool | 100.00% | 100.00% | 3595 | 0 | 0 |

incorrect_component_transitive_closure仅对模型的组件关系错误地做传递闭包，用来展示短路器件引发的误合并；不是公平竞争基线。frozen_ProbeFlow_R2_plus_same_port_pool使用已经训练的旧模型加同样端口汇总，未重新训练，训练划分也不同；只衡量此前能力能否实现本任务。observed_same_wire_only_no_image完全不看视觉，隔离端口附着已知信息的贡献。所有对照在选模完成后运行，不影响checkpoint选择。

旋转/组件编号检查来自训练结束，端口顺序/wire编号检查来自CPU审计。具体误差见summary.json、audit.json。若两种传播在受控数据上都满分，不能声称新传播方法优于旧模型；本轮价值主要是任务定义明确区分端口与组件、并完成从零可拟合性检验。

## 可复现与边界

- 模型：experiments/port_lift.py；训练：scripts/train_port_lift.py；控制与冻结核验：scripts/audit_port_lift.py。
- 权重与日志：output/training/port-lift-v1/best.pt、history.json；完整过程output/port-lift-train.log。
- 数据为同一生成器的独立seed布局，不证明家族独立；组件端口随机附着，不反映真实器件功能分布。主模型支持的10个二端器件不代表多引脚IC能力。
- 输入没有直接读truth，但已提供精确观察wire图/端口归属，属于几何Oracle，不是自动识别系统。尚无OCR/端点检测、pin-name/role、跨页逻辑label或比赛格式输出。
- 组件对共享关系不唯一确定超图net-set；全相连三角形可由单个三成员net或三个二成员net产生，不能用组件指标代替完整pin/net恢复。
- 真实31页作者数据仅提供组件关联而无端口监督，仍隔离。本轮不能增加本项目verified designs，实际双人审核仍0/25；官方evaluator等价性仍unresolved。
- RC0与ProbeFlow/NetSet归档SHA核验通过，未修改历史模型或提交包。冻结清单output/port-lift-research-manifest.json。

下一证据应来自可靠真实端口—导线对应，或另立仅预测非传递组件共享关系的真实监督任务；不能把组件关联强制改为pin分区标签。先验证跨域，再决定扩大模型或接入RC0。
