# GEA-Net 新架构实验记录

更新：2026-10-04。已实现、训练、导出并接入真实图片推理。当前作为可选实验架构，默认仍使用此前的引脚共识与几何约束关联。

## 架构与动机

此前关联器用距离、像素支持和外向方向的固定加权分，难以利用同一元件的其他引脚和附近导线网络的信息。本轮实现 Geometry-to-Electrical Association Network（GEA-Net），以关系消息传递学习候选连接的软评分。

原始几何图仍保留 pin endpoint、wire endpoint、junction、label 及真实 wire 边。学习分支只把当前观察中已连通的导线组收缩成 wire-group 节点，与相关元件的 pin 节点构成局部因素图。它没有使用目标网络来构图，也没有替代完整几何图。

两种关系分别聚合同一元件内的引脚信息、候选 pin–wire-group 关联信息。10 维节点特征经过 48 维编码和两层残差关系消息传递，12 维边特征与两个端点嵌入一起输出连接 logit。边特征包含距离、实际图像线支持、方向、导线切向、端点距离和相对元件位置；没有图像颜色、位号文本、案例 ID、目标网络成员或目标引脚编号查表特征。

实现参考了 [GraphSAGE 的邻域聚合思想](https://arxiv.org/abs/1706.02216)。本项目是针对原理图的关系与约束组合实验，没有据此宣称首创图消息传递或已经取得学术新颖性验证。

推理采用 `0.5 × (原几何分 − 0.62) + 0.5 × (模型输出 − 0.5)` 作为束搜索软增益。模型输出未作概率校准，0.5 权重仍是手工融合。原始几何分、图像支持、Topology Perception 的许可、电源冲突、无源器件短接约束及近似束搜索共识仍有效。它可以改变候选取舍，但不能添加元件、引脚、物理线段或修改坐标。同名标签飞线逻辑合并继续保留。

额外提供高精度非连接否决：仅在至少 30 个验证样本支持 ≥97% 非连接精度时启用。本次未达到要求，阈值为 −0.01，否决自动弃权；当前模型贡献来自软评分。没有把“不支持否决”包装成高置信度判断。

## 监督、训练与消融

真实 PCBnet 标注网络提供配对监督，元件框和导线采用标注几何。端点加入随机方向、2.5–7 像素位移，模拟定位误差；排除混合多个标注网络的观察导线组、无网络标注的 pin 以及近重复页 0237。页面划分沿用现有训练/验证，精确重复只取一次。

- 241 页训练、55 页验证；每页 3 次扰动，共 888 个图样本。
- 训练连接/非连接配对 29734 / 2272；验证 6764 / 434。
- 新版官方 10GT 未用于本次训练。
- 两层图网络、无消息传递 MLP 分别训练 12 轮；对照使用同样数据、类别平衡损失、种子和训练顺序。参数量分别 19921 / 5809，尚未完成等参数量和多种子对照。
- 最佳模型按验证平衡准确率选优；公开验证参与选优，不是独立测试。

| 扰动几何验证 | 平衡准确率 | 连接 F1 | 非连接 F1 |
| --- | ---: | ---: | ---: |
| 全部预测连接 | 0.50000 | 0.96892 | 0.00000 |
| 无消息传递 MLP | 0.83410 | 0.90688 | 0.38196 |
| 两层关系图网络 | 0.87257 | 0.93735 | 0.47964 |

正例约占 94%，所以“全部预测连接”也能获得很高连接 F1。本轮以平衡准确率和非连接指标检查效果。图网络相对 MLP 的平衡准确率提高约 3.85 个百分点，但该结果属于标注几何扰动任务，不能作为自动识别准确率。

## 整图效果与选择

| 55 页图像观测回放配置 | Pin F1 | Net F1 | Line F1 | 加权本地代理分 |
| --- | ---: | ---: | ---: | ---: |
| 此前默认引脚共识 | 0.09774 | 0.19660 | 0.07206 | 16.51966 |
| + GEA-Net 软评分 | 0.09774 | 0.19617 | 0.07356 | 16.51988 |
| + MLP 软评分 | 0.09774 | 0.19602 | 0.07470 | 16.52610 |

GEA-Net 为 4 例改善、4 例退步、47 例持平；均分变化约 +0.00021，Net F1 下降。图网络没有在整图代理分上超过 MLP，不能宣称图网络整体优于旧架构。

新版官方 10GT 已重新执行图像推理，代理分仍为 18.45322，与此前默认相同。0101（改善）和 0159（退步）的实际图片推理与缓存回放完全一致。本轮因此保留两个可选候选模型，默认不启用新的学习关联器。

55 页为保存的图像观察/候选证据回放；10GT 为真实图片端到端运行。二者评分规则不同，均没有官方 evaluator 等价验证。当前候选覆盖范围也很窄：只评估附近且未局部连接的已编号 pin，不能补齐漏检 pin 或处理任意远距断线。

## 接入与验证

候选权重为 `models/candidates/association-balanced/association.onnx` 与 `models/candidates/association-mlp-balanced/association.onnx`。单图、批处理、竞赛入口、纠错回放均支持 `--association-model`。学习分支无 PyTorch/PyG 推理依赖，使用本地 ONNX Runtime。

```powershell
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --output output/gea.json --association-model models/candidates/association-balanced/association.onnx
.venv/Scripts/python.exe agent_system -input data/examples -output output/gea-agent --association-model models/candidates/association-balanced/association.onnx
.venv/Scripts/python.exe scripts/evaluate_association.py
.venv/Scripts/python.exe scripts/benchmark_official.py --pin-ensemble auto --association-model models/candidates/association-balanced/association.onnx --output output/official-gea-recheck
```

`--association-model none` 关闭新分支。训练入口 `scripts/train_association.py --balance --epochs 12` 需要 PyTorch 环境及本地 PCBnet 原始数据；无消息传递对照增加 `--layers 0` 并指定独立输出目录。训练数据不随开发包分发。

61 项功能测试通过。20 个不同大小的验证图检查 ONNX 与 PyTorch logits 的最大误差 <3e−6；随机节点重排后的对应边输出误差 <1e−6。验证了动态形状与图置换一致性，但不是精度证明。图节点预算为 2048，超过预算或运行时模型错误会记录原因并回退原启发式关联。

## 下一轮改进方向

当前训练使用干净标注导线，而真实误差主要包含漏检引脚、错编号、导线分割缺失与 OCR 绑定错误，存在明显输入分布差异。下一轮应先生成训练页的实际检测观察，用严格的几何与标注匹配生成可靠正负监督，并增加独立数据集；再扩展到带图像局部编码、junction 类型预测和 pin 候选身份的联合推理。验证应同时观察 Net F1 与错误短接，而非只追求局部分类指标。
