# 新增官方资料接入记录

本轮读取用户放入“官方资料”的七个文件。正式报告继续暂缓；本文记录实现、数据审计、实验与部署接口变化。

## 资料与版本

| 文件 | 核实内容与用途 |
| --- | --- |
| 赛题六公开数据集.zip | 200 训练页、旧 10_GTcase、337 条公开元件库 |
| 10gt.zip | 另一组 10 图 GT、元件库和旧 type_list |
| 10GT-0930.zip | 上述 10 图的新版 GT 与更新 type_list |
| 赛题Q&A(1).docx | 非 box 引脚编号/名称不评分；电源/地随机编号按坐标匹配；Name 为芯片型号；单位按图像；线段端点匈牙利匹配 |
| 数据收集指南0915.docx | 数据贡献的有效案例、去重、连接抽检与提交要求 |
| README.md | 公开训练与旧 GT 的组织格式 |
| edarace-agent-base_ubuntu24.tar | Ubuntu24 基础镜像及固定 agent 输入输出契约；不含评分脚本 |

审计由 `scripts/audit_official.py` 重现，哈希、重叠、版本差异见 `output/official-audit/inventory.json`。ZIP 解压经过路径校验。镜像仅解析配置与小层 README，没有执行占位入口。

官方 200 训练页中，160 页与当前检测器训练图像像素一致，其余 40 页保持现有验证划分。它们提供赛题格式监督，不能计作 200 张全新独立图。旧公开 10_GTcase 的 10 页全部与当前检测器训练图像像素重叠，不能用于独立精度声明。

10GT-0930 的 10 页未发现当前训练图像的像素级重叠，来源设计是否完全独立仍未确认。与 10gt.zip 相比图像未变，十页 nets 均更新，部分元件/引脚字段也修改。新版十页通过本地结构校验。公开 200 页中 111 页与本地严格校验规则冲突，常见问题有噪声 pin key、重复归网、引用问题；这不等于官方已判定整页无效。引脚训练只使用有限且位于图内的坐标，保留弱标注属性。

## 赛题格式与评测

`pcbparse/competition.py` 将内部类型转为新版 r/c/l/v/box 等类型，合并晶振、运放、BJT、MOSFET 的细分。保留官方特殊拼写 battary、transfomer。未列类型拒绝导出。

新增 Q&A-informed 代理：非 box 引脚按位置一对一匹配，box 引脚仍要求编号与名称；电源、地及内部匿名对象按类型/坐标关联；网络使用匹配后的引脚映射，线段用端点匈牙利匹配。20px bbox、5px 引脚/导线及网络聚合仍是本地实现选择，`official_equivalence:false`。官方评分脚本未取得。此口径与以前严格 key/pinname 的 55 页代理不同，分数不能跨口径比较。

原赛题比赛 JSON 使用左下角坐标；新版 GT 图像核对也与左下角相符。数据收集指南对新增贡献数据另要求左上角绝对整数坐标，并声明相对坐标归一化基准。两者是不同接口，不能混用。训练读入官方 GT 后转回图像左上角裁剪，推理结果仍导出左下角坐标；本轮没有提交数据贡献包。

## Library-Conditioned Semantic Association

从 cpntLibrary 编译 337 个型号、3,838 个去重引脚号，其中 371 个编号有多个名称候选。语义库去除了 project/case 引用，不保存图像、坐标或 GT 网络。

局部 OCR 与库型号需要高相似度和候选间隔。引脚名称还需满足局部位置、行列对齐、库内编号一致。未编号视觉端点通过全局一对一分配关联到库编号；重复 GND/VIN、多义型号或缺少文字时弃权。模块不移动端点、不复制 GT 坐标、不新增导线。

这将库先验嵌入几何—拓扑—语义协同，使语义在视觉允许的候选中选择。目前仍使用手工相似度与阈值，不是训练好的异构 GNN。

相同 10GT-0930、固定检测/OCR/拓扑分支的对照：

| 模块 | Component F1 | Pin F1 | Net F1 | Line F1 | Q&A 代理分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原双图流程 | 0.37886 | 0.11816 | 0.07418 | 0.09221 | 17.8382 |
| 加元件库关联 | 0.37886 | 0.12248 | 0.07699 | 0.09996 | 18.1222 |
| 官方标注引脚适配 | 0.37886 | 0.13093 | 0.09269 | 0.10792 | 18.9623 |
| 新引脚 + 元件库关联 | 0.37886 | 0.13296 | 0.09504 | 0.11539 | 19.1704 |

仅元件库关联为一页改善、两页退化、七页不变，收益集中于个别案例，保留可选 `--semantic-library models/semantics/component-library.json`。语义对照以 `benchmark-official-semantics-integrated` 为准，它在主流程中使用同一导线 mask。`benchmark-official-new-pins` 曾关闭文本约束，其混合变量实验不用于引脚收益结论；正式对照为 `benchmark-official-new-pins-controlled`。

## 官方标注引脚适配

`prepare_official_pins.py` 从 200 训练页得到 8,977 个裁剪：7,071 训练、1,906 验证，保持 160/40 页划分，不用新版 10GT 训练。训练裁剪为标注 body 与端点并集框；运行仍使用预测检测框。

原引脚模型 warm start 后训练六轮独立候选，验证坐标 F1=0.8762（5px、oracle 裁剪，验证集选择阈值/检查点）。该数值不能与旧模型不同裁剪数据的指标直接比较，更不代表整图编号 F1。候选保存在 `models/candidates/official-pins`，记录在 `output/training/official-pin-adaptation`。默认权重是否替换以整图对照和原 55 页回归为准。

55 页原代理回归已完成且零失败：原权重 16.51843，候选 16.51335；14 页改善、14 页退化、27 页不变。候选 Pin F1 0.09811（原 0.09769），Net F1 0.19624（原 0.19660），Line F1 0.07176（原 0.07206）。没有整体收益，因此保留原默认模型，候选供官方数据适配与下一轮融合实验使用；不按新版小批 GT 的结果贸然覆盖默认。完整对照见 `output/ablation/official-pins-55.json`。

## 竞赛入口

基础镜像 README 明确固定接口：

```bash
/submission/agent_system -input /input -output /output
```

新增入口提供隔离逐案例推理、115 秒单图超时、6,900 秒总预算，输出 `final_results/<case_name>_Pred.json`、decision_log.json 和 runs/ 中间记录，输入目录只读。失败重跑清理对应旧答案，避免过期输出。

`Dockerfile.competition` 使用 `FROM edarace-agent-base:ubuntu24` 并替换占位入口。构建前运行 `docker load -i 官方资料/edarace-agent-base_ubuntu24.tar`。本机无 Docker，尚未验证构建。decision_log 完整字段依赖镜像 README 引用的 EDARace 手册，目前生成可审计的暂定字段，不能宣称官方验收完成。

可选大模型适配器支持官方 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，仅使用显式注入模型。agent 默认本地运行；`--llm-max-calls-per-case 2` 显式启用局部纠错，计入单图与总时间预算。未配置、超时或不合法修改均保留已完成的本地答案。真实接口仍待验证；失败或未返回 usage 时标记 token 统计不完整，不宣称零消费。

51 项测试通过，覆盖随机电源编号、box 引脚严格匹配、更新类型与类别来源、语义候选弃权、旧答案清理、在线超时回退与时间预算。Windows 上以 Python 执行 agent 已验证输出目录；Linux 镜像入口仍需 Docker 环境验证。

```powershell
.venv/Scripts/python.exe agent_system -input data/examples -output output/official-agent-smoke
.venv/Scripts/python.exe scripts/benchmark_official.py --output output/benchmark-official-baseline
.venv/Scripts/python.exe scripts/benchmark_official.py --semantic-library models/semantics/component-library.json --output output/benchmark-official-semantics-integrated
.venv/Scripts/python.exe scripts/prepare_official_pins.py
.venv/Scripts/python.exe scripts/benchmark_official.py --pin-model models/candidates/official-pins/pin-heatmap.onnx --output output/benchmark-official-new-pins-controlled
.venv/Scripts/python.exe agent_system -input data/examples -output output/official-agent-candidate --pin-model models/candidates/official-pins/pin-heatmap.onnx --semantic-library models/semantics/component-library.json
```

## 参考论文与创新边界

已核对 Q&A 给出的 [PCBnet 原始论文](https://arxiv.org/html/2608.27923v1)：包含视觉识别、几何邻近构图和知识辅助纠错。本项目进一步探索显式几何图/电气超图、拓扑感知否决几何邻近，以及保留多义元件库候选的受约束关联。这些目前是工程实现与实验假设，尚未证明学术新颖性或达到论文精度。
