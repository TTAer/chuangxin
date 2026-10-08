
# AI 驱动的 PCB 原理图自动解析

2026-10-08 新尝试：[原图线条残差与页面模板](docs/PAGE_WIRE_RESIDUAL_RESULTS.md)。局部模板开发误改26，页面共享模板误改1均失败；第三次零训练错误校准开发过关后，新640页primary完整分区244→246/256、修复2破坏1，hard112→115/128、FP254→126但FN292→311。收益小且有新错，发布门槛失败。构造混合线宽几何ready仍错判JOIN反例；无新模型优化、RC0与真实verified不变。

2026-10-08 新训练：[单点反事实拓扑代价](docs/TOPOLOGY_COST_RESULTS.md)。匹配普通CE/risk两组，各tiny400+主1304步，总3408；只用既有train/dev计算交点GT敏感度，新640页留出。primary risk相对CE减少1471关系FP、F1 87.64%→91.85%，仍低于冻结研究基线94.67%；hard69.16%也低于基线89.02%。绝对门槛失败，未接入RC0，无真实verified增长。

2026-10-08 新训练：[二维信息保留×模糊增强四组对照](docs/EVIDENCE_FACTORIAL_RESULTS.md)。完成四组GPU训练、6816已完成优化步；另有显存争用中断尝试，步数未记录且不混入完成预算。全新640页，开发选Spatial narrow；primary完整分区239→240/256、修复3破坏2，hard完整分区102→106/128但关系F1 93.43%→92.31%、FP净增179。四组门槛均失败，增强未带来收益，保持research-only与RC0冻结。

2026-10-07 新训练：[RadialEvidenceNet及双视图纠错](docs/RADIAL_EVIDENCE_RESULTS.md)。3,522参数两组新GPU模型，各tiny400+主1304步，新合成程序总5124页；开发交点100%，强模糊同页完整分区94.14%→91.02%/89.84%，直接替换失败。另批新384页固定双视图≥.99，3次修改全部改错，保守门槛也失败。两条路线冻结research-only，保留OrderedV2/RC0，不把高置信一致当独立证据。

2026-10-07 新推进：[有序线观测与非均匀校正](docs/ORDERED_RECTIFICATION_RESULTS.md)。保留 V1 开发失败；V2 修复旧 train11/dev1 线数异常，新原生不等间距256页同冻结模型覆盖14.84%→98.05%、全量关系37.25%→98.87%、完整分区95.31%。均匀对照略退、剪切/旋转各128页全拒绝，受控门槛通过但未接入RC0，无新训练或真实verified增长。

2026-10-07 新推进：[多对比度方向证据与弱线条配准](docs/GUARD_V2_RESULTS.md)。不重训，开发256页覆盖/程序100%；新主压力同页覆盖224→256/256、net F1 92.75%→100%、完整分区100%、程序255/256，唯一C→R错误保留。新剪切256与旋转128全部拒绝，首次通过此受控门槛；train/dev仍12条ready线数异常，不能把ready当真值。保留研究候选/unknown入口，真实独立精度仍未验证，RC0未改。

2026-10-07 深度推进：[全页方向检查与对比度/梯度成对训练](docs/CONTRAST_EDGE_RESULTS.md)。4096新布局两版各8轮、共4896优化步。剪切同页V2误接受72/256→新检查0/256；正常强模糊覆盖222/256=86.72%，未过覆盖门槛。该222页两模型完整程序/分区100%，模板87.39%、固定旧模型83.33%；全量Contrast/Edge F1 90.94%/91.40%，完整分区同86.72%，梯度无整页收益。研究候选优先简单Contrast，实际入口拒绝剪切，RC0不变。

2026-10-07 新推进：[多区域亚像素配准与576张新留出](docs/REGISTRATION_V2_RESULTS.md)。未重训模型：开发配准覆盖/完整程序100%，新主压力同页覆盖175→256、冻结模型F1 71.71%→88.99%，完整分区58.98%仍失败；新配准+模板受控主压力100%。hard覆盖90.23%、模型全量91.28%/分区74.61%；剪切64页中18页误ready且完整分区0%，ready不作为认证，CPU入口维持unknown，RC0不变。

2026-10-07 新推进：[图像配准与平移/缩放布局训练](docs/REGISTERED_CIRCUIT_RESULTS.md)。同页旧模型开发raw F1 30.91%→只读图像配准83.69%；触发2048新布局8轮GPU训练，开发96.32%、压力71.49%/完整分区42.19%，未过门禁。配准压力覆盖70.31%，该部分模型F1 90.66%/分区60%，模板94.53%/91.11%。只支持全局仿射正交线格，不冒充任意布局；失败入口已验证不生成程序，旧冻结与RC0保留。

2026-10-07 再推进：[可分离连接点与固定三方案评测](docs/SEPARABLE_CIRCUIT_RESULTS.md)。新渲染器5817次单指令比较无已知碰撞，4096新布局8轮训练。新开发模型/模板均100%；压力模型F1 87.26%、完整分区85.55%，模板完整程序/分区100%，组合未修复错误合并。一致性保留132/256页、0观察错误、覆盖51.56%；没有学习模型优势，保持受控研究，旧冻结与RC0保留。

2026-10-07 新一轮：[原生256px单步视觉读取与栅格可识别性](docs/FACTORIZED_CIRCUIT_RESULTS.md)。从零训练4096新布局8轮、187453参数；开发net F1 99.77%，压力62.24%、完整分区35.55%，门禁失败。线宽6为98.07%/67.41%，线宽7有不同电气关系却整页像素相同的严格反例。无训练模板压力70.41%且器件100%，没有整体模型优势证据；提供碰撞样例、分域结果和保守unknown推理入口，RC0不变。

2026-10-07 新视角完整训练：[CircuitProgramNet-H整图到电路程序](docs/CIRCUIT_PROGRAM_RESULTS.md)。新取得LogicBench官方train 700图、CI2N扩至681下载/649准入图，最终六类来源2282图、37918裁剪用于本轮图像预训练；8192合成整图训练12轮。固定模型开发256页完整程序100%，粗线压力256页net F1仅29.43%、完整程序0%，单步无劣于多步；冻结研究原型，未接入RC0。提供无GT图像推理脚本、重绘及SHA证据；独立真实精度仍未验证。

2026-10-07 下一轮：[端口观察压力与候选关系上下界](docs/PORT_LIFT_OBSERVATION_UNCERTAINTY.md)。冻结模型准确三/四端口队列100%，错误附着组件F1降至88.53%；候选完整时二/四端口覆盖65.83%/81.70%、确定部分0错误，候选缺失可产生798错，接口默认完整性未确认并保留unknown。本轮未重训、无独立真实精度，RC0不变。

2026-10-07 新架构GPU训练完成：[PortLiftNet独立端口与非传递关联](docs/PORT_LIFT_RESULTS.md)。7,477参数，从零训练1024布局12轮；128合成扰动布局的组件/导线F1与整页关系均100%。错误组件传递闭包产生2106 FP、F1降至77.35%；旧ProbeFlow+相同端口汇总也100%，未证明新传播优势。真实监督仍未准入，RC0保持冻结。

2026-10-06 新视角训练完成：[ProbeFlowNet视觉通导率与主动探针响应](docs/PROBE_FLOW_RESULTS.md)。取得CI2N 64张训练图用于图像预训练；R2新留出128个合成布局完整分区100%，简单学习交点+CC也100%，尚无传播优势或真实整页证据。新增31条作者核验net及257条派生参考隔离未训练；RC0与人工审核主线未改。

人工审核入口：[新版标注工作台 A](http://127.0.0.1:8767/review-studio/index.html?pass=A) · [B](http://127.0.0.1:8767/review-studio/index.html?pass=B)。支持缩放拖动标点、快捷键、独立草稿自动保存和 JSON 导入导出；[操作说明](docs/REVIEW_STUDIO.md)。双击 `启动审核工作台.cmd` 可在默认浏览器启动。

最新实验：[NetSetQuery完整监督 Oracle Ladder](docs/NETSET_ORACLE_LADDER_RESULTS.md)。49训练/14开发页，两组各24轮GPU训练；开发网络成员F1为33.11%/32.22%，完整监督第一关失败，停止覆盖率/真实观察/弱标签训练，不接入RC0。

新视角架构：[NetSetQuery整页电气网络集合预测](docs/ARCHITECTURE_NETSET_QUERY.md)。该文为训练前原型说明；实际训练结论以上述Oracle报告为准。

验证主线：[验证阶段执行状态与人工审核入口](docs/VALIDATION_PHASE_STATUS.md)。仍需真实双人审核和正式评测代码；新的隔离架构研发按用户最新授权执行。

当前决定（2026-10-06）：[PPN + MST Agreement Gate评估](docs/TOPOLOGY_AGREEMENT_RESULTS.md)。局部互补成立，65页系统门槛仍未通过；自动拓扑接入冻结，RC0保持不变。

最新研发结果（2026-10-06）：[拓扑发布审计与多来源 Masked Stroke Transformer GPU训练](docs/TOPOLOGY_RELEASE_AND_MASKED_STROKE_RESULTS.md)。新模型保留为研究候选，RC0提交基线不变。

2026-10-06 最新实验：[PortPartitionNet数据扩充与复训](docs/PORT_PARTITION_EXPANDED_RESULTS.md)。真实训练953→3,732，原133例固定，新增668例自动候选开发诊断；三种子F1均值原队列0.8170→0.8640、自动队列0.9195→0.9732。默认种子高权重合并precision在原队列仍93.04%，保持research/evidence-only，未改RC0或比赛输出。

2026-10-06 新实验：[PortPartitionNet训练与结果](docs/PORT_PARTITION_RESULTS.md)。按用户新增要求启动隔离架构研究：四端口视觉token、图消息传递和合法连通分区；1,086个弱监督局部样本，三种子跨方向开发F1均值0.8170，对照/分区消融为0.4019/0.6575。完成ONNX核验与无GT原图候选入口，尚非完整比赛解析器，不接入RC0。双人审核与Freeze-A仍待真实证据。

2026-10-06 当前：[双人审核执行单与协议锁](docs/HUMAN_REVIEW_HANDOFF.md)。现有25例和V2工具已锁定，人工审核仅聚焦四个优先来源；提供锁校验后导入入口，空白模板不能计为审核。用户确认实际双人审核尚未完成；verified designs仍0、坐标语义未冻结。162测试通过，全部冻结SHA不变，RC0保持冻结。

2026-10-06 最新：[独立几何审核 V2 与整页 revision 核查](docs/SEMANTIC_ENDPOINT_REVIEW_V2_PROGRESS.md)。25 案例支持独立空白两遍、先观察后揭示参考、无边界/不确定结论及实际侧确认；同 pin 资格与人工尺度归一化均待真实审核，统计 null。13 页/1303 CAD pin 段核查发现 BlueSMiRF 仅 50/89 一致；筛出四个几何一致的优先来源及一个待核跳线的第五候选。160 测试与全部冻结 SHA 校验通过，verified designs 仍 0，RC0 不变。以下为历史记录。

2026-10-06 最新：[Pin 坐标语义校准与独立评测基础设施](docs/PIN_POINT_CALIBRATION_PROGRESS.md)。新增 25 案例离线 Geometry Inspector、687 候选的法向/切向与恢复率统计、审核 JSON 导入核验，以及 13 页/976 CAD 实例的整页审核清单。首次推理门禁已接入统一坐标定义及独立复核；正式 verified designs 仍 0。146 测试及 RC0/V1/R 冻结文件 SHA 校验通过，RC0 不变。以下为历史记录。

2026-10-05 最新：[端点几何与身份联合核查](docs/ENDPOINT_QUALITY_PROGRESS.md)。完成 65 页、687 个新端点候选的像素证据和几何/身份 oracle 诊断；目前无满足晋级门槛的新分支。Freeze-A1 完成 27 IC 可见身份核对，正式独立设计仍 0，首次推理门禁保持关闭。129 项测试通过；RC0/V1/R 冻结文件逐 SHA 校验通过。以下为历史记录。

2026-10-05 最新：[文字证据与保守提交改进方案](docs/PIN_EVIDENCE_IMPROVEMENT_PLAN.md)。完成65页无GT裁剪的局部OCR闭环，两域未过整图gate；56难例完成代理图像草稿审查，raw HRR不改。Freeze-A1注册候选去重13页/27IC/262pin，正式就绪仍0。118测试通过，RC0/V1/R归档不变，本轮未训练模型。以下为历史记录。

2026-10-05 最新：[Residual 难例纠错与局部 OCR](docs/PIN_TOKEN_RESIDUAL_RESULTS.md)。198 页挖掘 303 个难例；残差 Logistic EPR 99.70%、HRR 16.07%，未达门槛，保持关闭。限定尺度局部 OCR 恢复 number 9/24、name 11/24（oracle ROI）；Freeze-A 已有局部像素标注，正式就绪页仍 0。115 项测试通过，RC0/V1 均冻结。以下为历史记录。

2026-10-05 最新：[PinTokenNet V1 实际训练](docs/PIN_TOKEN_V1_RESULTS.md)与[跨域框归因](docs/RC0_BOX_ERROR_DIAGNOSIS.md)。64 页生成 2,252 个实际 OCR 监督 token，完成两组 GPU 训练与 CPU ONNX 导出；pin 关联 87.85% 低于最近端点 93.37%，暂不接入。110 项测试通过，RC0 保持冻结；Freeze-A 正在采集与审核。以下为历史记录。

2026-10-05 最新：[RC0 评估与晋级门槛](docs/RC0_EVALUATION.md)。冻结同一镜像的默认/候选各 50 次批验全部通过，103 项测试通过。25 组跨域框 F1 为 0.4126 / 0.0897（IoU 0.3 / 0.5），不能替代同域盲测；Freeze-A 尚未取得，默认不切换。已准备文字—引脚联合绑定试验方案，尚未训练。以下为历史记录。

2026-10-05 最新：[局部位号锁定与电气权限保护](docs/STABLE_IDENTITY_AND_ELECTRICAL_GUARD.md)。新候选 PCB55 16.9399，9 页改善 / 1 页代理标签冲突退步；10GT 28.8156。7 个可见未分配键均恢复，正确方框位号保持 71；99 项测试通过，默认不切换。以下为历史记录。


2026-10-05 最新：[Box 生命周期审计与位号全局约束](docs/BOX_LIFECYCLE_AND_ASSOCIATION.md)。guarded 候选 PCB55 16.9726（9 页改善、4 页退步），10GT 28.8156；正确 Box 键 63→71，库内 GT 正确键且确认型号 17→19。修复 M 类整页导出失败，91 项测试通过；默认不切换，旧比赛候选冻结，新镜像仅四例验证。以下为历史记录。


2026-10-05 最新：[IC 型号检索与身份恢复](docs/IC_IDENTITY_RECOVERY.md)。完成 Top-K 型号检索、引脚证据消歧与输出身份审计，恢复 3 个 OCR 粘连型号。组合候选 PCB55 代理分 16.8075，5 页改善、无评分退步；10GT 28.8156，默认不切换。83 项测试通过，19 页实际图像重放完全一致。当前元件库身份覆盖仍仅 27.3%。

以下保留此前阶段记录。

2026-10-05 最新：[IC 候选过滤与检测框训练](docs/SIDE_REFINEMENT_2026-10-05.md)。完成 32 组阈值整图对照、两组配对框训练，新增唯一名称驱动的缺失引脚增补。候选 10GT / PCB55 为 28.8156 / 16.7617，默认不切换；79 项测试通过。当前镜像已通过 10/50 案例批验，可选分支通过两例容器验证，完整官方评测仍未核验。

以下为历史阶段记录。

最新进展：[SideStripNet 四边高分辨率引脚实验](docs/ARCHITECTURE_SIDE_STRIP.md)。完成三组新架构训练，自建 240 个合成布局族，新增下载 CircuitOCR 80 个图文样本（仅审阅，未用于训练）。local 对照整图代理分 28.8156 / 16.7454，尚未超过此前 10GT 最佳候选；默认仍为 26.5107 / 16.7037。76 项测试通过，新容器验证默认和可选单例。膨胀卷积优势未成立，新分支默认关闭。

以下保留此前阶段记录；当前实验、镜像和默认决策以上述新记录为准。

最新进展：[IC 结构化解析与运行验收](docs/IC_STRUCTURE_AND_RUNTIME.md)。新增四边归属、沿边序列对齐和图像笔画约束候选；10GT 候选代理分 28.8739，Pin F1 0.1811、Net F1 0.1280，收益集中于一页。55 页候选代理分 16.7187，保留可选 `--ic-pin-mode structure`，默认仍为此前 26.5107 / 16.7037 配置。72 项测试通过。GEA-Net 和方向场继续冻结。此前完整改进见 [元件框与语义识别改进](docs/RECOGNITION_IMPROVEMENT.md)。

依据目录中的 2026 创芯大赛赛题六开发。现已实现 **图片输入 → 元件检测 / OCR / 引脚定位 → 拓扑重建 → 比赛结构 JSON 输出**，并完成真实图片上的开发对照。当前整图精度仍不足以宣称比赛成品已达标；官方评测脚本尚未取得。按最新要求暂不编写技术报告。

最新进度、数据审计与实测限制见 [项目进度报告](docs/PROJECT_PROGRESS_2026-10-04.md)。`--component-box-mode detector` 可关闭本轮输出框修正，`--semantic-library none` 可关闭语义库。此前主/辅引脚共识仍可显式启用，完整回退命令见改进记录。各分数尚非官方评分，公开开发集不能当作独立测试。

## 自动识别

推理使用本地 ONNX 模型，不依赖 PyTorch 或在线大模型接口。Python 3.12 环境安装 `requirements-inference.txt`；在项目根目录运行：

```powershell
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --output output/real-example/current.json --diagnostics output/real-example/current-diagnostics.json --overlay output/real-example/current-overlay.png
.venv/Scripts/python.exe -m pcbparse batch data/examples --output output/submission-smoke
```

默认模型配置保存在 `models/pipeline.json`。`--pin-model none` 可关闭学习引脚分支；指定 ONNX 路径可以做消融实验。`--local-pin-ocr` 提供高倍率局部 OCR，额外耗时且尚未证实普遍收益。

默认已启用 **GraphSchematic Geometry-to-Electrical 双图推理与 Topology Perception v2**。`--graph-mode legacy` 保留旧流程，`--graph-mode association` 提供关联消融；`--graph-output` 和 `--review-output` 保存双图与受约束纠错队列。结构、实验与完整命令见 [GraphSchematic 执行记录](docs/GRAPHSCHEMATIC.md)。此前拓扑与引脚共识阶段的 55 页代理分为 16.5197；当前识别改进后为 16.7037，仍须改善 IC 端点、引脚编号和漏检。

扩充数据、许可、架构和验证限制见 [数据与模型执行记录](docs/DATASETS_AND_MODELS.md)。推理过程中只读输入图像与模型，不读训练标注。Dockerfile 已提供，是否通过实际构建以执行记录为准。

## 运行

新增官方资料已用于数据审计、Q&A 评测适配、元件库语义关联及引脚候选训练，详见 [官方资料接入记录](docs/OFFICIAL_MATERIALS.md)。竞赛入口可用 `.venv/Scripts/python.exe agent_system -input data/examples -output output/official-agent-smoke` 测试；按官方目录导出 `final_results/*_Pred.json`。`Dockerfile.competition` 已使用用户提供的 Ubuntu24 基础镜像构建，并通过断网、只读输入的单例运行，结果与 Windows 完全一致；完整官方 runner、评分器和详细 decision_log 规范仍待核验。部署证据见 `output/container-validation.json`。

在项目根目录执行：

```powershell
python -m pcbparse demo --output output/demo
python -m pcbparse validate output/demo/prediction.json
python -m pcbparse evaluate output/demo/prediction.json output/demo/golden.json --output output/demo/metrics.json
python -m unittest discover -s tests -v
```

本机也可以用已验证的 bundled Python：

```powershell
& 'C:/Users/18327/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m pcbparse demo
```

`output/demo` 包含人工定义的合成观测、独立定义的 golden、预测 JSON、代理指标和 SVG 示意图。100 分仅表示这个小型几何样例重建正确，不能说明 PCBnet 或比赛准确率。

## 接入观测

```powershell
python -m pcbparse build output/demo/observations.json --output submission/case_001/case_001.json --tolerance 2
```

输入 `observations.json` 是检测器/OCR/人工标注的中间数据，不是图片。字段说明：

| 字段 | 内容 |
| --- | --- |
| components | 位号为键；`type`、`Name`、`value`、`bbox` |
| pins | 位号 → `pin_<number>` → `pinname` 和 `point` |
| wires | 实际导线列表，每条恰好两个 `{x,y}` 点 |
| junctions | 已确认的电气交点坐标列表 |
| labels | `{name: "RUN", pin: "RUN_1.1"}` 等逻辑标签；仅提供已确认的标签 |
| coordinate_system | `bottom-left`（默认）或 `top-left` |
| image | 原图 `width`、`height`；转换左上角坐标时须给 height |

左上角转左下角采用连续图像边界约定 `y_out=height-y_in`。若来源标注采用整数像素中心，则需要先确认是否应使用 `height-1-y`；接入 golden 时必须验证，不能混用。裁剪坐标必须先加裁剪偏移，缩放坐标必须先还原至原图。

拓扑采用端点接触/端点落在线段上连接，内部十字交叉仅在 `junctions` 中存在交点证据时连接。重合引脚可以直接连接。同名 `labels` 合并逻辑网络，保留已有实际线段，不添加虚构跨图导线。输入导线的分段方式会保留，因此上游必须提供符合 GT 粒度的线段。没有引脚的孤立导线暂不导出。

输出仅包含 `components`、`pins`、`nets` 三段，网络使用 `hyperGraph` 字符串和 `edges` 对象。JSON 加载拒绝重复键，校验器拒绝未知引脚、重复归网、无穷/NaN 坐标和非两点线段。

## 本地评测的范围

Component 按 key/type/坐标/严格文本匹配；IC 比较 `Name`，电阻电容比较 `value`。Pin 按位号和 pin number 匹配，比较 pinname 和欧氏坐标距离。Net 用 pin-set F1 作权重进行匈牙利匹配，再累计共同 pin；Line 仅在匹配网络内一对一匹配，允许端点交换。

指标权重为 0.30 / 0.25 / 0.35 / 0.10；bbox 默认采用各坐标误差 ≤20，pin/line 默认欧氏距离 ≤5。**这是开发代理实现**：官方网络匹配权重、宏/微平均、空网络规则、bbox 距离规则尚未核实。输出带 `official_equivalence:false`，获得官方评测脚本后应替换。没有模拟 Token 排名分或数据贡献分。

详细需求、缺口及后续验收见 [项目推进记录](docs/PROJECT_STATUS.md)。

