# AI 驱动的 PCB 原理图自动解析

最新补充：[GEA-Net 新架构实验记录](docs/ARCHITECTURE_GEA_EXPERIMENT.md)。已完成关系图网络训练与 MLP 消融；新模型整图收益未稳定，保留为可选分支，默认沿用引脚共识。当前功能测试 61 项。

依据目录中的 2026 创芯大赛赛题六开发。现已实现 **图片输入 → 元件检测 / OCR / 引脚定位 → 拓扑重建 → 比赛结构 JSON 输出**，并完成真实图片上的开发对照。当前整图精度仍不足以宣称比赛成品已达标；官方评测脚本尚未取得。按最新要求暂不编写技术报告。

最新进度、数据审计与实测限制见 [项目进度报告](docs/PROJECT_PROGRESS_2026-10-04.md)。默认增加保持电气拓扑的双模型引脚融合；`--pin-ensemble none` 可关闭。新版官方 10GT 本地代理分为 18.4532，尚非官方评分。

## 自动识别

推理使用本地 ONNX 模型，不依赖 PyTorch 或在线大模型接口。Python 3.12 环境安装 `requirements-inference.txt`；在项目根目录运行：

```powershell
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --output output/real-example/current.json --diagnostics output/real-example/current-diagnostics.json --overlay output/real-example/current-overlay.png
.venv/Scripts/python.exe -m pcbparse batch data/examples --output output/submission-smoke
```

默认模型配置保存在 `models/pipeline.json`。`--pin-model none` 可关闭学习引脚分支；指定 ONNX 路径可以做消融实验。`--local-pin-ocr` 提供高倍率局部 OCR，额外耗时且尚未证实普遍收益。

默认已启用 **GraphSchematic Geometry-to-Electrical 双图推理与 Topology Perception v2**。`--graph-mode legacy` 保留旧流程，`--graph-mode association` 提供关联消融；`--graph-output` 和 `--review-output` 保存双图与受约束纠错队列。结构、实验与完整命令见 [GraphSchematic 执行记录](docs/GRAPHSCHEMATIC.md)。55 页开发代理分从 15.6900 提升至 16.5184；当前引脚融合回放为 16.5197（1 例改善、1 例退步、53 例持平），仍须继续改善元件文本与引脚编号。

扩充数据、许可、架构和验证限制见 [数据与模型执行记录](docs/DATASETS_AND_MODELS.md)。推理过程中只读输入图像与模型，不读训练标注。Dockerfile 已提供，是否通过实际构建以执行记录为准。

## 运行

新增官方资料已用于数据审计、Q&A 评测适配、元件库语义关联及引脚候选训练，详见 [官方资料接入记录](docs/OFFICIAL_MATERIALS.md)。竞赛入口可用 `.venv/Scripts/python.exe agent_system -input data/examples -output output/official-agent-smoke` 测试；按官方目录导出 `final_results/*_Pred.json`。官方基础镜像构建文件为 `Dockerfile.competition`，详细 decision_log 规范和 Docker 构建尚待验证。

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
