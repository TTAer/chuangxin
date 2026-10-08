# AI 驱动的 PCB 原理图自动解析

依据目录中的 2026 创芯大赛赛题六开发。现已实现 **图片输入 → 元件检测 / OCR / 引脚定位 → 拓扑重建 → 比赛结构 JSON 输出**，并完成真实图片上的开发对照。当前整图精度仍不足以宣称比赛成品已达标；官方评测脚本尚未取得。按最新要求暂不编写技术报告。

## 自动识别

推理使用本地 ONNX 模型，不依赖 PyTorch 或在线大模型接口。Python 3.12 环境安装 `requirements-inference.txt`；在项目根目录运行：

```powershell
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --output output/real-example/current.json --diagnostics output/real-example/current-diagnostics.json --overlay output/real-example/current-overlay.png
.venv/Scripts/python.exe -m pcbparse batch data/examples --output output/submission-smoke
```

默认模型配置保存在 `models/pipeline.json`。`--pin-model none` 可关闭学习引脚分支；指定 ONNX 路径可以做消融实验。`--local-pin-ocr` 提供高倍率局部 OCR，额外耗时且尚未证实普遍收益。

扩充数据、许可、架构和验证限制见 [数据与模型执行记录](docs/DATASETS_AND_MODELS.md)。推理过程中只读输入图像与模型，不读训练标注。Dockerfile 已提供，是否通过实际构建以执行记录为准。

## 运行

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
