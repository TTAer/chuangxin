# 元件框与语义识别改进记录

2026-10-04。本轮依据用户粘贴的评审建议，把研发重点转向元件输出、引脚身份和可部署性。GEA-Net、DirectionalFieldNet 及当前引脚共识实现保留为实验/回退分支，没有继续堆叠新架构。本文是开发进度记录，正式比赛技术报告继续暂缓。

## 实测发现

对既有 55 页和新版 10GT 输出做几何优先的一对一诊断，再检查 key/type/bbox/Name/value。它是错误定位工具，不与评分器的身份优先匹配完全等价，也不是因果证明。最初只用 IoU 会把“检测框包含 GT 小框”误算成漏检，现已加入近中心包含关系检查。

| 元件错误桶 | 55 页 | 新版 10GT |
| --- | ---: | ---: |
| 正确 | 959 | 89 |
| 漏检或严重偏移 | 1077 | 68 |
| 已检测但身份未分配 | 0 | 6 |
| key 错 | 153 | 0 |
| type 错 | 23 | 7 |
| bbox 范围错 | 8 | 84 |
| IC Name 错 | 34 | 0 |
| value 错 | 181 | 0 |

各行是按优先级归类的 GT 元件数；例如 key 和 value 同时错时归 key，完整失败字段仍记录。55 页公共注释存在缺失，匿名对象及转换规则也会影响桶，不把这些数目当作官方 FN 分类。诊断详情见 `output/recognition-audit/`。贴文中的完成度百分比、示例错误占比没有作为项目实测采用。

10GT 的关键发现是输出框范围，而非 OCR key：C151 已正确读出编号，但检测框宽约 59 像素，GT 框宽 16 像素。当前检测器训练使用 PCBnet 的 `location`，输出范围常含引脚；新版 GT 的范围不同。它会先使元件身份匹配失败，再阻断代理评分中的引脚映射。反过来，代理评估中 IC Name / R、C value 错误只影响元件项，不能笼统宣称所有元件字段错误都会级联阻断 Pin。

10GT IC 引脚诊断为：组件身份受阻 57、30px 内无对应端点 59、端点误差超过 5px 20。55 页 IC 引脚中，组件身份受阻 1250、端点缺失/偏离 347、超过 5px 45、编号错 44、名称错 11。当前 IC 端点召回和局部 OCR 仍是明显问题。

## 实现

新增 `ObservedPassiveBodyBounds`：电容用实际可见的平行极板对确定范围，矩形电阻用局部闭合轮廓确定范围。要求方向、重叠、间距和中心位置支持；缺乏图像证据时保留检测框。OCR 区域只删除与局部裁剪确实相交的像素，避免负索引切片误删其他区域。它不依据 case ID、GT、固定元件型号或 GT 坐标进行修正。

**检测框和输出框分开**：原始检测框继续用于 OCR、引脚模型裁剪、符号遮罩和电气双图；只在最终输出时更新有证据的 R/C bbox。引脚坐标、编号、net 成员和物理边不由这个步骤修改。`component_body_boxes` 和图像证据记录在 observations/diagnostics，LLM 图回放也应用同一输出框。

补齐了官方引脚模型与语义库的四组消融。语义库仍要求局部可见文本、强型号匹配、唯一编号候选；同名 GND/VIN 多编号时弃权。修复一个无效改动：启用语义库但 observations 未变时保留已有关联结果，避免“无语义编辑却重新关联”混入语义增益；型号文字修正现在也记录为显式 edit。

## 四组消融

全部使用相同已缓存的检测/OCR观测，语义改变时读取原图重建图像支撑，不读 GT 决策。每个域内保持相同评分器。

| 配置 | 55 页旧严格代理 | 新版 10GT Q&A 代理 |
| --- | ---: | ---: |
| 原默认：主/辅引脚共识 | 16.519662 | 18.453222 |
| 原默认 + 语义库 | 16.703103 | 18.576009 |
| 官方适配引脚模型单独替换 | 16.513353 | 18.962256 |
| 官方适配引脚 + 语义库 | 16.744491 | 19.170352 |

19.17035 的组合在完整 55 页也有平均收益，证实其价值不只来自 10GT 个别案例。两列使用不同代理口径，不能直接比较分值高低。

## 新默认与实际图片验证

最终采用**官方适配主引脚模型 + 语义库 + 被动元件输出框修正**，不启用引脚 ensemble、GEA-Net 或方向场。完整 55 页和 10GT 已实际推理，修正越界切片后重跑受影响案例；每个最终输出与独立缓存对照比较。三个彩色/灰度、不同规模页面额外检查 auto 默认选择器。结果记录在 `output/recognition-live-verification.json` 和 `output/ablation/recognition-pipeline.json`。

| 数据与指标 | 原默认 | 新默认 |
| --- | ---: | ---: |
| 10GT 元件 F1 | 0.378862 | 0.612751 |
| 10GT 引脚 F1 | 0.128491 | 0.139456 |
| 10GT 网络超图 F1 | 0.081985 | 0.099652 |
| 10GT 导线 F1 | 0.100563 | 0.115393 |
| 10GT 加权代理分 | 18.453222 | **26.510656** |
| 55 页元件 F1 | 0.215821 | 0.214720 |
| 55 页引脚 F1 | 0.097743 | 0.102281 |
| 55 页网络 F1 | 0.196597 | 0.199111 |
| 55 页导线 F1 | 0.072055 | 0.073618 |
| 55 页加权代理分 | 16.519662 | **16.703718** |

10GT 10 页均改善；55 页 16 页改善、13 页退步、26 页持平，元件项略降但引脚、网络和导线平均提高。保留这一退步，不声称每页都更好。首次完整图像推理 55 页约 387 秒、10GT 约 74 秒；后续修正重跑另计，不据此外推所有比赛案例时限。

这些是已经用于模型选择的公开开发集，**不是独立测试或官方成绩**。尤其元件 bbox 的 20px 和引脚的 5px 门限来自本地代理实现，官方等价性尚未核验；元件框收益的主要来源是输出匹配改善，不能当作检测器召回提升。Pin 和 Net 仍未达到贴文中 0.20 / 0.15 的开发目标。

## 冻结数据与部署

AMSNet 预留 25 个规范化 SPICE 文本组、32 个案例，清单 `data/splits/amsnet-frozen.json`。此前检查过的 0、127 例不入池；同一文本组整体保留，禁止后续训练使用。没有模型推理或生成分数。SPICE 文本一致不是图同构判定，且此数据属于跨域来源，标注适配/插图来源尚待核查，**不能把这个清单声称为已就绪的独立 PCB 测试集**。

WSL Ubuntu22.04 的 Docker Engine 29.3 已载入用户提供的 `edarace-agent-base:ubuntu24`，并成功构建 `graphschematic-recognition:20261004`。镜像 ID 为 `sha256:0efc92976ef6189f56d162f424f52f5213bfd7edea4d6f4e7c5ab96a03f65aef`。新增 `.dockerignore` 排除原始数据、官方资料、虚拟环境、训练输出和分块下载缓存，补齐通用 Dockerfile 中的新默认语义库文件。

首次在线 pip 构建因下载超时失败；改为从 PyPI 元数据选取 Python3.12 / Linux x86_64 wheel，逐文件核验官方 SHA256，13 包共 139,051,448 字节，再用 `pip --no-index` 成功安装。`scripts/prepare_linux_wheels.py` 可准备本地依赖；开发 ZIP 仅保留准备脚本和说明，不包含这些 wheel 或官方基础镜像。没有本地 wheel 清单时构建回退到在线安装，系统依赖步骤仍可能需要网络。

容器以 `--network none` 和只读 `/input` 挂载运行官方形式入口，公开示例 1 页完成，耗时约 7.4 秒，`final_results/guide-schematic_Pred.json` 与 Windows 新默认逐项完全一致，LLM 调用为 0。证据见 `output/container-validation.json`，可用 `scripts/verify_container_output.py` 复核已有输出。这是实际构建和单例离线运行验证，完整官方 runner、私有评测和 decision_log 规范仍未验收，不能外推全量时限。

67 项功能测试涵盖元件框/引脚/网络分离、缺乏极板证据时弃权、裁剪外 OCR 不影响图像、原有拓扑与受约束纠错等。正式技术报告、官方评分器核验和在线 VLM 增益仍待完成。

## 运行与回退

```powershell
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --output output/current.json --diagnostics output/current-diagnostics.json
# 只关闭本轮输出框修正
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --component-box-mode detector --output output/detector-boxes.json
# 回退此前完整默认：原主引脚 + 共识；不使用语义库和输出框修正
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --pin-model models/pins/pin-heatmap.onnx --pin-ensemble models/candidates/official-pins/pin-heatmap.onnx --semantic-library none --component-box-mode detector --output output/previous-default.json
```

下一阶段优先做四边结构化 IC 端点召回、引脚编号/名称 OCR 的边界与次序约束；其次是未分配 designator 与漏检。保留 Geometry-to-Electrical、Topology > Geometry > Color、飞线合并和电路先验；新的局部指标必须用整图实测验证。
