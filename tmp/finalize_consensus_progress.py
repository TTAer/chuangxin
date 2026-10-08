import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))
def write(path, data):
    (ROOT / path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

old55 = read('output/benchmark-graph-topology-v2-55/summary.json')
new55 = read('output/benchmark-pin-consensus-55/summary.json')
old10 = read('output/benchmark-official-baseline/summary.json')
new10 = read('output/benchmark-official-pin-consensus-final/summary.json')
live = read('output/pin-consensus-live-check/summary.json')
def paired(before, after):
    lookup = {str(r.get('case', r.get('id'))): r for r in before['records']}
    deltas = {str(r.get('case', r.get('id'))): r['Score_Accuracy_SingleCase_proxy'] - lookup[str(r.get('case', r.get('id')))]['Score_Accuracy_SingleCase_proxy'] for r in after['records']}
    return {'deltas': deltas, 'improved': sum(v > 1e-9 for v in deltas.values()), 'regressed': sum(v < -1e-9 for v in deltas.values()), 'unchanged': sum(abs(v) <= 1e-9 for v in deltas.values())}
assert live['passed'] and new10['completed'] == 10 and new55['completed'] == 55
write('output/ablation/pin-consensus.json', {
    'method': 'TopologyPreservingPinConsensus', 'official_equivalence': False,
    'old_proxy_55': old55['mean_metrics'], 'ensemble_proxy_55': new55['mean_metrics'],
    'paired_55': paired(old55, new55), 'accepted_pin_updates_55': new55['accepted_pin_updates'],
    'old_qa_gt10': old10['mean_metrics'], 'ensemble_qa_gt10': new10['mean_metrics'],
    'paired_gt10': paired(old10, new10), 'cross_style_live': live,
    'evaluation_notes': ['55 cached image-only observation replays; 3 cross-style live predictions matched exactly', '10 updated official public GT cases evaluated end to end', 'public data used for method selection; not an independent final test', 'proxy profiles differ; scores cannot be compared across the two datasets']})
config = read('models/pipeline.json')
aux = 'models/candidates/official-pins/pin-heatmap.onnx'
config.update(version=3, pin_ensemble=aux, selected_run='output/benchmark-pin-consensus-55', comparison='output/ablation/pin-consensus.json')
config['scores']['benchmark-pin-consensus-55'] = new55['mean_metrics']['Score_Accuracy_SingleCase_proxy']
config['model_sha256'][aux] = hashlib.sha256((ROOT / aux).read_bytes()).hexdigest()
config['selection_notes'] = 'Original primary pin model plus topology-preserving consensus; 55-page replay: 1 improve/1 regress/53 same; updated GT10: 3 improve/0 regress/7 same. Public development selection, not official validation.'
write('models/pipeline.json', config)

report = '''# GraphSchematic 项目进度报告

更新：2026-10-04。本阶段优先提高自动识别与可复现性，暂缓正式比赛技术报告。

项目已具备图片到比赛结构 JSON 的自动运行链路和批处理入口，目前属于可运行的开发原型。整图元件、引脚和网络识别精度仍偏低，尚未完成官方评测、Docker 构建与比赛环境验收，不能认定为最终比赛成品。

## 本次完成的改进

实现 TopologyPreservingPinConsensus：保留原主引脚网络，增加官方训练数据适配的辅助网络。只有相同引脚编号及名称、两模型端点距离不超过 6 像素时才提出坐标均值；几何双图检查电气分区，最终检查网络成员与物理导线签名，变化则回退。已经确认的全局关联在融合后重放，不在此步骤重新提出连接。坐标最多移动 3 像素；此方案采用手工一致性门限，不代表概率置信度校准。

该模块已接入单图、批处理和竞赛入口，默认开启；`--pin-ensemble none` 可关闭。辅助模型没有直接替换原主模型，因为其单独运行在 55 页开发集上略有退步。融合与语义库属于独立步骤；额外启用语义库后的结果需要单独评估。

修复了缓存回放与实时推理在融合后再次关联时可能不一致的问题。3 张彩色/灰度、不同规模的实际图片输出与缓存融合完全一致。完成 56 项功能测试，重新运行新版官方 10GT 全部图片，无失败；该次总耗时约 75.2 秒，不据此外推全部竞赛案例时限。

## 架构进度

| 模块 | 当前实现与限制 |
| --- | --- |
| 视觉感知 | 多来源 YOLO11n、OCR、tip/base 双热图引脚模型；密集 IC 文本和引脚编号仍是主要瓶颈 |
| Geometry-to-Electrical | pin/wire endpoint、junction、label 几何节点及实际导线边；电气超图导出网络成员；同名标签飞线合并保留，不制造物理线段 |
| Topology Perception | 灰度线结构、T 接点/交点证据与局部神经网络；不确定接点阻止贸然合并；按 Topology > Geometry > Color 使用证据，颜色只作为特征 |
| Constraint-Guided Global Association | 有界束搜索、像素/距离/方向证据、电源冲突和无源器件短接约束；目前为手工评分的近似推理，尚未训练 GNN 关联器 |
| 引脚融合 | 本次新增：双模型一致性与电气签名门控，默认开启，保留单模型消融路径 |
| 电路先验 | 官方元件库编译为 337 个条目、3838 个唯一引脚编号；歧义引脚名放弃推断；可选启用，尚不能证明普遍增益 |
| 大模型纠错 | 白名单提议、同一观测校验、硬约束、超时和本地回退已实现；官方在线服务未配置，尚无真实在线增益结果 |
| 竞赛入口 | `agent_system -input … -output …`、`final_results/*_Pred.json`、批处理隔离和日志已实现；详细 decision_log 官方规范尚缺，镜像尚未构建验证 |

## 数据与训练进度

| 来源 | 实际规模及用途 |
| --- | --- |
| PCBnet | 300 例，216 PNG + 84 JPG；去重训练 241 页，清理近重复后开发验证 55 页；旧模型选优仍曾使用原 56 页划分 |
| AITEE | 828 张去重图片，662 训练 / 166 验证；教育电路风格，与 PCB 主域分别评估 |
| 官方公开训练图 | 200 页，其中 160 页与现有训练、40 页与现有验证像素重合，不能算作新增 200 页独立图像；新增官方标注适配引脚裁剪 7071 训练 + 1906 验证 |
| 官方新版 10GT | 使用 0930 更新标注；10 张与已使用训练图无像素重合，但未证明设计来源完全独立；只用于开发评价与方法选择，未用于引脚训练 |
| OmniSch | 实际获取 5 个设计、50 个渲染样例，未获取完整数据集；许可尚未明确，未加入训练 |

已完成 PCB 检测训练、多来源适配、主引脚训练、官方引脚候选微调和 Topology Perception 训练。裁剪定位 F1、检测 mAP 与以下整图代理分是不同指标，不能互相替代。原始第三方数据不放入开发包。

## 当前效果

以下均为本地代理评分，尚未得到官方 evaluator 等价验证。两个数据集使用不同匹配规则，不能横向比较分数；公开开发数据参与了方法选择，不能视为独立测试。

55 页 PCB 开发集：本次用已保存的图像观测回放，另外以 3 页实际图片确认输出一致。

| 配置 | Component F1 | Pin F1 | Net F1 | Line F1 | 加权代理分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 上版默认 | 0.21582 | 0.09769 | 0.19660 | 0.07206 | 16.51843 |
| 当前默认融合 | 0.21582 | 0.09774 | 0.19660 | 0.07206 | 16.51966 |

1191 个引脚坐标更新被接受；1 例改善、1 例退步、53 例持平，平均增益很小。网络与线段代理指标保持一致。保留逐例差异，后续应重点分析退步案例 0107 和改善案例 0140。

新版官方 10GT：全部从图片重新推理，按 Q&A-informed 本地匹配规则评估。

| 配置 | Component F1 | Pin F1 | Net F1 | Line F1 | 加权代理分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 上版默认 | 0.37886 | 0.11816 | 0.07418 | 0.09221 | 17.83821 |
| 当前默认融合 | 0.37886 | 0.12849 | 0.08198 | 0.10056 | 18.45322 |
| 候选引脚 + 语义库（可选） | 0.37886 | 0.13296 | 0.09504 | 0.11539 | 19.17035 |

当前默认相对上版提高约 0.6150 分，3 例改善、0 例退步、7 例持平。可选配置的 19.17035 是此前同一组公开 GT 的开发结果，未证明在 55 页或隐藏赛题上更好，因此未作为默认。

## 复现与交付

在项目根目录、安装推理依赖后运行：

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m pcbparse infer data/examples/guide-schematic.png --output output/current.json
.venv/Scripts/python.exe agent_system -input data/examples -output output/current-agent
.venv/Scripts/python.exe scripts/evaluate_pin_consensus.py
.venv/Scripts/python.exe scripts/benchmark_official.py --pin-ensemble models/candidates/official-pins/pin-heatmap.onnx --output output/official-consensus-recheck
```

缓存回放命令需要本机已有观察文件；官方评测命令需要已解压的官方资料。这些原始数据与大型实验输出不包含在开发包内。开发包 `submission/pcbparse-development.zip` 包含源代码、推理权重、测试、文档和消融摘要；`BUILD_MANIFEST.json` 及外部 manifest 提供逐文件哈希。它是开发交付物，尚不是已验证的最终参赛镜像。

## 后续优先事项

1. 针对错误位号、密集 IC 引脚编号和端点定位建立失败案例训练与评价，提高低水平的整图 Pin/Net 指标。
2. 将当前手工关联评分升级为可训练的几何图消息传递与电气超图一致性模块；先建立无泄漏的数据划分和监督样本，再做公平消融。
3. 扩充真实 PCB 风格且许可明确的数据，审计设计级近重复，建立方法选优之外的冻结测试集。
4. 用比赛允许的模型配置实际在线纠错，测量准确率、调用量和时延；取得完整日志规范与官方 evaluator，验证容器和批量时限。
5. 精度与环境验收通过后，再整理正式比赛技术报告和最终提交物。
'''
(ROOT / 'docs/PROJECT_PROGRESS_2026-10-04.md').write_text(report, encoding='utf-8')
status = ROOT / 'docs/PROJECT_STATUS.md'
s = status.read_text(encoding='utf-8').replace('功能测试现有 26 项。', '功能测试现有 56 项。')
s = s.replace('## 当前推进', '最新完整进度见 [2026-10-04 项目进度报告](PROJECT_PROGRESS_2026-10-04.md)。当前默认新增保持电气拓扑的双模型引脚融合；10GT 本地代理分 18.4532，55 页回放 16.5197。均非官方评分。\n\n## 当前推进', 1)
status.write_text(s, encoding='utf-8')
readme = ROOT / 'README.md'
s = readme.read_text(encoding='utf-8')
s = s.replace('## 自动识别', '最新进度、数据审计与实测限制见 [项目进度报告](docs/PROJECT_PROGRESS_2026-10-04.md)。默认增加保持电气拓扑的双模型引脚融合；`--pin-ensemble none` 可关闭。新版官方 10GT 本地代理分为 18.4532，尚非官方评分。\n\n## 自动识别', 1)
s = s.replace('提升至 16.5184', '提升至 16.5184；当前引脚融合回放为 16.5197（1 例改善、1 例退步、53 例持平）')
readme.write_text(s, encoding='utf-8')
graph = ROOT / 'docs/GRAPHSCHEMATIC.md'
graph.write_text(graph.read_text(encoding='utf-8') + '\n\n## 2026-10-04 拓扑保持引脚共识\n\n新增 `pcbparse/pin_consensus.py`：两个定位模型只对同名同编号、距离 ≤6 px 的端点提出均值，并用几何图分区和最终网络/导线签名约束接受。复用融合前已确认的关联，不在融合步骤提出新的网络连接。不是训练式 GNN，也没有置信度校准。默认配置保留原主模型，官方适配模型作为辅助；可用 `--pin-ensemble none` 消融。55 页回放只有很小平均增益，1 例退步；10GT 端到端代理分为 18.4532。完整记录见 [项目进度报告](PROJECT_PROGRESS_2026-10-04.md) 与 `output/ablation/pin-consensus.json`。\n', encoding='utf-8')
print('Updated configuration, ablation, and progress documents.')
