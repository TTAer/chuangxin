# 验证阶段执行状态

2026-10-06。按用户最新评估停止新模型、训练数据扩充、阈值和Agreement规则试验。RC0冻结；PPN/MST归档为研究证据；学习型拓扑自动接入保持NO_GO；没有通过门禁的拆分不解释为“拆分无效”。本轮推进审核入口和正式评测器取证，不代替人类审核。

## 四项主线进度

| 指标 | 当前证据状态 | 下一项必需输入 |
|---|---|---|
| Reviewer A | 0/25，未收到完成JSON | 真实审核者A独立观察并导出25例记录 |
| Reviewer B | 0/25，未收到完成JSON | 不同的真实审核者B独立完成，不提前讨论 |
| verified designs | 0；Directional Pad各门禁仍false | 双人L1观察、来源/revision/许可与设计家族排除证据 |
| official evaluator equivalence | unresolved | 官方评分代码及版本，或官方执行的可比输入/输出与逐项诊断 |

0/25是已知执行状态与工作区输入状态，不推断工作区外是否存在私人草稿。没有收到真实输入时，一致率为null，不报告0%或100%，也不改CoordinateConventionFrozen。

## 现有审核入口已启动

协议锁通过，25张图像裁剪通过HTTP取回并与原SHA一致。两页返回HTTP200。这验证页面和证据资产可访问，不等于真实审核、浏览器交互全流程认证或标注完成。没有改25例、页面代码、协议或图像。

- [Reviewer A：独立空白审核](http://127.0.0.1:8767/pin-geometry-review-v2/index.html?pass=A)
- [Reviewer B：独立空白审核](http://127.0.0.1:8767/pin-geometry-review-v2/index.html?pass=B)

服务仅监听本机127.0.0.1:8767，窗口隐藏；日志保存在`output/validation-transition/review-server.*.log`。两位审核者各自打开空白页、填写真实ID、保持参考隐藏，完成观察再揭示参考，各自导出JSON。不要刷新丢失尚未导出的草稿；两位不要共享记录。

将导出结果放入例如`data/research/human-review-A.json`、`data/research/human-review-B.json`后，执行已有入口：

```powershell
.venv/Scripts/python.exe scripts/prepare_review_handoff.py --reviews data/research/human-review-A.json data/research/human-review-B.json
```

不要删除或重建现有协议锁。收到真实审核后才计算samepin、side、conclusion、point、uncertain/continuous-no-boundary、域/方向拆分，再解释Semantic-Eligible P5/Coverage。当前不新建统计工具或替人填写记录。

整页第二步继续使用[Directional Pad现有审核材料与表](FIRST_VERIFIED_PAGE.md)。先完成Pilot，再处理整页双人L1；CAD只能作为L2，不填充不可见的pin真值。六个阶段状态保持false，未创建Freeze-A首次推理清单，没有独立集推理。

## 官方评测器取证

重新取得[2026赛题六官方公开响应](https://cpipc.acge.org.cn/cw/getContestWebsiteQuestionById?questionId=2c9080179fb07f4f019fb8529dc21ff1)，保存原始响应和SHA。公开内容仍给出指南下载、服务器指定位置与技术群信息，未提供可直接调用的评分脚本链接。未登录账号、加入群或向他人发送消息。

检查本地三个官方数据ZIP的32/33/636个成员，未发现名称包含evaluator/eval_/evaluate/scor的文件；这是文件名检索，不声称证明所有压缩成员都不存在任何评分逻辑。现有基础镜像说明只明确agent接口与输出目录，decision_log字段仍引用未取得的EDARace手册。

[PCBnet作者库](https://github.com/ECNUACRush/PCBnet/)公开网页当前只列README，说明源码尚未发布。其文档中的计划目录不是已经存在的官方比赛评测器。GitHub树API此次返回403限流，该失败如实记录，没有用推测补齐仓库内容。作者论文连接精度不能替代本赛题官方评分等价性。

| 已确认的规则 | 依据 | 仍未由正式代码确认的部分 |
|---|---|---|
| 四项权重30/25/35/10；默认bbox20px、pin/line5px | 指南第6–8页 | 边界是否包含等号、匹配并列时的裁决 |
| 非box pin number/name不评分 | 本地Q&A评测问题2 | 非box身份重映射怎样进入NetF1 |
| Vcc/GND按坐标匹配、GND序号不影响评分 | Q&A评测问题5/6 | 是否放宽所有∅前缀非电源元件；bbox/point具体匹配方式 |
| box型号Name、R/C value使用图像文本 | 指南与Q&A数据问题9/11 | text错误是否影响后续pin/net关联 |
| line端点匈牙利匹配、只在matched net内比较 | 指南第8页、Q&A评测问题3 | 端点距离代价/方向、空net、重复/切分线段边界处理 |
| net为pin set并匈牙利匹配 | 指南第8页 | assignment目标是交集、F1或其他代价；micro/macro聚合与空集定义 |

本地`evaluate_qa`对所有∅前缀放宽身份、按net-set F1最大化配对、使用micro计数及≤阈值，均只能保留为代理选择。这里登记歧义，不据此修改冻结评估口径。

在不修改代码的情况下执行7个最小代理样例：恒等、非box名称变化、value大小写变化、pin恰好5px与5.001px、线段端点反向、同样几何拆成两段。后者NetF1=1、NetLineF1=2/3，展示线段表示对代理分母的影响。所有official_result=null；代理自测不能证明与官方一致。机器证据：`output/validation-transition/proxy-contract-probes.json`。

## 最终包一致性

只检查既有`submission/graphschematic-rc0-evaluation.zip`，没有重新打包或提交。53个冻结运行源码/所选权重在工作区和ZIP内均一致；Agreement归档351条SHA也全部通过。原ZIP SHA256：`04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4`。

这只是工程包一致性；其manifest中的103项测试是打包时历史记录，当前最新167项测试另有记录，不能改冻结包内历史数值。当前不重复昂贵容器批验；没有代码/权重变化，不需要用重复推理制造进度。正式赛题服务器、官方evaluator、独立精度门禁仍未验收，包不能称为最终赛事通过品。

## 已完成与未完成

### 首次独立评分溯源要求

固定记录first_inference_timestamp、pipeline_hash、manifest_hash、review_hash、evaluator_hash、scoring_config_hash。后两项分别绑定实际评分程序/适配器内容与实际评分配置内容；评分口径改变须新增评分版本，保留旧结果及作废原因，不恢复盲测身份。当前尚未进行首次独立评分，这些实际运行字段不预填。

### 算法研发暂停与恢复条件

RC0定位为“Frozen, reproducible, deployable baseline pending independent accuracy validation”。在以下任一真实证据出现前，保持新模型、训练扩充和阈值/拓扑自动接入试验暂停：A/B双人审核完成；至少一个页面级审核状态有真实证据支持完成；取得正式evaluator或可比官方逐项结果；Freeze-A产生首个verified design。条件出现只允许重新评估研发是否有必要，不自动晋级RC1、解冻坐标语义或授予Commit。

若取得正式评分器，先对固定RC0与历史关键分支的已有预测统一重评分，记录evaluator/config哈希并建立proxy→official偏差表，再决定算法工作。没有可比输入/输出时不宣称等价。

本轮完成：既有审核入口启动、25张锁定裁剪校验、正式评测资料检索与歧义登记、冻结源码/ZIP一致性检查。未完成：真实A/B审核、Directional Pad认证、独立verified设计、官方评测等价性。需要真实人类输入和官方评分资料后才能继续推进这些状态。
