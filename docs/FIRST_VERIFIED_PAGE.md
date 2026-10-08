# 首张完整审核页：Directional Pad

2026-10-06。状态：选定人工审核对象，尚未verified。实际双人审核未完成；本页没有运行RC0，未产生独立精度结论。

选择本页用于先走通一个完整流程：没有DNP、没有多unit实例，93/93个可见非零CAD pin段与指定PDF页匹配，已有原生PDF文字和本地许可文件。相比文字层缺失的小页，更便于逐项核对可见身份。上述条件只支持审核优先级，不证明整页revision或真值正确。

## 实际审核材料

先完成25例独立盲审，再处理本页整页审核。观察L1时先看原图，不先阅读CAD答案。以下均是工作区现有文件，无需下载新来源。

| 材料 | 路径 |
|---|---|
| 原图，L1观察入口 | [page.png](D:/创芯EDA/gpt方案/data/processed/FreezeA-A1-candidates/sparkfun--SparkFun_Qwiic_Directional_Pad--e395a48889/page.png) |
| PDF第1页，revision参考 | [原始PDF](D:/创芯EDA/gpt方案/data/raw/FreezeA-manufacturer/sparkfun--SparkFun_Qwiic_Directional_Pad/docs/assets/board_files/SparkFun_Qwiic_Directional_Pad_Schematic_v1_1.pdf) |
| CAD，L2来源 | [原始CAD](D:/创芯EDA/gpt方案/data/raw/FreezeA-manufacturer/sparkfun--SparkFun_Qwiic_Directional_Pad/Hardware/SparkFun_Qwiic_Directional_Pad.kicad_sch) |
| 整页审核表 | [review-template.json](D:/创芯EDA/gpt方案/output/freeze-a1-calibration/70f1adb2f703/review-template.json) |
| L1/L2分层记录 | [two-layer.json](D:/创芯EDA/gpt方案/output/freeze-a1-two-layer/70f1adb2f703.json) |
| 几何段匹配证据 | [revision-audit.json](D:/创芯EDA/gpt方案/output/freeze-a1-revision-audit/70f1adb2f703.json) |
| 本地许可声明 | [LICENSE.md](D:/创芯EDA/gpt方案/data/raw/FreezeA-manufacturer/sparkfun--SparkFun_Qwiic_Directional_Pad/LICENSE.md) |
| 来源/历史家族审核 | [family-review.json](D:/创芯EDA/gpt方案/data/research/freeze-a1-family-review.json) |

## 特别容易误计的内容

CAD有66个放置实例，其中32个带可见非零pin段候选、26个带隐藏pin元数据。这两类不是互斥分类，不能相加得到可见器件总数。电源符号、logo、机械标记和普通元件均需按可见性和标注接口规则区分，不能将66直接填入L1。

原生PDF文字包含U4/PCA9554PW和pin文字，但部分文字重复，I²C还出现编码异常。因此文字抽取只能帮助定位，不能依据词出现次数认定多个器件，也不能代替像素审核。pin编号和名称只能填写实际可见的内容；CAD中存在而图像不可见的属性留在L2。

Directional Pad与Navigation Switch当前保守归为同一近似PCA9554设计家族；不能当作两个独立设计。历史训练/调参语料的家族排除证据尚缺。本地license文件已存在，但来源和适用范围审核标记仍为false，此处不将文件存在视为审核完成。

## 两位实际审核者要完成的事项

1. 分别记录可见器件key、可见pin号/名称、可观察pin笔画及contact；无法观察的内容保留unknown。每份审核保留实际审核者ID和证据文件，不复制另一人的观察。
2. 观察记录完成后对照PDF/CAD，确认整页revision、每个放置实例以及隐藏pin、DNP、unit、机械/图形元素的处理。几何93/93仍不能替代文字和器件核查。
3. 将各自完成记录路径填入既有整页审核表；L1记录实际像素证据，L2保留CAD设计属性。现有L1为空，不能批量从L2自动填充。
4. 补齐source/license范围与历史设计家族排除证据。所有完成标记必须引用真实证据；若历史来源无法排除，该页只能保留候选身份。
5. 查看双方分歧并作显式reviewed decision。坐标语义不唯一时可保持unresolved；不要为达成verified而强行选body/tip/contact。当前全局坐标语义仍未冻结，不创建首次推理manifest。

第一目标是完整审核一页、暴露真实流程问题。只有全部既有门禁满足，才能计为verified=1；人工完成某个子任务不能单独晋级。随后再推进另外三个优先来源。

## 25例Pilot之后的固定路径

先报告samepin、side、conclusion、point四项一致率，包含分子/分母、未判定、continuous-no-boundary，以及域/方向拆分，再解释Semantic-Eligible P5及合格例/25覆盖率。

| 真实审核结果 | 后续行动 |
|---|---|
| 身份/方向高度一致，物理点结论稳定且域间无明显冲突 | 人工复核分歧后制定50–80例确认抽样；不直接宣布官方定义或自动冻结 |
| 身份稳定，但body/tip/contact不一致 | 继续关闭unresolved自动新增；身份/OCR仅保留证据或Proposal用途，不做snap |
| 身份或方向本身不稳定 | 维持保守弃权，记录局部raster信息不足的限制 |

预登记阈值见[审核执行单](HUMAN_REVIEW_HANDOFF.md)，阈值只筛选下一步，不自动切换CoordinateConventionFrozen。RecoveredGT暂不扩实现，证据不足仍null。没有实际审核输入前，不开发新算法、规则、模型或页面。

## 盲测不可恢复的记录规则

Freeze-A-Pilot至少4个verified设计，只做灾难性泛化回归检查，禁止调阈值、规则、库或个案修复；Freeze-A-Official仍至少10个verified设计。Pilot与Official用途及清单必须区分。

未来首次推理及独立评分必须记录first_inference_timestamp（带时区）、pipeline_hash、manifest_hash、review_hash、evaluator_hash、scoring_config_hash，并证明推理发生在标注冻结之后。本轮这些字段均未产生，不填假时间或空签名。evaluator_hash绑定实际执行的评分源码/可执行文件与适配器清单；scoring_config_hash绑定实际使用的坐标、匹配阈值、类型映射、聚合方式及边界处理配置，不能只填写工具名称或版本号。评分器或配置变化时保留原记录并另存新评分版本，不覆盖首次结果，也不把重评分当成新的首次盲测。结果一旦被查看，该设计即为observed evaluation sample，即使没有调参，也不能重新声称完全blind；若用于调参，则转为development，不能参加后续首次独立验证。Official清单需明确排除已观察的Pilot设计或另列为非盲补充评测。

## 当前进度

### 分阶段状态，不降低最终门槛

页面本身的审核进度与全局坐标定义分别记录。`evaluation_ready`才计入正式verified设计；以下子状态均不能代替最终门禁，也不因文件存在或几何对齐自动变成true。

| 子状态 | 当前 | 完成所需证据 / 当前卡点 |
|---|---|---|
| source_verified | false | 来源、许可适用范围、整页revision显式审核；已有本地文件和93/93几何筛选，但未完成来源认证 |
| page_content_verified | false | 两位实际审核者分别确认visible key、pin文本、hidden/DNP/unit等处理；当前无实际完成记录 |
| geometry_reviewed | false | 双人实际观察pin stroke/body/tip/contact，记录不可观察与分歧；当前L1为空 |
| family_exclusion_verified | false | 源仓库/revision lineage、结构fingerprint、历史训练/开发provenance对照；历史家族排除证据仍缺 |
| coordinate_convention_ready | false | 全局证据及显式reviewed decision；不是单页完成标记，也不能由一致率阈值自动触发 |
| evaluation_ready | false | 上述各项及既有冻结门禁全部满足；当前页面级和全局级两类证据均缺，不是仅卡在坐标定义 |

以后页面内容或几何审核完成时，可以单独记录相应子状态及真实证据路径，即使坐标定义仍未冻结。这里的状态表仅用于进度管理，既有程序门禁与正式verified计数保持不变。

### 异常登记与处理

实际审核者发现每个例外时，登记独立异常ID、具体器件/pin/区域、原图证据、两份观察记录、差异类型、处理结论、处理者与证据路径。`N_exceptions`只统计真实登记的异常；尚未完成审核时为null，不报0。已解决、未解决和无法观察分别计数。完成勾选不能覆盖未解决例外。

以下是已经在材料检查中发现的风险，属于准备记录，不是人类审核异常或已完成resolution：

| 准备记录 | 现有证据 | 当前处理 |
|---|---|---|
| PDF抽取中部分器件/pin文字重复 | 本页原生PDF文字层 | 待像素核对，禁止按抽取次数增加器件 |
| I²C抽取出现Unicode异常 | 本页原生PDF文字层 | 待像素核对，保留原文与规范化文本的对应证据 |
| CAD隐藏pin不等于可见pin | L2中26个实例含隐藏pin元数据 | 仅保留设计元数据；可见性仍由实际L1观察决定 |

实际异常登记模板（由审核者填写，不预填为完成）：

| exception_id | 器件/pin/区域 | 类型 | severity | scope | reviewer A证据 | reviewer B证据 | resolution / unresolved | adjudicator及证据 |
|---|---|---|---|---|---|---|---|---|
| 待登记 | — | — | — | — | — | — | — | — |

severity使用informational / affects_L1 / affects_revision / blocks_verification；scope使用local_pin / component / page / source。由实际证据确定，不把编码显示问题自动归为无影响，也不将准备风险自动计为真实异常。

重点检查visible key与CAD reference不符、可见pin name与CAD不符、连续笔画无明确boundary、隐藏pin不可见、logo/机械元素的标注范围、重复文字和编码异常。这里列的是检查类别，不代表已经发现每类异常。

### Pilot / Official数量规划

目标分配为4个Pilot-only与至少10个Official-blind，合计至少14个合格且相互独立的设计。这里是后续资源规划，不是当前已获得数量，也不是立即扩大下载范围的指令。先完成25例和第一张整页真实审核，再根据实际证据推进资源补充。

13张候选页不等于13个独立设计，Directional/Navigation近似家族不能重复计数。Pilot查看过的设计不能凑入10个首次盲测Official；若来源不足，报告实际blind数量与未达门槛，不混算或下调门槛。当前仍只有0个evaluation_ready设计。

| 项目 | 当前 |
|---|---|
| 实际审核者A完成 | 0/25 |
| 实际审核者B完成 | 0/25 |
| 完整verified设计 | 0 |
| CoordinateConventionFrozen | false |
| RC0 | FROZEN / DEPLOYABLE / 尚未独立验证精度 |

本轮仅核对既有证据、确定首张审核页并收紧执行顺序；没有增加测试数、改V2或重训。用户此前确认实际审核尚未完成，工作区也未发现约定的human-review JSON。下一步的必需输入是两位真实审核者分别完成的记录。
