# 双人审核执行单（2026-10-06）

当前用户确认两位审核者均尚未完成。实际有效双人审核为0，verified designs为0，CoordinateConventionFrozen=false。RC0尚缺独立精度验证，本轮没有训练、阈值调整、坐标修正或独立集推理。

## 先完成25例 Convention Pilot

现有V2工具和25例保持不变。协议、页面、审核代码和图像/裁剪SHA已经锁定于 `output/review-handoff/convention-pilot-protocol-lock.json`。这是审核协议锁，不是坐标语义冻结，也不是人工认证。空白模板可填写，因此不纳入不可变锁；导出的记录仍须绑定每例指纹。

1. 两位不同的实际审核者独立操作，不看彼此记录，不边讨论边审核。各自使用自己的真实 reviewer ID。
2. 审核者A打开现有页面 `http://127.0.0.1:8767/pin-geometry-review-v2/index.html?pass=A`；审核者B使用 `?pass=B`。本地服务未运行时，在项目目录执行 `.venv\Scripts\python.exe -m http.server 8767 --bind 127.0.0.1 --directory output`，并保持终端运行。
3. 默认隐藏GT和prediction。先记录物理pin判断、实际side、可观察body/tip/contact及certainty。无法观察则明确填写uncertain或continuous-no-boundary，不强行点击。
4. 观察完成后才揭示参考；进入下一例重新保持参考隐藏。勾选完成再导出，各自保存独立JSON，例如 `data/research/human-review-A.json` 与 `data/research/human-review-B.json`。允许提前导出草稿，但草稿不等于完成。
5. 两份文件均完成后，在项目目录执行：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_handoff.py --reviews data/research/human-review-A.json data/research/human-review-B.json
```

该入口先核对协议锁和源文件，再调用已锁定的V2导入器核对GT、原始身份来源及审核记录。文件指纹不一致时停止，不自动重建协议锁。不要删除锁以绕过不一致。

## 审核后统计的定义

以下定义在读取真实审核前登记。当前无审核，指标保持null，不将0例解释为100%一致。

| 指标 | 分母与一致条件 |
|---|---|
| A_samepin | 双人均给出非uncertain分类的案例；分类相同且同pin时观察pin号一致；另报告未判定数量 |
| A_side | 双人均给出实际非unknown side的案例；side相同；另报告unknown数量 |
| A_conclusion | 双人均clear的案例；conclusion一致；不确定项另列，不混入有效一致结论 |
| A_point | 双人conclusion相同且为body/tip/contact、均clear并均有该点的案例；两点距离≤3px；continuous-no-boundary不适用 |
| Semantic-Eligible P5 | 使用已锁V2的身份、盲审、side和坐标门禁；合格案例中预测点到精确身份GT≤5px |
| Coverage | 合格案例/本轮全部25例，并分别报告各域计数；不将25例试点外推为687候选的总体覆盖率 |
| rn、rt | 分别为en/Lpin、et/Lpin；Lpin仅来自两位审核者实测body→tip，保留各自长度及区间；不可观察则null |
| RecoveredGT | 需精确身份及旧锚点覆盖证据，并按GT pin去重；证据未绑定时null，不能直接沿用最近点几何命中数 |

四项分解一致率、rt和RecoveredGT尚未接入已冻结V2导入器，收到真实审核后在独立后处理报告中计算，不修改审核页面。25例只能用于协议和语义试点，不能证明官方坐标定义。

samepin≥90%、side≥90%、clear conclusion≥80%仅作为下一阶段审核扩展的预登记筛选条件，必须同时展示分母、各域/方向覆盖、分歧实例和不确定率。不能靠这些比率自动设置CoordinateConventionFrozen=true。若呈现可信的统一模式，再确定50–80例确认方案；若域间矛盾或观察不可唯一恢复，则继续关闭unresolved自动新增。当前不扩样。

## 只审核四个来源

执行顺序更新：先完整走通[Directional Pad首张审核页](FIRST_VERIFIED_PAGE.md)，再逐页推进其他三个来源，不平均铺开。本页仍未verified。

详细路径见 `output/review-handoff/source-risk-handoff.json`。其中提供整页审核表、L1/L2分层文件、revision证据的现有路径，不创建新的审核界面。

| 顺序 | 来源 | 可见pin段匹配 |
|---|---|---|
| 1 | Directional Pad | 93/93 |
| 2 | GNSS Flex小页 | 39/39 |
| 3 | AP3012 | 50/50 |
| 4 | RFID USB-C | 93/93 |

整页源风险筛选固定为≥98%人工优先、90%至<98%核查revision、<90%暂停；零分母或异常计数拒绝。BlueSMiRF为50/89，暂停。100%几何段匹配不认证文字、器件、网络或revision一致。

对每个优先来源，实际审核者须确认整页revision、可见component key、可见pin号/名称、hidden/DNP/unit处理、L1像素观察、独立第二审核；另补齐license/source证据与历史设计家族排除证据。不可见CAD属性只能留在L2，不能填入L1。填写既有审核表，不伪造人类ID或完成标记。

Pilot规划要求≥4个实际verified设计，仅用于固定RC0的泛化风险go/no-go，不能调参或宣称正式精度；Official仍要求≥10个verified设计。若Pilot被用于调参，则标记为开发集并失去盲测资格。此次只登记用途，未创建Pilot冻结manifest、未降低现有正式门禁、未授权推理。坐标语义冻结和全部证据门禁满足后，才能考虑创建对应清单并首次推理。

补充：Pilot结果一旦被查看，即便未调参，也成为observed evaluation sample，不能重新标记完全blind。未来需记录首次推理时间、pipeline/manifest/review hash；Official首次独立验证应排除已观察Pilot或将其单列为非盲补充。

## 本轮验证与待办

已新增协议锁篡改/缺失拒绝测试，以及revision阈值边界和BlueSMiRF筛选测试。全套162项测试通过；RC0/V1/R的53/89/215个冻结文件SHA均不变。用两份空白模板实测导入，接受审核0、配对0，未误计为人工结果。实际人类审核是下一步必需输入；代理可以检查、统计和汇总，但不能代替这两位人类。
