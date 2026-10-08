# 独立几何审核 V2 与整页 revision 核查（2026-10-06）

本轮继续处理真值和评测基础设施，没有训练、改阈值、snap、推理 Freeze-A 或更改 RC0。交付 25 案例的独立审核 V2、Semantic-Eligible P5 / 归一化误差计算、整页 CAD/PDF pin 几何核查、五个优先来源和 observed/design 两层证据。160 项测试通过，冻结文件逐 SHA 校验通过。

## 审核工具已经可用，但人工记录仍为零

打开 `output/pin-geometry-review-v2/index.html`，或本机 `http://127.0.0.1:8767/pin-geometry-review-v2/index.html`。仍使用原来的 25 个案例，没有按开发误差扩大/重选样本：GT10 5 个点/2 张源图，PCB55 12 个点/9 张源图，Freeze-A1 8 个点/2 张源图。图和点数量不等于独立设计数。

V2 默认只看像素，隐藏 GT、预测和 CAD 属性；支持先记录观察点再主动揭示参考。第二遍入口使用反向案例顺序和空白表，不加载第一遍记录。两位实际审核者需要分别填写自己的标识并独立导出。

审核字段现在包括：

- `body/tip/contact` 实际观察点，可以留空。
- `certainty`: clear / ambiguous / unknown。
- `conclusion`: body / tip / contact / continuous-no-boundary / uncertain。
- 实际观察到的左/右/上/下侧，不默认复用模型侧分配。
- 同 pin / wrong pin / 假端点 / 重复 / 组件或标注不匹配 / 未确定。
- 可见 pin 编号、同 pin 的核验依据、是否在观察前揭示了参考。

因此“不知道”也是可保留的审核观察，不再被迫选一个点。浏览器实际验证了默认参考隐藏、连续无边界空点位、导航后保留草稿、空审核者禁止导出；测试草稿未导出或记为人工记录。

导出后运行 `scripts/prepare_pin_geometry_review_v2.py --reviews <第一位.json> <第二位.json>`。两份空白 JSON 模板也已生成，未签署、未完成。审核 JSON 和完整案例记录有 SHA 绑定；图片、裁图和 GT 源变更会失败。导入器不会修改 GT。

## 新指标的边界写清楚了

Semantic-Eligible P5 只在两位不同的实际审核者明确同意“同一根 pin”后计算，并要求：明确可见编号、所属器件及笔画依据；方向已独立确认；所选定义的点位在两遍间一致至 3px；先观察后揭示参考；精确编号能关联到可比较 GT。不能仅凭最近 GT 或 wire 投影选出“同 pin”。PCB 转换器自动编号不进入身份资格；显式原始编号仍标为弱真值，不能冒充独立 GT。

归一化法向误差只使用两遍实际观察到的 body→tip 长度，记录两位测量值、中位长度和对应误差区间。不用模型候选到自动 body 边的 gap 代替真实 pin 长度，也没有因为看到 −11 px 就加固定偏移。按域、实际审核方向与原有接触状态代理分别统计；接触状态代理仍不是独立确认的连接真值。

目前真实人工审核输入 **0**，因此 Semantic-Eligible P5、人工 tip/body/contact 距离、reviewer agreement 和归一化误差统计均为空/null。测试中存在合成审核 fixture，仅用于验证条件和篡改拒绝逻辑，不计入进展。

## 整页核对发现了一个实质 revision 风险

从 13 页对应 CAD 的全部 **976 个放置实例**提取 1303 条可见非零 pin 线段，以固定 144 dpi、0.25 px 容差与选中 PDF 的实际向量段核对，没有拟合 scale/offset。1259 条一致。这个数字只证明 pin 几何片段的对应，不证明全页文字、元件、net 或 revision 正确。

**BlueSMiRF-v2 只匹配 50/89 条**，虽然局部注册 IC 的显示身份此前一致。它应退出最先审核队列：局部成功不能覆盖整页版本差异。这也是本轮最重要的新证据。

按局部 IC 身份冲突、许可证据文件、multi-unit、DNP、全页 pin 几何兼容、原生 PDF 文字和实例规模排序，并合并 Directional/Navigation 近似设计族，得到以下优先队列：

| 来源 | PDF pin 段匹配 | CAD 实例 | 当前状态 |
|---|---:|---:|---|
| Qwiic Directional Pad | 93/93 | 66 | 优先人工核对；未 verified |
| GNSS Flex Breakout（小页） | 39/39 | 36 | 优先人工核对；未 verified |
| Qwiic 5V Boost AP3012K | 50/50 | 59 | 优先人工核对；未 verified |
| RFID USB-C Reader | 93/93 | 63 | 优先人工核对；未 verified |
| BMV080 Breakout | 54/55 | 47 | JP1.2 线段相差约 9.12px，需先核实 |

五个候选均在选中 CAD 页记录为无 DNP、无 unit>1 实例，但这些仍是设计元数据。前四个几何全部匹配，仍需整页显示修订、可见 key/number/name、许可范围、历史设计族排除和两遍独立人工审核。没有把四个来源自动宣布为四个独立设计。

## 两层证据与冻结顺序

13 页新增 `output/freeze-a1-two-layer/`：Level 1 是待人工填写的可见组件、编号/名称、pin stroke、接触；Level 2 是 CAD 的 reference、value、unit、DNP、hidden pin、零长度 pin 和坐标候选。Level 1 当前为空，CAD 属性不自动流入视觉 GT；未从几何片段制造 electrical net 真值。

首次推理门禁增加显式 `CoordinateConventionFrozen` 条件；读取 frozen manifest 时仍验证统一约定、设计数量和依赖 SHA。实际 `--freeze` 返回 NO_GO，没有生成 frozen manifest，没有首次模型评分。

目前里程碑：coordinate convention 尚未冻结；正式 verified designs **0**。本轮把下一步收敛为两位实际审核者完成这 25 个案例，以及优先核对前四个源设计。没有模拟人工签署，也没有降低设计族排除标准。

研究归档：`submission/semantic-endpoint-review-v2-assessment.zip`，只含代码与证据元数据；HTML 仍需本工作区的既有 crops。它不是比赛运行包。RC0/V1/R 的 53/89/215 个冻结文件保持不变。
