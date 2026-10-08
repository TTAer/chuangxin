# Freeze-A1-Identity 像素身份核查（2026-10-05）

本轮实际逐裁图比较 13 个去重原厂页面的 27 个 IC/262 条已注册引脚，新增原生 PDF token 关联、逐实例身份记录和首次推理门禁。没有运行 RC0，没有训练，没有拟合或优化注册坐标。**正式可评估的独立设计仍为 0，未建立 frozen 盲测清单。**

| 项目 | 实际数量 |
|---|---:|
| 已查看原始像素 IC 裁图 | 27 |
| 代理局部比较一致的 IC / 页面 | 21 / 12 |
| 这些 IC 涉及的仓库来源 | 10（不是 10 个独立设计） |
| 代理核对可见 pin 编号 / 名称 | 238 / 240 |
| 原生 PDF 文字关联：编号 / 名称 / key+model | 50 / 50 / 6 |
| 原生 PDF 文字关联涉及页面 / IC | 2 / 3 |
| 正式独立设计 / 独立人工审核 / 首次模型推理 | 0 / 0 / 0 |

代理核查不是独立人工审核。238 个编号与名称对仅是 **IC 局部身份候选**，并没有全页 component GT、Net GT，也没有把 13 页或 10 个仓库算作独立设计。Directional Pad 与 Navigation Switch 共用近似的 PCA9554 核心布局，来源审查清单保守归入同一个候选设计族。

## 实际发现并处理的标签问题

- Soft Power Switch Mk2 的 3 个注册实例，其 PDF 显示模型分别为 `74LVC1G175/14/17`，当前 CAD 字符串却为 `SN74LVC1G175/14/17DCK`。原固定几何注册无法证明修订相同。本轮对这页停止批准身份候选；保留显示转录和差异。
- Soft Power Switch 的供电脚单元显示 `U2B/U3B`，CAD 的 property 是 package reference `U2/U3`。不能把 unit B 的矩形供电主体当作完整器件，更不能凭相同 reference 合并或宣布漏检。USB-C 版两个 unit B 的模型文字超出原裁图，保持未批准。
- MCP73831 的 mirror-y 实例保持已有的 5 条可见引脚身份核对；并未把这个局部镜像成功泛化到其他实例。LM66200 的 `GND/VOUT` 名称可见，数字编号未显示，两个编号从像素身份 GT 候选中排除。
- GNSS Flex pHAT 的 CAT24C512 EEPROM 是灰色主体加红叉，CAD 也记录 `dnp=yes`。作为 DNP 证据保留，从 populated-component 身份候选排除。
- AP3012 原候选 reference 为 CAD property/project `U1/U5` 冲突；原厂像素实际显示 `U5`。仅解决这一局部显示，不自动批准整页修订。

原生 PDF text layer 在两页存在，其他页面文字为矢量笔画。已先按文字与坐标去除重复 PDF 文字，再采用固定邻域、确切语义文字和一对一占用关联到 IC/pin，避免同一 token 被多个 pin 复用。原始文字、坐标、关联距离都保留；overbar/subscript 的 CAD 表示单独保留，不冒充字面 OCR 字符串。无 PDF text layer 的裁图核查不制造文字 bbox。

## 固定评估和首次推理门禁

新增 `scripts/freeze_a1_identity_evaluator.py`，只定义局部 IC body IoU≥0.5、可见引脚距离≤3 px、严格显示 key/model 及 number/name 配对；输出明确不含全页召回和 Net 分数。匹配由几何阈值内的一对一分配决定，未按身份正确性挑配对。此评估器尚未用于任何模型分数。

`scripts/freeze_a1_identity_gate.py --freeze` 至少要求 10 个已验证 design ID，并同时要求来源许可范围、整页 CAD/PDF 修订、历史设计族排除证据以及独立标注审核。仓库数量不能替代 design ID；宣称历史排除而没有 corpus manifest 也会失败。冻结时绑定 source/image/annotation/agent-review/family-review/evaluator/匹配依赖及许可文件 SHA；已存在的 frozen 清单遇到任何变更只能报错，不能覆盖。

本轮实际执行 `--freeze` **返回退出码 1，NO_GO**，未创建 `data/splits/freeze-a1-identity-frozen.json`。这个失败是预期门禁行为；未越过门禁运行 RC0。新增 6 项测试通过，包括仓库重复计数、排除证据缺失、unit reference 严格评分和 token 复用。

## 交付位置与剩余工作

- `data/research/freeze-a1-agent-pixel-review.json`：27 个逐裁图显示核查记录、原图/裁图 SHA、编号隐藏及标签差异。
- `data/research/freeze-a1-family-review.json`：11 个仓库来源的许可文件证据及未完成的设计族/修订审核；没有假签署。
- `output/freeze-a1-identity/identity-audit.json`：去重逐 IC、source metadata、可见 pin 候选和 106 个原生 PDF token 关联。
- `output/freeze-a1-identity/readiness.json`：逐 IC 阻塞原因以及全部冻结所需 SHA。

目前缺口是旧训练/开发数据设计来源、整页 PDF 修订对应及独立标注审核；不是像素注册的小数误差。历史数据设计来源不完整，字节 SHA 无重用不能证明设计独立。继续补齐来源审查后，先用真正满足条件的局部 IC 身份子集冻结并作一次性评估，再扩大到全页；禁止把候选页用于训练/调参后称为盲测。
