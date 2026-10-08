# 局部位号锁定与电气权限保护（2026-10-05）

本轮处理评估提出的三项改进：Local-first / No-steal、文字角色证据，以及新增身份的电气提交权限。视觉权重、训练数据和导线感知算法没有更新，仍未使用预留独立集。技术报告继续暂缓。

## 本轮结论

| 候选 | PCB55 代理分 | 相对旧组合改善 / 退步 | 10GT 代理分 |
|---|---:|---:|---:|
| 旧 identity + SideStrip 组合 | 16.807477 | — | 28.815553 |
| 上轮 guarded | 16.972601 | 9 / 4 | 28.815553 |
| stable 局部锁定，无电气保护 | 16.936125 | 8 / 2 | 未单独重跑；组合对照见下 |
| stable + 全部新增元件保护 evidence | 16.887482 | 9 / 1 | 28.815553 |
| **stable + 已确认型号保护 catalog** | **16.939933** | **9 / 1** | **28.815553** |

catalog 候选相对旧组合 +0.132456，相对现有正式默认 16.703718 +0.236215。其均分比上轮 guarded 低 0.032668，但整页退步从 4 页减少到 1 页，选择依据是风险收敛，而非最高开发分。

唯一整页代理退步是 0064（-1.147882）：图中 OCR 有 USB2，转换 GT 使用 C_2。该事实记录在 `output/ablation/proxy-conflict-cases.json`，仅用于后验评估；运行时不读取此文件、不按页号回滚、不猜测不可观测 C_ 编号。0007、0040 与旧组合完全相同；0197 综合分从 32.248826 提高到 32.419618。

**NetLineF1 仍未全部恢复。** catalog 候选均值 0.073278，旧组合 0.073922。0197 的 PinF1 0.260870→0.258228，NetLineF1 0.172471→0.155629，虽然该页综合分上升，也不能说每个分项均获益。开发集的 9 / 1 是综合代理分统计，不是逐项或官方比赛无退步的保证。

## 先修正上轮诊断

四个退步页的原始 wires、junctions 和共同元件 pin 坐标没有变化；输出边只增加了 1 / 2 / 2 / 2 条，没有删除已有几何边。0007 也是新增 Other/P12，而不是改写旧键。因此，不能把 NetLineF1 下跌直接归因于物理导线被修改。新增身份会让原本无引脚关联的导线进入网络，并改变 net 配对与评分分母。

本轮又核验全部 65 页的新旧原始 wire / junction / contact-decision SHA 完全相同，见 `output/ablation/stable-physical-audit.json`。原始几何证据不变与导出电气超图相同是两项独立性质，不能相互替代。

## 文字归属与增量恢复

`pcbparse/text_roles.py` 输出可解释的 designator_score、OCR 置信度、相对间距、文字高度和拒绝理由。这是规则证据分，**不是经训练校准的概率**。对重复 Pxx 列、明显属于更近 IC 的文字及远离本体的新增候选做保守处理；不使用颜色作为连接判据。

`--designator-mode stable` 先保留已有可输出绑定。距离本身不能删除旧身份；只有明确 GPIO 列 / 竞争本体的文字角色矛盾才解除旧绑定。随后锁定有双向优势的局部绑定，剩余少量候选进入匈牙利匹配。新增未知类别保持弃权，已有未知类别仍可保留，避免新增 Other/P12 与删掉 Motor/M 等两种风险。

`output/ablation/stable-visible-unassigned.json` 对那 7 个局部可见但原先未分配的对象逐项记录 bbox、OCR token 置信度、局部文字与角色证据，最终 **7 个键全部输出**。这只是开发诊断中的 key 恢复，不代表 Name、pin 和 net 全部正确；OCR 引擎未提供逐字符置信度，该字段明确记为不可用。

全体 GT Box：输出几何匹配 95 个（上轮 98），正确 key **仍为 71 个**（旧组合 63）；去掉了部分没有正确键收益的候选。库内 51 个 GT 的漏斗为：30 个输出、25 个正确键、19 个正确键且确认型号与 GT 一致。10GT 的对应库内漏斗仍为 15→10→10→10。

## Geometry-to-Electrical 提交权限

`pcbparse/identity_transaction.py` 为新增元件检查型号库编号/名称、真实端点笔画、已有 wire 接触，以及是否会桥接不同 wire group。已存在元件的 pin 坐标不被此保护模块修改；wire / junction / contact 决策有不可变签名。

新增元件有 identity_confirmed / pin_grounded / electrical_confirmed 分级。有效但没有足够拓扑证据的 pin 保留在几何图和 pin 输出中，`electrical_pin_permissions` 阻止其参与新的电气连接。库里不存在的编号如 pin_200、pin_I0 不作为已确认引脚输出，保留为审阅假设。原有同名飞线合并继续有效。

两种范围均保留作为消融：

- `--identity-transaction-mode evidence`：检查全部新增元件，使用固定无源器件编号结构或局部 OCR 编号；仍会损失部分正常召回，因此没有选为候选。
- `--identity-transaction-mode catalog`：只对已确认型号的新增元件做保护；其他元件保留已有感知链，报告状态为 existing_perception，**不声称这些未知元件已通过新增保护**。这是本轮候选。

典型效果是 0197：确认型号的 LED 原先带有 OCR 错配 pin_200 / pin_I0；检查库内编号后，这些不再直接进入网络。型号库只确认编号与名称，从不提供图中上下左右顺序。

候选运行参数：

```text
--designator-mode stable --identity-transaction-mode catalog
--ic-identity-mode retrieval
--side-pin-model models/candidates/side-strip-local/side-strip.onnx
--side-name-anchors
```

## 验证与交付范围

99 项测试通过。两域 65 页候选完成检测、引脚、拓扑和完整图重新推理，使用此前 OCR 缓存；全部变化页加固定对照再次以新鲜 OCR 重放，证据 `output/ablation/stable-live.json`。65 页通过比赛出口类型和引用验证。

新镜像 `graphschematic-stable:20261005` 在断网、2 CPU / 2 GiB、只读输入下验证默认指南图及 0067、0101、0197 三个候选案例，身份/名称/网络严格一致，仅浮点坐标允许 <0.001 像素差异。53 个运行源码和选定模型 SHA 与镜像逐项一致。

新包 `submission/graphschematic-stable-development.zip` 使用 `scripts/package_stable_development.py` 构建；旧 guarded 开发包冻结为 `submission/graphschematic-association-20261005-frozen.zip`，此前 identity 包及其 50 案例记录也继续保留。新包解压后的比赛入口用默认指南图和 0197 候选进行实际验证。旧构建脚本属于对应历史阶段，新包的 manifest 不借用旧镜像的批验状态。

**当前默认仍是 legacy / transaction none。** 本轮候选没有完成新镜像的 50-case 压力批验，也没有独立 PCB 盲测或正式 evaluator 核验，因此没有直接升级比赛默认。现有 20–30 页盲测缺口仍需独立数据与标注解决；不能把本轮反复使用的 65 页或重复运行当作新的独立集。

下一步冻结这三项逻辑，用独立集检验文字归属、未知型号分支的电气风险，以及 NetLine 分项，再决定是否执行 RC 的默认 / 候选 50-case 双批验和切换。
