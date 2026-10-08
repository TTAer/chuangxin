# 局部高分辨率 pin OCR 开发诊断（2026-10-05）

本实验仅诊断现有 RapidOCR 在小引脚文字上的表现，不训练 OCR，也不接入 RC0。数字与名称来自 CVJson 的显式 `pinNumberByOCR` / `pinNameByOCR` 文本引用；这些是弱引用，不是经人工审核的 pin identity 真值。

## 固定协议

沿用 PinTokenNet V1 的 48 个训练页、16 个开发验证页以及原有全图 OCR 缓存。仅挑选 Box 的唯一角色、唯一引脚引用，并要求全图 OCR 没有同时满足字面相等与原文本框 IoU≥0.25 的结果。训练/验证 × number/name 四层各取12条，按页和文本 ID 排序，逐页轮取；没有使用 Freeze-A、Freeze-B、10GT，也没有按实验结果更换样本。

从原图按 GT 文本框外扩8像素裁剪，固定运行原尺寸、2×、3×、CLAHE+2×、自适应二值化+2×、顺时针90°+2×、逆时针90°+2×七组对照。放大采用 cubic；CLAHE clip=2、4×4 tiles；二值化 block=11、C=2。不猜字符、不自动补号，不虚构 top-k 或字符置信度，不使用 GT 建 whitelist。

协议在 OCR 开始前保存为 `output/ablation/local-pin-ocr-protocol.json`。像素文件哈希逐样本核对，输出保留各变换的真实识别串、置信度、还原到原图的框和耗时。

## 指标含义

严格恢复要求完整字符串相等且对应框 IoU≥0.25。仅在 crop 内某处出现同样的数字不能算恢复，因为相邻 pin 常有相同或相近数字。另列字符串出现但定位失败、空间最近候选错误串，帮助区分可疑定位与识别风险。

七变换并集为借助 GT 判断正确答案的 oracle 上界，不能当自动选择策略。GT 裁剪本身也是 oracle ROI，因此所有结果属于条件局部诊断，不是整图 PinF1、真实未解决 pin 的自动恢复率或绑定准确率。

## 标签风险

固定样本中包含明确引用但内容为 `R10 1k` 的“pin number”，以及 `C4` 等可能是 ball number 也可能是元件位号的字符串。这提示引用角色存在噪声。保留原样，逐条标记；不事后改 GT 或删失败样本。附加 `[A-Z]?[0-9]+` 语法有效子集只用于诊断，语法匹配不能证明数字确属该 pin。

## 复核

- 脚本：`scripts/audit_local_pin_ocr.py`
- 固定协议：`output/ablation/local-pin-ocr-protocol.json`
- 实际结果：`output/ablation/local-pin-ocr.json`
- 两项坐标/评分测试：`tests/test_local_pin_ocr.py`

## 实际结果与预处理问题

旧协议完成20/48条（12个 number、8个 name，140次完整OCR）后停止，保留所有已完成结果，状态为 `partial_stopped_after_preprocessing_confound_identified`。其七变换并集严格恢复 number 0/12、name 1/8；单独2×均为0。没有把未执行的28条计成失败或完成。

停止原因是源码及实际 detector tensor 审计：`ImageParser` 配置 side=1600，但 RapidOCR 的 resize 类型为 **min**。首样本 crop 31×61、放大后62×122、93×183，三者实际输入都变成 **1600×3136**。所以原尺寸、2×、3×比较存在严重归一化混杂，而额外重复推理代价很高。记录见 `output/ablation/local-pin-ocr-preprocess.json`。

随后保存独立协议，保持同48条、同8像素padding与严格评分，另用 `RapidOCR(det_limit_type='max', det_limit_side_len=736, threads=2)` 只跑固定2×。这是根据已定位的预处理问题追加的开发诊断，不是独立测试或超参数盲测。旧记录未作废，未挑换样本。该版本 `get_preprocess` 会在小输入时使用动态threshold960，配置736并非固定模型输入尺寸；实测首样本2×输入为64×128。

| bounded-local 2× | 样本 | 严格恢复 | 有空间候选但选中错误串 |
|---|---:|---:|---:|
| pin number弱引用 | 24 | 9（37.5%） | 4 |
| pin name弱引用 | 24 | 11（45.8%） | 10 |
| number语法有效子集 | 21 | 9（42.9%） | 见逐条记录 |

全部48次bounded-local OCR已实际执行，详细记录为 `output/ablation/local-pin-ocr-bounded.json`；它的预声明协议为 `output/ablation/local-pin-ocr-bounded-protocol.json`。同前20条做配对比较，旧2×恢复0条，bounded2×恢复8条。其余28条没有旧模式结果，因此不做完整48条的成对收益声明。

固定number样本中3条不符合简短编号语法：保留为弱标签风险；另有1条ball编号/元件位号歧义。语法有效子集的9/21也不是已审核pin number准确率。name错误候选10条说明局部OCR依然可能引入不正确的身份证据。上述恢复都未经过预测端点的绑定审核，不构成真实PinF1提升。

## 决定

支持继续开发**限定尺度的局部文字证据**，但不把旧全图OCR参数直接复用于紧裁图，也不把变换并集或GT裁剪结果当自动策略。下一步应由预测的 unresolved pin 区域触发，再用最近锚点保护、空间唯一性、真实候选库和电气权限做保守审查；先证明错误串不破坏易例，再做整图Pin/Net验证。

当前 RC0、V1 权重、训练记录和 split 均保持不变。两项评分/旋转坐标测试通过。
