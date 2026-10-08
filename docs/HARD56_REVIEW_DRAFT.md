# Hard56 代理图像审查草稿

本轮固定审查全部 56 个开发验证 hard case，未根据纠正结果删样本。已逐一通过四张联系表核对实际像素。审查是 AI 代理草稿，不是人工审核或独立真值。

分类统计：{"geometry-fixable": 4, "unknown": 2, "label-ambiguous": 43, "ocr-fixable": 7}。geometry-fixable 与 ocr-fixable 是待验证的可纠正假设；并非已确认可恢复样本。其余 label-ambiguous 包括弱 pinNumber 字段放入器件型号、位号、容值，以及相邻行引用偏移或近重合端点。

原始 HRR 保持 **9/56 = 16.07%**，原测试门槛不变。reviewed-correctable HRR 为 **null**，因为尚无独立人工修正真值。没有训练、运行时改动或 Freeze-A/B 数据使用。

## 关键观测

0016 的数字 12/13/17/10 多次弱关联到下一行；0043 将 STM32F103C8T6、0.1uF、U1、RT9193-33、USB 标成 pin_number；0064 的 VBUS 弱关联到安装脚 7，而印刷文字实际在电气脚 1 行。不能让关联模型追随这些未审核的目标。

潜在几何难例包括 0005 GND、0108 SENSE、0128 CV 的跨边最近端点，以及 0108 数字 20 的行间字形位置。潜在 OCR 工作包括连接器合并 token 234、4 DO 和 0150 的 IO5/IO0 被转录为 105/100。当前输入 token 已与弱源原文精确匹配；此处没有把最近端点错误直接称作 OCR 错。

原缓存要求弱目标位于 top8，故 candidate-missing 计数 0 是筛选条件，不能据此断言真实候选召回完整。

## 审核交付

- 明细：`output/ablation/hard56-review-draft.json`。
- 联系表：`output/ablation/hard56-review/contact-1.png` 至 `contact-4.png`。
- 交互入口：`output/ablation/hard56-review.html`。支持保存浏览器本地审核备注、导出单独 JSON；不修改草稿、训练标签或评分。人工审核时需确认文字转录、文字角色、实际所属元件、端点身份，再建立独立修正标签。

父实验依然以 raw56 为原分母。代理草稿不能用于宣称新的 HRR 或竞争分数。
