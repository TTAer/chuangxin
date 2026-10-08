# Freeze-A 原 CAD → 原厂 PDF → 像素标注（2026-10-05）

本轮已完成第一张真实厂商原理图像素页以及一个 IC 的原始标注样本，不再只有下载数量。该页仍是 **PARTIAL_PIXEL_SAMPLE_NOT_READY_FOR_BLIND_TEST**；未运行任何识别模型、未用于训练、未算作独立测试设计。RC0/V1 源码、配置与权重未改变。

## 实际成果与复现

来源为 [SparkFun Qwiic 5V Boost AP3012K](https://github.com/sparkfun/SparkFun_Qwiic_5V_Boost_AP3012K)。使用接收时已存档的 KiCad schematic 与原厂 schematic.pdf，不执行厂商代码。新增 PyMuPDF 1.28.2 安装在工作区虚拟环境；不依赖 KiCad CLI。

执行 `.venv/Scripts/python.exe scripts/render_freeze_a_source_sample.py`，得到 `data/processed/FreezeA-source-pixels/ap3012/page.png`（1584×1224，144 dpi）、`annotation.json`、未加标记审核裁图 `u5-review.png` 和审核覆盖图 `u5-overlay.png`。机器审计为 `data/splits/freeze-a-pixel-audit.json`。PDF、CAD、像素图与标注 SHA256 分别记录，没有覆盖旧冻结清单。

已标注 **1 个 IC U5/AP3012，5 个可见引脚，12 个可见字符串**：U5、AP3012、pin 1/SW、2/GND、3/FB、4/SHDN（带上划线）、5/IN。每个 token 记录可见文本框及 component/pin 归属；每个 pin 同时记录外端连接点、主体接触点、number/name、可见性。其余页内组件未标注，不能拿局部标注计算整页召回率。文本框经过本编码代理看图核对，尚未经过独立人工复核，明确不冒称已有人类审核。

## 坐标注册验证

该实例 unit=1、rotation=0、无 mirror。库坐标 y 向上，页面 y 向下；渲染尺度为 `2 × 72 / 25.4 pixel/mm`，页面偏移为零。主体矩形四条边与 PDF 原始矢量矩形吻合；五个 CAD pin 段的外端与内端分别匹配原 PDF 矢量线。主体边和 pin 段的最大误差均约 **0.0000245 px**，属于浮点输出误差。

这是一个明确限定范围的注册样本，不是已完成通用 KiCad 渲染器。脚本对位置、角度、镜像、页尺寸和误差做硬性断言；多单元只取当前 unit 与公共 unit，隐藏 pin 排除。本例没有被隐藏的 pin。后续旋转、镜像、零长度、隐藏名字、隐藏数字、多页实例，必须单独验证，不能套用此处变换。

## 发现的原 CAD 身份冲突

原 CAD 同一 symbol 的 `property Reference` 为 **U1**，而 `instances/project` 中还出现 **U5**；PDF 实际可见位号为 **U5**。标注因此采用经图像核实的 U5，同时保留 U1/U5 冲突和解析路径，未把最先找到的 CAD 字段当作像素真值。这证明原 CAD 的 placed records 数量与身份字段不能未经页面核对就转为训练标签。

## 来源许可与排除证据

本仓库 `LICENSE.md` 明确硬件为 CC BY-SA 4.0，软件为 MIT；此处采用硬件范围，保留 SparkFun Electronics、设计者 P. Lewis、来源和[许可链接](https://creativecommons.org/licenses/by-sa/4.0/)，注明 rasterization/标注/crop/overlay 修改。第三方图像及衍生物保留在本地来源池，不加入比赛提交 ZIP。具体附属库/图标范围的发布审查仍待完成。

前一轮本来源字节快照与历史文件检查无完全相同文件，**这只能排除字节重用**。PDF 标题块还写明来自 Indoor Air Quality Combo 与 SparkX Qwiic Boost；这些近亲设计必须与官方160/40、PCB55、10GT和历史训练图像按设计族核查。旧图像来源不完备，目前没有足以证明独立性的证据，审计保留 `verified_independent_designs=0`。

因此 Freeze-A 正式就绪页依然为 **0**。下一步应扩展通用实例变换并完成全页 visible component/pin/token 标注，同时取得设计排除及独立人工审核记录；达到20–25设计并冻结 image/annotation SHA 后才可首次推理。此页不用于 PinToken 参数选择。
