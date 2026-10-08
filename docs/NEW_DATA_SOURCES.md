# 新增数据来源与实际使用范围

本地数量以 `output/new-dataset-research/local-audit.json` 和下载清单为准，不把作者公布的全量规模计入已获取数据。

| 来源 | 已获取 | 本轮用途 |
| --- | --- | --- |
| [CGHD v16，Johannes Bayer](https://zenodo.org/records/17469897) | 160 组原图、二值分割、XML 和实例标注，33 个 drafter | 640 个裁剪，只监督笔画分支；未下载整个 4.9 GB 归档 |
| [AMSNet 1.0 作者仓库](https://github.com/AMS-Net/ams-net.github.io) | 734 个原始案例，各含原图、元件标注图、网络标注图、bbox JSON、SPICE | 结构与来源核查；尚未接入训练 |
| [Netlistify 作者仓库](https://github.com/NYCU-AI-EDA/Netlistify) | 未获取 | 作者的 Hugging Face 数据需要登录并接受联系方式分享条件，未绕过访问条件 |
| CircuitPile-9k | 未获取 | 检索到旧作者说明，仓库访问返回 404；不计入可用数据 |
| [CircuitVQA 原论文](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2024/lncs14941/lncs14941437.pdf) | 未获取图像 | 问答条数不能当作独立图像数，且与 CGHD 存在来源关联 |

AMSNet 有 2202 个 JPG 文件，但其中 1468 个是同一原图的标注视图。734 个原图 SHA256 均不同，734 个案例均有 bbox JSON 和 CIR。CIR 和框标注尚不足以直接建立精确像素引脚编号对应关系；标注网络图也不是可直接用作训练的导线分割掩膜。作者仓库包含 GPL-3.0，书籍电路插图的独立权利未厘清，因此本轮用于检查，不分发原图、不用于训练。AMSNet 2.0 未下载。

CGHD 160 个图像文件 SHA256 均不同，与现有 PCBnet/AITEE 文件哈希未重合；这是字节级核查，不能证明设计独立。按 drafter 留出 129 个训练样本、31 个验证样本，drafter 不交叉，但设计前缀 C1、C2 同时出现在两侧。因此该验证是书写风格泛化检查，不能宣称独立电路设计泛化。CGHD 标注的是整张图的笔画，包含文字、符号和导线，不能把它直接当作导线或电气连接真值。

## CGHD 授权与归属记录

作者：Johannes Bayer；版本与原始页面：[Zenodo 17469897](https://zenodo.org/records/17469897)。图像经缩放、裁剪用于 DirectionalFieldNet 的笔画辅助监督；本项目未重新发布原始图像或标注。每个提取成员核对 ZIP CRC，下载清单保存 SHA256；未下载全量归档，故未声称验证全量 MD5。

Zenodo API 标为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，归档 README 标为 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)，二者存在差异，保留两项声明。随开发包发布的 `directional-fields` 和 `directional-fields-local` 模型权重采用 CC BY-SA 3.0，并保留作者、来源和训练修改说明；不把元数据差异当作无条件授权结论。这个权重声明不改变其他已有代码、数据和模型各自的授权。


## 2026-10-04 四边引脚数据补充

自建 SidePinBench 240 个布局族，按族划分 192/48，共 1,920 条带；与既有官方数据的 4,292 条带合计 6,212。仅有几何端点监督，不是独立真实电路设计。完成三组训练，新增合成数据尚未在两个开发域同时提高整图指标。

实际下载 [CircuitOCR](https://github.com/ZhangJ83/circuit_ocr_dataset_final) 64 个训练、16 个验证图文样本；80 个解码像素哈希与官方库存无完全重合，设计独立性仍未知。数据没有引脚坐标/网络真值；README 声称 MIT，但 LICENSE 路径返回 404，图片权利未核查，故仅审阅、未训练，不打包原图或转录。SchGen、OpenSchematics 仅核查数据卡与目录，未下载完整数据。详见 [新架构执行记录](ARCHITECTURE_SIDE_STRIP.md) 与 `output/side-source-audit.json`。
