# Freeze-A 实际采集与缺项（2026-10-05）

本轮已经下载真实制造商 PCB 原理图资料，并抽取原始 CAD 身份字段；Freeze-A **仍为 NOT_READY_SOURCE_INTAKE**。没有运行模型、没有伪标签、没有把下载仓库数当作独立测试页数。RC0 代码、模型、镜像和既有评分协议未修改。

## 实际落盘结果

| 项目 | 数量 | 含义 |
|---|---:|---|
| 尝试来源仓库 | 25 | SparkFun、OLIMEX、Hat Labs 原厂/作者仓库 |
| 已取得 CAD 的来源仓库 | 24 | 候选设计族，尚未完成设计级排除 |
| 清单内原始文件 | 272 | CAD、厂商 PDF、README、许可证；逐文件 SHA256 |
| 厂商 PDF 文件 | 45 | 可能包含相同版本副本、多页或板级说明；不是45独立测试页 |
| 去除同仓库字节相同副本的现代 KiCad sheets | 39 | 含子页、版本，不是39独立设计 |
| CAD placed component records | 2332 | 含电源符号、旧版本；未按比赛类别和可见性审核 |
| CAD 库引脚记录 | 4998 | 包含多单元/隐藏引脚定义，不能直接充当图像 GT |
| 既有文件字节排除检查 | 7682 | 既有 raw/processed/examples 的图像、PDF、CAD |
| 字节完全相同的新旧来源文件 | 0 | 仅排除字节复用，不证明设计不重叠 |
| 已审核独立设计 / 已审核像素 GT / 推理次数 | 0 / 0 / 0 | 目前不满足冻结门槛 |

机器清单：`data/splits/freeze-a-source-intake.json`。原始资料：`data/raw/FreezeA-manufacturer/`。每个成功来源有 `source-record.json`，记录来源 URL、取得的 branch、文件 SHA256；整库取得时还记录 archive SHA256。branch 可以移动，因此固定的是接收的字节快照，不假称固定了 Git commit。

原始 CAD 证据：`data/splits/freeze-a-cad-evidence.json`。它记录 placed component 的 Reference、Value、library ID、实例位置/旋转/镜像、库中的 number/name 和 symbol-local mm 坐标。解析39个现代 KiCad sheets 无报错，但未完成单元可见性筛选、坐标变换和渲染注册，因此显式标为 `not_ground_truth_for_pixel_scoring=true`。

完整准备状态：`data/splits/freeze-a-readiness-audit.json`。

## 来源与许可

原厂来源示例：[SparkFun Directional Pad](https://github.com/sparkfun/SparkFun_Qwiic_Directional_Pad)、[OLIMEX ESP32-DevKit-LiPo](https://github.com/OLIMEX/ESP32-DevKit-LiPo)、[Hat Labs SH-ESP32](https://github.com/hatlabs/SH-ESP32-hardware)。来源和本地许可证副本一起保留；本轮仅提取数据/文档，不执行下载代码。

已检查的 SparkFun `LICENSE.md` 区分硬件 CC BY-SA 4.0 与代码 MIT；不能把硬件资料作为 MIT 数据发布。OLIMEX ESP32-DevKit-LiPo 的 README 区分硬件 Apache 2.0、软件 GPL3、文档 CC BY-SA 3.0。Hat Labs 根 LICENSE 为 CC BY 4.0，但 `ESP32.kicad_sch` 标题块声明 CC BY-SA 4.0，存在文件层级差异，复用前需按具体文件确认，不自动继承根许可证。

Portability Shield、WAV Trigger Pro、Fingerprint FPC2534、OLED 1.5in 四个已采集仓库在当前提取清单中没有独立 LICENSE 文件。它们保留为待审核来源；不因同属 SparkFun 就自动判定许可完成。每个来源都仍需检查附属 symbol/library 文件和 PDF 的许可范围，当前不将这批第三方原图加入比赛发布 ZIP。

## 失败源与恢复

初始20个来源成功。BMV080 首次因 Windows 长路径失败，排除 3D 模型附带文件并检查路径长度后恢复。RTK Postcard、BlueSMiRF v2、Pico Flipper 整仓库超过70 MB下载上限，转为读取原厂 resources 文档并下载30 MB以内的原始 CAD/PDF 小附件，三者均恢复。清单保留取得方式和实际文件 URL。

`sparkfun/SparkFun_GNSS_LG290P` 的 main/master 均 HTTP 404，本轮未取得该源，明确计为失败；不能因为搜索提到 LG290P 产品就算 CAD 已取得。GitHub API 首次403限流后改为厂商仓库 codeload/resource assets；不绕过账号授权或访问控制。

## 冻结前的实际待办

1. 按厂商项目/PCB功能/修订关系合并设计族：同项目修订、子页、相同电路的转存副本只能算一个设计族；Soft Power Switch 等近亲项目尤其需要复核。
2. 与官方160训练/40验证、PCB55、10GT及其他历史训练/开发来源做设计级比对。旧图像来源存在未知元数据，必须人工核查图像与电路结构，字节不相同不足以放行。
3. 每族确定一个有 IC 与可见引脚身份的真实 schematic page，取得一致的 PNG 渲染和 page/mm→pixel 注册。当前没有可调用的 KiCad CLI，不能假称39个 CAD 页已经渲染。
4. 用原始 CAD Reference/Value/pin 定义辅助审核，逐页复核 component 主体 bbox、可见 key/name、IC pin point、可见 number/name；隐藏引脚和未知字段保留 unknown。PDF/CAD 的可见性和像素坐标未审核前不进入 evaluator。
5. 审核至少20–25个独立设计、完成具体许可/署名/标注审核记录，生成固定 image/annotation SHA256；在首次推理前冻结。此后不用于 PinToken 参数选择。

复现命令：`python scripts/acquire_freeze_a_sources.py`（已存在来源字节改变时拒绝覆盖）；失败回退 `python scripts/acquire_freeze_a_small_assets.py`；离线审计 `python scripts/audit_freeze_a_sources.py`。下载上限、ZIP CRC、路径检查与逐文件哈希约束均在脚本中。本轮下载进程已全部结束。

这批资料已经把 Freeze-A 从空模板推进到真实 CAD/PDF 采集池，但独立性与像素 GT 仍未满足，不能支持 RC0 切默认，也不能用于本轮 PinToken 的训练或调参。
