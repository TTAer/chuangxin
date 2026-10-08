# 2026 创芯大赛 EDA 赛题六公开信息核对

核对日期：2026-10-03，北京时间。

## 比赛和时间

本项目参加的是“长沙银行杯”2026 中国研究生创“芯”大赛 EDA 精英挑战赛，赛题六“AI驱动的PCB原理图自动解析”，命题单位为芯粒 CAD 和制造浙江省工程研究中心、比昂芯。不要混用创芯大赛主赛道的提交格式或2025年的赛题六。

[官方首页](https://cpipc.acge.org.cn/cw/hp/2c9080158ee9c272018f229208b610a6) 当前公布：报名 2026-08-04 至 2026-10-16；资格审查 2026-08-10 至 2026-10-18；作品提交 2026-08-10 至 2026-11-07。每队1-3位学生、1-2位指导老师，每位学生参加一个队伍和一道题，可跨校组队。若有本科生队员，队长须为研究生。

## 赛题六提交入口

[官网赛题六公开 JSON](https://cpipc.acge.org.cn/cw/getContestWebsiteQuestionById?questionId=2c9080179fb07f4f019fb8529dc21ff1) 的 `workNocheckDesc` 明确写明：作品提交至“赛题服务器指定位置”；赛题技术 QQ 群 **690857791**。该公开响应未给服务器 URL、API 密钥、环境具体版本或完整 evaluator。项目未登录报名账号、未加入群、未发送消息或替用户提交。

[官方指南下载](https://cpipc.acge.org.cn/sysFile/downFile.do?fileId=64bf6bd960b142678e4a463ecbb7a175) 文件大小 2527816 字节，SHA256 为 `45eedbc195ad71bbb03044c4606ea85ccf3eccebbb3d9b3304d6e325e1acc5e8`，与用户目录中的指南完全一致。指南要求源码或可执行文件及 Word 技术报告；预测每例的 components、pins、nets。全部正式测试例运行时间须小于2小时，默认 bbox 阈值20像素，pin/line阈值5像素。评分 Component/Pin/Net/Line 权重为30%/25%/35%/10%，还包含 token 排名和数据贡献部分。官方环境提及 qwen3.7-plus；真实可调用模型名和配置须以参赛服务器为准。

## 数据已获取

[PCBnet 作者仓库](https://github.com/ECNUACRush/PCBnet/) 公开提供 [Google Drive Dataset300](https://drive.google.com/file/d/1q_rwzchFRyd0hJ3nsOWFcPZn-zny_ccW/view) 和阿里云盘 `https://www.alipan.com/s/HYuvtbNZf3q`，提取码89di。

本次已成功获取 Google Drive 压缩包，61466370字节，SHA256 `0d4bca064d765272b588a0ab1886b03a8d5b516f8e991500efc11b8a5246c929`。实际含300例：216张PNG、84张JPG，每例原始 CVJsonStd JSON、EDS、SPICE net。全300例已通过开发格式适配校验。数据的原始标注用左上角图像坐标，转换输出时采用左下角坐标。

采用文件内容SHA256分组：244例训练、56例验证，精确重复文件不会跨两侧。仍需对近似重复/同源设计进行分组审计，当前不声称对未知设计的泛化能力。原始标注的文字字段、位号缺失和其他不完整项逐例记录，因此转换后数据是开发参考，不能称为赛事最终 golden。

作者 README 的许可栏说明将于资源发布时公布许可。当前下载内容只用于本地研发，提交包不包含原始数据。不得把训练集指标充作黑盒成绩。

## 采用的技术来源

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) 与 [训练接口](https://docs.ultralytics.com/modes/train/)：以官方 YOLO11n COCO 权重为初始化，在PCB训练样例上微调元件检测。使用已安装的 ultralytics 8.3.161，训练脚本与参数保存在项目中。Ultralytics 代码/模型使用 AGPL-3.0 条款，候选包保留说明并提供源码。
- [RapidOCR](https://github.com/RapidAI/RapidOCR)：使用 rapidocr_onnxruntime 1.4.4 和随包 PP-OCRv4 ONNX 模型实现本地OCR。项目代码和第三方权重分别记录出处，不宣称 OCR 权重由本项目训练。
- OpenCV 几何处理：颜色分离导线与符号、直线合并与交点证据、模板辅助 pin 位置估计；网络重建和开发代理评测是本地实现。

## 还需官方确认

赛题服务器地址、实际提交命令/目录、指定Python/PyTorch版本、API配置与配额、官方评测代码、最新Q&A、元件库、数据提交说明和报告模板。本阶段将这些作为明确的交付核对项，不编造已通过官方平台的结论。
