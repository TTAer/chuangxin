# 验证推进：固定来源取证

本轮继续暂停算法研发，没有训练、扩大数据集、调阈值、修改评测器或运行独立集推理。实际新增的是固定版本的来源证据和公开评分源码可用性核查。

## Directional Pad 的新增证据

以 `git ls-remote` 取得厂商 main 提交 `97943e7af32b431a7294c276c6518cfa115ab99d`，随后从该固定提交分别读取 CAD、PDF、LICENSE.md、README.md。本地四个文件与远端逐字节一致。又独立读取厂商文档站的 PDF，与本地 PDF 逐字节一致。

| 文件 | SHA256 |
| --- | --- |
| CAD | e395a48889462344fb2139e3eff35120dc138731c5b4c26afd85f8ea7e2d8c64 |
| PDF（仓库、官网、本地一致） | 35017b851811775f820faa48a9b625a8fa8bb3ecb9406399973a3b05c8f715b9 |
| LICENSE.md | 60350b68666a0d4903a7197114c723fc6d398d72218b785cc074013e39db374d |
| README.md | bfb7c604c33e3ba4c3cfb6312ae15b84b76e5c2c31fb71dafc58af922aab6e49 |

厂商资源页同时链接 PDF 与硬件仓库：[资源入口](https://docs.sparkfun.com/SparkFun_Qwiic_Directional_Pad/resources/)。固定来源：[仓库提交](https://github.com/sparkfun/SparkFun_Qwiic_Directional_Pad/tree/97943e7af32b431a7294c276c6518cfa115ab99d)。本轮未修改原始 source-record.json，新增取证作为旁附记录，避免覆盖历史 acquisition 信息。

**同一提交包含 CAD/PDF，不等于 CAD 与 PDF 全部内容对应。** 已有93/93几何段筛选仍不能替代整页文字、隐藏引脚和器件审核。许可文件一致也只证明来源字节一致，不能自动完成第三方元素/适用范围审核。因此 source_verified、page_content_verified、geometry_reviewed、family_exclusion_verified、coordinate_convention_ready、evaluation_ready 仍为 false。

历史训练/开发集来源记录不完整，Directional 与 Navigation 保守同族处理不变，不能借固定commit宣布 family exclusion 已完成。

## 官方评分证据尝试

GitHub API仍返回403限流；改用只读 shallow bare clone，实际取得 PCBnet commit `5c29eb16d7c62237797aae500534fbd677ab60cc`。该提交递归树只有 `README.md`，没有源码。结论限定为该提交，不推断私有资料、其他历史分支或未来发布。[作者公开仓库](https://github.com/ECNUACRush/PCBnet)。论文库本身也不必然是赛事官方 evaluator 来源。

重新读取[官方赛题六响应](https://cpipc.acge.org.cn/cw/getContestWebsiteQuestionById?questionId=2c9080179fb07f4f019fb8529dc21ff1)，SHA256仍为 `b50d677e385503b715ffb50d928a57615a14ecd240a08e15ce512ad355283b9c`，与前次一致。本轮未取得新的正式评分代码或官方逐项结果。官方等价性仍 unresolved，没有以本地 proxy 自测替代。

已整理一份未发送的[官方 evaluator 取证问题单](OFFICIAL_EVALUATOR_EVIDENCE_REQUEST.md)，可交给有赛事服务器/技术群渠道的团队成员。没有登录账号、加入群或对外发送信息。

## 人工审核输入检查

检查既有 `data/research`、`output/review-handoff` 与当前用户 Downloads 下相关名称的 JSON。没有发现新增人工审核导出；已有 agent-pixel-review 和 family-review 不算真实 A/B 结果。这是指定目录的输入检查，不能推断私人记录或所有其他目录。

实际审核仍需要分别使用既有入口：[A](http://127.0.0.1:8767/pin-geometry-review-v2/index.html?pass=A)、[B](http://127.0.0.1:8767/pin-geometry-review-v2/index.html?pass=B)。本轮未读取浏览器草稿、替填记录或修改协议，也未重复审核页面批验。

Reviewer A/B仍各0/25，verified designs=0。首次独立推理、review hash及评分结果均未产生。RC0冻结与NetSetQuery条件停止继续有效。

## 保存的证据

`output/validation-evidence-20261006/source-intake-evidence.json` 保存每次请求URL、UTC取证时间、SHA、HTTP状态和本地比较结果；同目录保留公开原始响应。`pcbnet-git-tree.json` 记录实际commit及递归文件列表，bare clone用于追溯此次树检查。`manifest.json` 冻结本轮证据及记录 SHA，不把网络取证时间当作首次独立推理时间。

本轮推进的是来源可追溯性，不是准确率或人工审核完成数。下一步能改变核心状态的仍是两份真实审核结果、整页实际审核证据或正式 evaluator。
