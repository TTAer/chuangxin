# Circuit OCR Dataset Final

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PaddleOCR-VL](https://img.shields.io/badge/Model-PaddleOCR--VL--0.9B-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/yingchu83/CircuitOCR)

面向电路原理图 OCR 的高质量标注数据集，用于视觉语言模型（VLM）微调与评估。配套训练代码、评测脚本、技术报告见 [主项目仓库 (circuit-ocr-paddle)](https://github.com/ZhangJ83/circuit-ocr-paddle)。

---

## 最新评测基准 (Phase 1 / V10-Fixed)

> 评测环境：使用 `eval_benchmark_v3.py` 在 easy50-pure 测试集（44 样本）上运行，采用 `LoRAModel` 封装器逐参数注入。

| 模型 / 检查点 | ExactMatch | CompF1 (标号F1) ↑ | CompPrec ↑ | CompRec ↑ | TokenRec ↑ | NED ↓ | RepRate ↓ | Diversity ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PaddleOCR-VL-0.9B 基座** | 0% | 0.0455 | 0.0455 | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
| **S400 (LoRA step 400)** | 0% | 0.1820 | 0.1862 | 0.2501 | 0.1302 | 0.8298 | 20.5% | 95.5% |
| **S600 (LoRA step 600) ★** | 0% | **0.2061** | **0.2024** | **0.3114** | **0.1540** | **0.8031** | **15.9%** | **90.9%** |
| **S800 (LoRA step 800)** | 0% | 0.2080 | 0.2862 | 0.1996 | 0.1191 | 0.8063 | 40.9% | 93.2% |

- **S600 检查点表现最优**：CompF1 较基座提升 4.5×，Token 召回率提升 96×，NED 相对误差降低 13.6%。
- **S800 出现过拟合**：重复率（RepRate）上升至 40.9%。
- **ExactMatch 均为 0%**：模型能定位并识别主要元器件，但暂无法单次无错重建整张复杂图纸。

---

## 数据集构成与划分

| 数据集划分 | 样本量 | 平均行数 | 平均字符数 | 数据构成与说明 |
| :--- | :---: | :---: | :---: | :--- |
| **训练集 (Train)** | 1,554 | 32.1 | 265 | 包含 KiCad 真实原理图 + 300 张合成文字对抗图 |
| **验证集 (Val)** | 171 | 33.4 | 278 | 独立原理图样本，经 7 轮质量校验 |
| **测试集 (Test)** | 44 / 89 | 30.5 | 250 | clean PNG 测试子集（easy50-pure: 44, easy100-pure: 89） |

---

## 数据收集与清洗流程

整个数据集经过 7 步递进式质量控制流程构建，确保标注质量和数据多样性：

```
KiCad 原理图设计
      │
      ▼
  SVG 矢量导出
      │
      ▼
  stroked-text 文字提取 ─── 自动提取所有可见文字
      │
      ▼
  初版 GT 标注生成
      │
      ▼
  第 1-3 轮：自动化校验 ─── 格式检查、路径验证、非法字符过滤
      │
      ▼
  第 4-5 轮：半自动清洗 ─── 元件标号正则匹配、参数值单位标准化
      │
      ▼
  第 6 轮：对比校验 ─── 与原始 KiCad 网表交叉验证
      │
      ▼
  第 7 轮：人工抽检 ─── 随机抽取 10% 样本逐行校对（标注准确率 > 99%）
      │
      ▼
  最终数据集 (Dataset A) ─── 混入 20% 合成文字对抗图片防模态塌缩
```

### 7 轮递进式质量验证

| 轮次 | 校验方法 | 检验工具/方式 | 覆盖率 | 发现与解决问题 |
|:----:|------|------|:----:|------|
| 1 | JSON 格式校验 | Python `json.loads` + schema 验证 | 100% | 解析失败样本直接剔除 |
| 2 | 图片路径验证 | `os.path.exists` 遍历 | 100% | 缺失图片标记并重新导出 |
| 3 | 非法字符过滤 | 正则匹配控制字符/乱码 | 100% | Unicode 替换、BOM 头清理 |
| 4 | 元件标号匹配 | `\b([RCDLQUJYF])\d+\b` 正则 | 100% | 检测标号格式不一致 |
| 5 | 参数值标准化 | 单位归一化（Ω/F/H/V/A） | 100% | `10k`/`10K`/`10kΩ` 统一 |
| 6 | KiCad 网表交叉验证 | 网表导出对比 | 100% | 缺失元件、多余元件标记 |
| 7 | 人工抽检 | 随机抽取 10%（152 样本）逐行校对 | 10% | 发现并修正边缘 case，准确率 > 99% |

---

## 数据集文件清单

- `ocr_vl_sft-train-v9-pure.jsonl` — 1,554 训练样本
- `ocr_vl_sft-val-v9-pure.jsonl` — 171 验证样本
- `ocr_vl_sft-synthetic-v3.jsonl` — 500 合成文字样本（参考）

---

## 引用

```bibtex
@misc{zhang2026circuitocr,
  title={CircuitOCR: LoRA Fine-Tuning PaddleOCR-VL for Circuit Schematic Understanding},
  author={Jianning Zhang},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/ZhangJ83/circuit-ocr-paddle}}
}
```

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。
