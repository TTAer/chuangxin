# LogicBench-1K

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/samples-1000-blue)](metadata/statistics.json)
[![Version](https://img.shields.io/badge/version-1.0.1-green)](VERSION)
[![GitHub](https://img.shields.io/badge/GitHub-tkcl--research%2FLogicBench1k-blue)](https://github.com/tkcl-research/LogicBench1k)
[![Hugging Face](https://img.shields.io/badge/🤗%20Datasets-TKCL--HF%2FLogicBench--1K-yellow)](https://huggingface.co/datasets/TKCL-HF/LogicBench-1K)

**LogicBench-1K** is a benchmark corpus of **1,000 digital logic circuit diagrams** with **topology-level ground-truth graph annotations**, introduced in:

> **Stroke-Level Connectivity Verification: Grounding Vision-Language Models Against Topology Hallucination in Diagram Understanding**  
> Abdullah Ibne Hanif Arean, Niamul Hassan Samin, Md Arifur Rahman, Renu Akter Suity, Juena Ahmed Noshin, and Md Ashikur Rahman  
> *International Conference on Document Analysis and Recognition (ICDAR), 2026*

The dataset supports systematic evaluation of **symbol detection**, **graph recovery (connectivity)**, and related document-AI tasks where **pixel-grounded topological fidelity** matters.

## Release (camera-ready)

LogicBench-1K is released with the camera-ready version of the paper. The public package is available on [GitHub](https://github.com/tkcl-research/LogicBench1k) and [Hugging Face](https://huggingface.co/datasets/TKCL-HF/LogicBench-1K).

| Component | Location |
|-----------|----------|
| Images (1,000) | [`images/`](images/) |
| JSON ground truth | [`annotations/`](annotations/) |
| Annotation schema | [`schema/annotation.schema.json`](schema/annotation.schema.json), [`docs/ANNOTATION_SCHEMA.md`](docs/ANNOTATION_SCHEMA.md) |
| Official split (700 / 100 / 200) | [`annotations/splits/`](annotations/splits/) |
| Dataset statistics (tables) | [`metadata/STATISTICS.md`](metadata/STATISTICS.md) |
| Dataset statistics (JSON) | [`metadata/statistics.json`](metadata/statistics.json) |
| Release manifest | [`metadata/dataset_manifest.json`](metadata/dataset_manifest.json) |

## Overview

| Property | Value |
|----------|-------|
| Samples | 1,000 |
| Image format | JPEG, 512×512 px |
| Annotation format | JSON (directed graph per image) |
| Train / Val / Test | 700 / 100 / 200 |
| Gate vocabulary | `AND`, `OR`, `NOT`, `NAND`, `NOR`, `XOR`, `XNOR` |
| Total gates / edges | 6,174 / 6,423 |
| Unique gate mixtures | 343 (34.3% coverage) |
| Gates per diagram | 1-31 (mean 6.174, median 5.0) |
| Edges per diagram | 0-57 (mean 6.423, median 5.0) |
| Inter-annotator agreement (100-image subset) | κ = 0.91 (gates), κ = 0.88 (connectivity) |

Full tables and split-stratified breakdown: [`metadata/STATISTICS.md`](metadata/STATISTICS.md).

LogicBench-1K is designed for research on **connectivity hallucination** in vision-language models and for benchmarking verification frameworks such as **SLCV** (Stroke-Level Connectivity Verification) and **PaRCO** (Parallel Reconstruction and Constraint-Oriented reasoning).

## Repository layout

```
LogicBench-1K/
├── images/
├── annotations/
│   └── splits/
├── metadata/
│   ├── STATISTICS.md
│   ├── statistics.json
│   └── dataset_manifest.json
├── schema/
├── scripts/
│   ├── validate_dataset.py
│   └── generate_statistics.py
├── docs/
├── README.md
├── LICENSE
└── VERSION
```

## Quick start

### Load a single sample

```python
import json
from pathlib import Path

sample_id = "lb1k_00042"
root = Path(".")

with open(root / "annotations" / f"{sample_id}.json") as f:
    graph = json.load(f)

image_path = root / graph["image_path"]
nodes, edges = graph["nodes"], graph["edges"]
```

### Iterate a split

```python
from pathlib import Path

split_path = Path("annotations/splits/train.txt")
for line in split_path.read_text().splitlines():
    sample_id = line.strip()
    ann = Path("annotations") / f"{sample_id}.json"
    img = Path("images") / f"{sample_id}.jpg"
```

### Validate and regenerate statistics

```bash
python3 scripts/validate_dataset.py
python3 scripts/generate_statistics.py
```

## Dataset statistics

Camera-ready statistics are published in two formats:

- **[`metadata/STATISTICS.md`](metadata/STATISTICS.md)**: publication tables for sample diversity, gate-type frequencies, edge histograms, and per-split summaries; includes a paper-ready summary paragraph.
- **[`metadata/statistics.json`](metadata/statistics.json)**: machine-readable report (`schema_version` 1.1) with metric definitions, `sample_diversity`, `topology_distribution` (global + by split), and `paper_summary`.

Key metrics:

| Category | Metrics |
|----------|---------|
| Sample diversity | Unique gate mixtures, mixture coverage, node/edge count distributions (min, max, mean, median, std, Q1/Q3), histograms |
| Topology distribution | Gate-type counts and proportions, directed graph density, terminal in/out-degree histograms |
| Splits | Stratified statistics for train (700), val (100), test (200) |

Regenerate after any annotation change:

```bash
python3 scripts/generate_statistics.py
```

## Annotation format

Each file `annotations/lb1k_XXXXX.json` describes a **directed connectivity graph**:

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | array | Logic gates; each has `id` (e.g. `gate_0`) and `type` (gate class) |
| `edges` | array | Directed wires; each has `source` and `target` node IDs |
| `image_path` | string | Relative path to the paired image (`images/lb1k_XXXXX.jpg`) |

**Example** (abbreviated):

```json
{
  "nodes": [
    { "id": "gate_0", "type": "AND" },
    { "id": "gate_1", "type": "OR" }
  ],
  "edges": [
    { "source": "gate_0", "target": "gate_1" }
  ],
  "image_path": "images/lb1k_00001.jpg"
}
```

Full specification: [`docs/ANNOTATION_SCHEMA.md`](docs/ANNOTATION_SCHEMA.md)  
JSON Schema: [`schema/annotation.schema.json`](schema/annotation.schema.json)

## Splits

Official splits are **disjoint** and sum to 1,000 samples:

| Split | File | Count | Sample ID range |
|-------|------|-------|-----------------|
| Test | `annotations/splits/test.txt` | 200 | `lb1k_00000`-`lb1k_00199` |
| Train | `annotations/splits/train.txt` | 700 | `lb1k_00200`-`lb1k_00899` |
| Validation | `annotations/splits/val.txt` | 100 | `lb1k_00900`-`lb1k_00999` |

Use these files for comparable reporting with the reference paper. Do not merge train and validation for final test evaluation.

## Benchmark tasks

LogicBench-1K defines three evaluation tasks (see paper §4.1):

1. **Symbol detection**: localise and classify gates (IoU >= 0.5 for localisation).
2. **Graph recovery**: reconstruct directed connectivity; report **Connectivity F1** and **Netlist Edit Distance (NED)**.
3. **Functional equivalence**: reserved for future work in the reference publication.

Metrics and baseline tables: [`docs/BENCHMARK_TASKS.md`](docs/BENCHMARK_TASKS.md).

## Dataset construction (summary)

Per the reference paper:

- Source schematics from educational materials were annotated by **two domain experts**.
- The corpus includes **digital-native** (*D*<sub>clean</sub>: vector-derived raster) and **scanned-legacy** (*D*<sub>noisy</sub>: scanning artefacts, affine skew ±15°, salt-and-pepper noise) variants to study robustness under degradation.
- Inter-annotator agreement was measured on a **100-image doubly annotated subset** (Cohen's κ = 0.91 for gates, κ = 0.88 for connectivity).
- Annotations encode **logical connectivity** (which gate output feeds which gate input), not geometric wire polylines.

## Citation

If you use LogicBench-1K, please cite:

> **Stroke-Level Connectivity Verification: Grounding Vision-Language Models Against Topology Hallucination in Diagram Understanding**  
> Abdullah Ibne Hanif Arean, Niamul Hassan Samin, Md Arifur Rahman, Renu Akter Suity, Juena Ahmed Noshin, and Md Ashikur Rahman  
> *International Conference on Document Analysis and Recognition (ICDAR), 2026*

```bibtex
@inproceedings{arean2026slcv,
  author    = {{Abdullah Ibne Hanif Arean} and {Niamul Hassan Samin} and {Md Arifur Rahman} and {Renu Akter Suity} and {Juena Ahmed Noshin} and {Md Ashikur Rahman}},
  title     = {Stroke-Level Connectivity Verification: Grounding Vision-Language Models Against Topology Hallucination in Diagram Understanding},
  booktitle = {International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2026}
}
```

## License

The LogicBench-1K dataset is released under the [Creative Commons Attribution 4.0 International License](LICENSE) (CC BY 4.0).

## Authors

Abdullah Ibne Hanif Arean, Niamul Hassan Samin, Md Arifur Rahman, Renu Akter Suity, Juena Ahmed Noshin, and Md Ashikur Rahman

## Contact

For questions, corrections, or access issues, open a GitHub issue in this repository or contact the author through The Kow Company Limited.

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| 1.0.1 | 2026-06-08 | Camera-ready statistics (`STATISTICS.md`, `statistics.json` schema v1.1); documentation alignment |
| 1.0.0 | 2026-05-20 | Initial public release; normalized `image_path` fields; repository packaging |

See [`VERSION`](VERSION) for the current release identifier.
