> [中文版](./README.md)

# CI2N Datasets

**The official public datasets and benchmark for the Image2Net paper.** This repository supports training and evaluation for analog circuit diagram-to-netlist conversion, including device identification, crossing identification, device orientation classification, and netlist validation data.

[Paper abstract](https://arxiv.org/abs/2508.13157) | [PDF](https://arxiv.org/pdf/2508.13157) | [HTML paper](https://arxiv.org/html/2508.13157) | [Chinese README](./README.md)

## Associated Paper

**Image2Net: Datasets, Benchmark and Hybrid Framework to Convert Analog Circuit Diagrams into Netlists**

Haohang Xu, Chengjie Liu, Qihang Wang, Wenhao Huang, Yongjian Xu, Weiyu Chen, Anlan Peng, Zhijun Li, Bo Li, Lei Qi, Jun Yang, Yuan Du, Li Du

- arXiv: [`2508.13157`](https://arxiv.org/abs/2508.13157)
- Current revision: v2, updated on 2025-12-08
- Subjects: `cs.AR` (primary), `cs.AI`, `cs.CV`, and `eess.IV`
- Paper length: 10 pages, 12 figures, and 6 tables
- Official relationship: both the abstract and paper body identify this repository as the public location for the datasets and benchmark

The paper introduces the Image2Net hybrid framework and the Netlist Edit Distance (NED) metric for converting multi-style, complex analog circuit diagrams into netlists and measuring topology differences. This repository publishes the paper's datasets and evaluation material; it should not be mistaken for the complete Image2Net framework code repository.

## Dataset Contents

```text
ci2n_datasets/
├── device_identification/   # Circuit-device detection and classification
├── device_orientation/      # Device orientation and mirroring classification
├── jumper_identification/   # Crossing / jumper identification
└── validation/              # Netlist ground truth and NED/GED tools
```

| Directory | Role in the paper | Primary formats |
|---|---|---|
| `device_identification` | Locate and classify devices in circuit diagrams | Images, Labelme JSON |
| `device_orientation` | Determine orientation and mirroring for MOS, BJT, Diode, AMP, and related devices | Class-organized JPG/PNG |
| `jumper_identification` | Recognize line crossings and jumpers across drawing styles | Images, YOLO OBB labels |
| `validation` | Evaluate converted netlists against manually checked ground truth and compute NED | PNG, JSON, Python |

The paper's v2 statistics report 2,914 circuit diagrams and 84,195 annotations. The crossing dataset covers 1,983 images and 28,195 crossings, while orientation and mirroring classification contains 12,086 annotations. The paper also describes 104 manually verified circuit diagram/netlist pairs for evaluation. The repository continues to evolve, so its current file counts may differ from the publication-time statistics.

## Annotation Formats

### Device Identification

Device-detection annotations use Labelme JSON with labels and rectangular bounds:

```json
{
  "shapes": [
    {
      "label": "resistor",
      "points": [[597.0, 298.0], [668.0, 339.0]],
      "shape_type": "rectangle"
    }
  ],
  "imageHeight": 400,
  "imageWidth": 800
}
```

### Crossing / Jumper Identification

Each YOLO OBB row describes one oriented bounding box:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

- `class_id`: crossing category
- remaining eight values: normalized coordinates of the four corners

### Netlist Validation

JSON files under `validation/golden` describe device types and port connections:

```json
{
  "ckt_netlist": [
    {
      "component_type": "NMOS",
      "port_connection": {
        "Drain": "net8",
        "Gate": "net3",
        "Source": "net1"
      }
    }
  ]
}
```

## Benchmark and NED

The paper converts netlists into heterogeneous graphs, computes errors with Graph Edit Distance (GED), and normalizes the result by the total number of devices, nets, and ports in the ground-truth circuit to obtain NED. A lower NED means the predicted netlist is topologically closer to the ground truth.

Validation scripts are available under `validation/`. Review that directory's dependency files and script arguments before running them in an isolated Python environment. GED can be computationally expensive for complex graphs.

## Citation

If this repository supports your research, cite the Image2Net paper:

```bibtex
@misc{xu2025image2netdatasetsbenchmarkhybrid,
  title        = {Image2Net: Datasets, Benchmark and Hybrid Framework to Convert Analog Circuit Diagrams into Netlists},
  author       = {Haohang Xu and Chengjie Liu and Qihang Wang and Wenhao Huang and Yongjian Xu and Weiyu Chen and Anlan Peng and Zhijun Li and Bo Li and Lei Qi and Jun Yang and Yuan Du and Li Du},
  year         = {2025},
  eprint       = {2508.13157},
  archivePrefix= {arXiv},
  primaryClass = {cs.AR},
  url          = {https://arxiv.org/abs/2508.13157}
}
```

## License

Repository code and material are released under the [Apache License 2.0](./LICENSE). Data users must also follow the requirements associated with original source material, paper citation, and any applicable data licenses.

## Feedback

Report dataset, annotation, or evaluation-script issues through [GitHub Issues](https://github.com/LAD021/ci2n_datasets/issues).
