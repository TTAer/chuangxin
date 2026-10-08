---
language:
- en

license: mit

tags:
- PCB
- EDA
- KiCAD
- Hardware-Design
- Schematic-Generation
- LLM
- Circuit-Design

library_name: transformers
---

# SchGen

[![Technical Report](https://img.shields.io/badge/Paper-SchGen-b31b1b)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**SchGen** is a dataset of approximately **8K paired natural-language requests and Python-based schematic generation code** for research on LLM-driven PCB schematic generation.

The generated Python code can be rendered into **KiCad schematic designs**, enabling research on hardware generation from natural-language descriptions.

➡️ **Paper:** 
```
@misc{luo2026schgenpcbschematicgeneration,
      title={SchGen: PCB Schematic Generation with Semantic-Grounded Code Representations}, 
      author={Qinpei Luo and Ruichun Ma and Xinyu Zhang and Lili Qiu},
      year={2026},
      eprint={2605.30345},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.30345}, 
}
```  
➡️ **License:** MIT License  

---

## Overview

SchGen contains:

- 8,420 pairs of:
  - Natural-language hardware design requests
  - Python schematic-generation code

The dataset was developed for:

- Fine-tuning LLMs on schematic generation
- Benchmarking hardware-generation capabilities
- Research on AI-assisted PCB design workflows

The data was created between **August 2025 and September 2025**.

---

## Dataset Construction

The dataset was created through a multi-stage pipeline:

1. GPT-5 generated draft schematic designs based on open-source hardware references
2. Human annotation and correction ensured schematic correctness in KiCad
3. GPT-5 synthesized corresponding user requests from finalized schematics

Although open-source hardware references (e.g. SparkFun schematic PDFs under CC BY-SA 4.0) were used during drafting, the final dataset does not directly contain those original designs.

---

## Features

- 🔌 **KiCad-Compatible Generation**  
  Python code can be rendered into executable KiCad schematic designs.

- 🧠 **Natural Language to Hardware**  
  Enables research on generating PCB schematics directly from user requests.

- 📐 **Structured Schematic APIs**  
  Uses programmatic schematic construction APIs rather than raw schematic files.

- 🛠️ **Research-Oriented Dataset**  
  Designed for experimentation, benchmarking, and fine-tuning of LLMs.

---

## Evaluation

The dataset was evaluated using several schematic-generation metrics:

- **Valid Circuits**  
  Measures whether generated code executes successfully and produces valid schematics.

- **Spatial Violation**  
  Measures overlaps among schematic objects such as symbols, labels, and wires.

- **Netlist Accuracy**  
  Compares generated netlists with ground-truth schematic connectivity.

Evaluation was performed on `gpt-oss-20B` before and after fine-tuning on SchGen.

---

## Limitations

SchGen is intended for **research purposes only**.

Current limitations include:

- Primarily small and medium-scale schematic modules
- Limited support for:
  - RF/high-frequency circuits
  - cutting-edge industrial hardware
  - complex multi-board systems
- English-only natural-language requests
- Possible annotation or conversion errors from image/PDF-based schematic reconstruction

The dataset should not be used directly in safety-critical or highly regulated applications.

---

## License

This project is licensed under the MIT License.

---

## Contact

This project was conducted by members of Microsoft Research.

For questions, feedback, or collaboration inquiries:

- ruichunma@microsoft.com

If issues or problematic content are identified, the repository may be updated with appropriate mitigations.