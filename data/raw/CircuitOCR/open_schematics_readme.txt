---
license: cc-by-4.0
task_categories:
- text-generation
- image-to-text
- text-to-image
language:
- en
tags:
- electronics
- schematics
- kicad
- hardware
- pcb
- circuit-design
- engineering
size_categories:
- 10K<n<100K
---

![Open Schematics](open-schematics.png)

# Open Schematics Dataset

The largest dataset of electronic schematics and PCB layouts on the internet, built as an engineering reference for schematic and PCB layout work. It's a self-growing, autonomous dataset that continuously scans the web for new engineering designs and updates itself accordingly.

## Dataset Description

Each record corresponds to one schematic file and includes the raw source, rendered images, structured metadata, and all associated PCB files from the same project.

### Dataset Structure

| Field | Type | Description |
|-------|------|-------------|
| `schematic` | string | Raw schematic file content (`.kicad_sch`, `.sch`, or `.SchDoc`) |
| `schematic_image` | image | Rendered PNG of the schematic (None if unavailable) |
| `pcb_files` | list[string] | Raw content of PCB files (`.kicad_pcb` / `.pcb`) from the same project |
| `pcb_images` | list[image] | Rendered PNGs of each PCB file, parallel to `pcb_files` (None per entry if unavailable) |
| `components_used` | list[string] | Component names extracted from library symbols (`.kicad_sch` only; None for other formats) |
| `schematic_json` | string | kiutils JSON representation of the schematic (`.kicad_sch` only; None otherwise) |
| `schematic_yaml` | string | kiutils YAML representation of the schematic (`.kicad_sch` only; None otherwise) |
| `name` | string | Source repository (`owner/repo`) |
| `description` | string | Repository description |
| `extensions_used` | list[string] | File extensions present: schematic extension first, then unique PCB extensions (e.g. `[".kicad_sch", ".kicad_pcb"]`) |

### Schematic formats included

- `.kicad_sch` — KiCad v5+ (primary format; includes JSON/YAML/components)
- `.sch` — KiCad v4 legacy
- `.SchDoc` — Altium Designer

### PCB filters applied

- PCB files larger than 10 MB are excluded (extreme outliers; affects <1% of files)
- Projects with more than 20 PCB files retain the 20 with the largest rendered image (highest detail), ranked by pixel area

## Dataset Creation

### Source Data

Schematics and PCB layouts collected from publicly available data from the web.

### Data Collection

Files were fetched and processed to extract:
- Raw schematic and PCB source files
- Rendered images (PNG/SVG → PNG)
- Structured metadata via [kiutils](https://github.com/mvnmgrx/kiutils) (JSON/YAML)
- Component lists from library symbol references

### Data Processing

- Empty `.kicad_sch` files (no library symbols or schematic symbols) are filtered out
- `.sch` and `.SchDoc` rows are included with raw text only; JSON/YAML/components are None when kiutils conversion is unavailable
- Component names are extracted from `libSymbols` fields and deduplicated per library prefix
- `extensions_used` is inferred from actual file paths on disk, not from YAML metadata fields

## Use Cases

- **Circuit Design AI**: Training models to generate or understand electronic schematics
- **PCB Layout Understanding**: Learning the relationship between schematics and their physical PCB layouts
- **Component Recognition**: Identifying electronic components from schematic and PCB diagrams
- **Design Assistance**: Building tools to help engineers with circuit and PCB design
- **Multimodal Learning**: Paired schematic text + image + PCB text + image for cross-modal tasks
- **Documentation Generation**: Automatically generating documentation from schematics
- **Design Validation**: Training models to detect errors or suggest improvements
- **Educational Tools**: Creating interactive learning systems for electronics education

## Limitations

- JSON/YAML/components are only populated for `.kicad_sch` format (~85% of rows)
- PCB images are unavailable for ~55% of PCB entries (no PNG or SVG rendered)
- Schematic images are unavailable for ~10% of rows
- Quality and complexity of designs vary significantly across projects
- Component naming conventions vary between projects and library sources

## License

This dataset is released under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license. Please ensure you comply with the original licenses of the source projects when using this data.

## Citation

If you use this dataset in your research or projects, please cite:

```bibtex
@dataset{open_schematics_2026,
  title={Open Schematics: A Dataset of Electronic Schematics and PCB Layouts from Hardware Projects},
  author={bshada},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/bshada/open-schematics}}
}
```

## Contact

For questions, issues, or contributions, please open an issue on the dataset repository.
