# A Public Ground-Truth Dataset for Handwritten Circuit Diagrams (CGHD)
This repository contains images of hand-drawn electrical circuit diagrams as well as accompanying bounding box annotation, polygon annotation and segmentation files. These annotations serve as ground truth to train and evaluate several image processing tasks like object detection, instance segmentation and text detection. The purpose of this dataset is to facilitate the automated extraction of electrical graph structures from raster graphics.


## Structure
The folder and file structure is made up as follows:

```
gtdh-hd
│   README.md                   # This File
│   classes.json                # Classes List
│   classes_color.json          # Classes to Color Map
│   classes_discontinuous.json  # Classes Morphology Info
│   classes_ports.json          # Electrical Port Descriptions for Classes
│   consistency.py              # Dataset Statistics and Consistency Check
│   loader.py                   # Simple Dataset Loader and Storage Functions
│   segmentation.py             # Multiclass Segmentation Generation
│   utils.py                    # Helper Functions
│   requirements.txt            # Requirements for Scripts
│
└───drafter_D
│   └───annotations             # Bounding Box, Rotation and Text Label Annotations
│   │   │   CX_DY_PZ.xml
│   │   │   ...
│   │
│   └───images                  # Raw Images
│   │   │   CX_DY_PZ.jpg
│   │   │   ...
│   │
│   └───instances               # Instance Segmentation Polygons
│   │   │   CX_DY_PZ.json
│   │   │   ...
│   │
│   └───segmentation            # Binary Segmentation Maps (Strokes vs. Background)
│   │   │   CX_DY_PZ.jpg
│   │   │   ...
...
```

Where:

 - `D` is the (globally) running number of a drafter
 - `X` is the (globally) running number of the circuit (12 Circuits per Drafter)
 - `Y` is the Local Number of the Circuit's Drawings (2 Drawings per Circuit)
 - `Z` is the Local Number of the Drawing's Image (4 Pictures per Drawing)

Please Note: *The described scheme applies to all drafters with positive numbers. Drafters with negative or zero IDs have varying amounts of images.*

### Raw Image Files
Every raw image is RGB-colored and either stored as `jpg`, `jpeg` or `png` (both uppercase and lowercase suffixes exist). Raw images are always stored in sub-folders named `images`.

### Bounding Box Annotations
For every raw image in the dataset, there is an annotation file which contains **BB**s (Bounding Boxes) of **RoI**s (Regions of Interest) like electrical symbols or texts within that image. These BB annotations are stored in the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) format. Apart from its location in the image, every BB bears a class label, a complete list of class labels including a suggested mapping table to integer numbers for training and prediction purposes can be found in `classes.json`. As the bb annotations are the most basic and pivotal element of this dataset, they are stored in sub-folders named `annotations`.

Please Note: *For every Raw image in the dataset, there is an accompanying BB annotation file.*

Please Note: *The BB annotation files are also used to store symbol rotation and text label annotations as XML Tags that form an extension of the utilized PASCAL VOC format.*

#### Known Labeled Issues
 - C25_D1_P4 cuts off a text
 - C27 cuts of some texts
 - C29_D1_P1 has one additional text
 - C31_D2_P4 has a text less
 - C33_D1_P4 has a text less
 - C46_D2_P2 cuts of a text

### Binary Segmentation Maps
Binary segmentation images are available for some raw image samples and consequently bear the same resolutions as the respective raw images. The defined goal is to have a segmentation map for at least one of the images of every circuit. Binary segmentation maps are considered to contain black and white pixels only. More precisely, white pixels indicate any kind of background like paper (ruling), surrounding objects or hands and black pixels indicate areas of drawings strokes belonging to the circuit. As binary segmentation images are the only permanent type of segmentation map in this dataset, they are stored in sub-folders named `segmentation`.

### Polygon Annotations
For every binary segmentation map, there is an accompanying polygonal annotation file for instance segmentation purposes (that's why the polygon annotations are referred to as `instances` and stored in sub-folders of this name), which is stored in the [labelme](https://github.com/wkentaro/labelme) format. Note that the contained polygons are quite coarse, intended to be used in conjunction with the binary segmentation maps for connection extraction and to tell individual instances with overlapping BBs apart.

### Netlists
For some images, there are also netlist files available, which are stored in the [ASC](http://ltwiki.org/LTspiceHelp/LTspiceHelp/Spice_Netlist.htm) format.


## Processing Scripts
This repository comes with several python scripts. These have been tested with [Python 3.11](https://docs.python.org/3.11/). Before running them, please make sure all requirements are met (see `requirements.txt`).

### Consistency and Statistics
The consistency script performs data integrity checks and corrections as well as derives statistics for the dataset. The list of features include:

 - Ensure annotation files are stored uniformly
   - Same version of annotation file format being used
   - Same indent, uniform line breaks between tags (important to use `git diff` effectively)
 - Check Annotation Integrity
   - Classes referenced the (BB/Polygon) Annotations are contained in the central `classes.json` list
   - `text` Annotations actually contain a non-empty text label and text labels exist in `text` annotations only
   - Class Count between Pictures of the same Drawing are identical
   - Image Dimensions stated in the annotation files match the referenced images
 - Obtain Statistics
   - Class Distribution
   - BB Sizes
   - Image Size Distribustion
   - Text Character Distribution

The respective script is called without arguments to operate on the **entire** dataset:

```
python consistency.py
```

Note that due to a complete re-write of the annotation data, the script takes several seconds to finish. Therefore, the script can be restricted to an individual drafter, specified as CLI argument (for example drafter 15):

```
python consistency.py -d 15
```

In order to reduce the computational overhead and CLI prints, most functions are deactivated by default. In order to see the list of available options, run:

```
python consistency.py -h
```

### Multi-Class (Instance) Segmentation Processing
This dataset comes with a script to process both new and existing (instance) segmentation files. It is invoked as follows:

```
$ python3 segmentation.py <command> <drafter_id> <target> <source>
```

Where:

 - `<command>` has to be one of:
   - `transform`
     - Converts existing BB Annotations to Polygon Annotations
     - Default target folder: `instances`
     - Existing polygon files will not be overridden in the default settings, hence this command will take no effect in an completely populated dataset.
     - Intended to be invoked after adding new binary segmentation maps
       - **This step has to be performed before all other commands**
   - `wire`
     - Generates Wire Describing Polygons
     - Default target folder: `wires`
   - `keypoint`
     - Generates Keypoints for Component Terminals
     - Default target folder: `keypoints`
   - `create`
     - Generates Multi-Class segmentation Maps
     - Default target folder: `segmentation_multi_class`
   - `refine`
     - Refines Coarse Polygon Annotations to precisely match the annotated objects
     - Default target folder: `instances_refined`
     - For instance segmentation purposes
   - `pipeline`
     - executes `wire`,`keypoint` and `refine` stacked, with one common `source` and `target` folder
     - Default target folder: `instances_refined`
   - `assign`
     - Connector Point to Port Type Assignment by Geometric Transformation Matching
 - `<drafter_id>` **optionally** restricts the process to one of the drafters
 - `<target>` **optionally** specifies a divergent target folder for results to be placed in
 - `<source>` **optionally** specifies a divergent source folder to read from

Please note that source and target forlders are **always** subfolder inside the individual drafter folders. Specifying source and target folders allow to stack the results of individual processing steps. For example, to perform the entire pipeline for drafter 20 manually, use:

```
python3 segmentation.py wire 20 instances_processed instances
python3 segmentation.py keypoint 20 instances_processed instances_processed
python3 segmentation.py refine 20 instances_processed instances_processed
```

### Dataset Loader
This dataset is also shipped with a set of loader and writer functions, which are internally used by the segmentation and consistency scripts and can be used for training. The dataset loader is simple, framework-agnostic and has been prepared to be callable from any location in the file system. Basic usage:

```
from loader import read_dataset

db_bb = read_dataset()                    # Read all BB Annotations
db_seg = read_dataset(segmentation=True)  # Read all Polygon Annotations
db_bb_val = read_dataset(drafter=12)      # Read Drafter 12 BB Annotations

len(db_bb)  # Get The Amount of Samples
db_bb[5]    # Get an Arbitrary Sample

db = read_images(drafter=12)   # Returns a list of (Image, Annotation) pairs
db = read_snippets(drafter=12) # Returns a list of (Image, Annotation) pairs
```


## Citation
If you use this dataset for scientific publications, please consider citing us as follows:

```
@inproceedings{thoma2021public,
  title={A Public Ground-Truth Dataset for Handwritten Circuit Diagram Images},
  author={Thoma, Felix and Bayer, Johannes and Li, Yakun and Dengel, Andreas},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={20--27},
  year={2021},
  organization={Springer}
}
```


## How to Contribute
If you want to contribute to the dataset as a drafter or in case of any further questions, please send an email to: <johannes.bayer@mail.de>


## Guidelines
These guidelines are used throughout the generation of the dataset. They can be used as an instruction for participants and data providers.

### Drafter Guidelines
 - 12 Circuits should be drawn, each of them twice (24 drawings in total)
 - Most important: The drawing should be as natural to the drafter as possible
 - Free-Hand sketches are preferred, using rulers and drawing Template stencils should be avoided unless it appears unnatural to the drafter
 - The sketches should not be traced directly from a template (e.g. from the Original Printed Circuits)
 - Minor alterations between the two drawings of a circuit (e.g. shifting a wire line) are encouraged within the circuit's layout as long as the circuit's function is preserved (only if the drafter is familiar with schematics)
 - Different types of pens/pencils should be used for different drawings
 - Different kinds of (colored, structured, ruled, lined) paper should be used
 - One symbol set (European/American) should be used throughout one drawing (consistency)
 - It is recommended to use the symbol set that the drafter is most familiar with
 - It is **strongly** recommended to share the first one or two circuits for review by the dataset organizers before drawing the rest to avoid problems (complete redrawing in worst case)

### Image Capturing Guidelines
 - For each drawing, 4 images should be taken (96 images in total per drafter)
 - Angle should vary
 - Lighting should vary
 - Moderate (e.g. motion) blur is allowed
 - All circuit-related aspects of the drawing must be _human-recognizable_
 - The drawing should be the main part of the image, but _naturally_ occurring objects from the environment are welcomed
 - The first image should be _clean_, i.e. ideal capturing conditions
 - Kinks and Buckling can be applied to the drawing between individual image capturing
 - Try to use the file name convention (`CX_DY_PZ.jpg`) as early as possible
   - The circuit range `X` will be given to you
   - `Y` should be `1` or `2` for the drawing
   - `Z` should be `1`,`2`,`3` or `4` for the picture

### Object Annotation Guidelines
 - General Placement
   - A **RoI** must be **completely** surrounded by its **BB**
   - A **BB** should be as tight as possible to the **RoI**
   - In case of connecting lines not completely touching the symbol, the BB should be extended (only by a small margin) to enclose those gaps (especially considering junctions)
   - Characters that are part of the **essential symbol definition** should be included in the BB (e.g. the `+` of a polarized capacitor should be included in its BB)
 - **Junction** annotations 
   - Used for actual junction points (Connection of three or more wire segments with a small solid circle)
   - Used for connection of three or more straight line wire segments where a physical connection can be inferred by context (i.e. can be distinguished from **crossover**)
   - Used for wire line corners
   - Redundant Junction Points should **not** be annotated (small solid circle in the middle of a straight line segment)
   - Should not be used for corners or junctions that are part of the symbol definition (e.g. Transistors)
 - **Crossover** Annotations
   - If dashed/dotted line: BB should cover the two next dots/dashes
 - **Text** annotations
   - Individual Text Lines should be annotated Individually
   - Text Blocks should only be annotated If Related to Circuit or Circuit's Components
   - Semantically meaningful chunks of information should be annotated Individually
     - component characteristics enclosed in a single annotation (e.g. __100Ohms__, __10%__ tolerance, __5V__ max voltage)
     - Component Names and Types (e.g. __C1__, __R5__, __ATTINY2313__)
     - Custom Component Terminal Labels (i.e. __Integrated Circuit__ Pins)
     - Circuit Descriptor (e.g. "Radio Amplifier")
   - Texts not related to the Circuit should be ignored
     - e.g. Brief paper, Company Logos
     - Drafters auxiliary markings for internal organization like "D12"
     - Texts on Surrounding or Background Papers
   - Characters which are part of the essential symbol definition should __not__ be annotated as Text dedicatedly
     - e.g. Schmitt Trigger __S__, , and gate __&__, motor __M__, Polarized capacitor __+__
     - Only add terminal text annotation if the terminal is not part of the essential symbol definition
   - **Table** cells should be annotated independently
 - **Operation Amplifiers**
   - Both the triangular US symbols and the european IC-like symbols for OpAmps should be labeled `operational_amplifier`
   - The `+` and `-` signs at the OpAmp's input terminals are considered essential and should therefore not be annotated as texts
 - **Complex Components**
   - Both the entire Component and its sub-Components and internal connections should be annotated:

     | Complex Component | Annotation                                             |
     | ----------------- | ------------------------------------------------------ |
     | Optocoupler       | 0. `optocoupler` as Overall Annotation                 |
     |                   | 1. `diode.light_emitting`                              |
     |                   | 2. `transistor.photo` (or `resistor.photo`)            |
     |                   | 3. `optical` if LED and Photo-Sensor arrows are shared |
     |                   |    Then the arrows  area should be includes in all     |
     | Relay             | 0. `relay` as Overall Annotation                       |
     | (also for         | 1. `inductor`                                          |
     | coupled switches) | 2. `switch`                                            |
     |                   | 3. `mechanical` for the dashed line between them       |
     | Transformer       | 0. `transformer` as Overall Annotation                 |
     |                   | 1. `inductor` or `inductor.coupled` (watch the dot)    |
     |                   | 3. `magnetic` for the core                             |


#### Rotation Annotations
The Rotation (integer in degree) should capture the overall rotation of the symbol shape. However, the position of the terminals should also be taken into consideration. Under idealized circumstances (no perspective distortion and accurately drawn symbols according to the symbol library), these two requirements equal each other. For pathological cases however, in which shape and the set of terminals (or even individual terminals) are conflicting, the rotation should compromise between all factors.

Rotation annotations are currently work in progress. They should be provided for at least the following classes:
- "voltage.dc"
- "resistor"
- "capacitor.unpolarized"
- "diode"
- "transistor.bjt"

#### Text Annotations
 - The Character Sequence in the Text Label Annotations should describe the actual Characters depicted in the respective BB as Precisely as Possible
 - BB Annotations of class `text`
 - Bear an additional `<text>` tag in which their content is given as string
 - The `Omega` and `Mikro` Symbols are escaped respectively
 - Currently Work in Progress
 - The utils script allows for migrating text annotations from one annotation file to another: `python3 utils.py source target`


### Segmentation Map Guidelines
 - Areas of __Intended__ drawing strokes (ink and pencil abrasion respectively) should be marked black, all other pixels (background) should be white
 - shining through the paper (from the rear side or other sheets) should be considered background

### Polygon Annotation Guidelines

0. Before starting, make sure the respective files exist for the image sample to be polygon-annotated:
   - BB Annotations (Pascal VOC XML File)
   - (Binary) Segmentation Map
1. Transform the BB annotations into raw polygons
   - Use: `python3 segmentation.py transform`
2. Refine the Polygons
   - **To Avoid Embedding Image Data into the resulting JSON**, use: `labelme --nodata`
   - Just make sure there are no overlaps between instances
   - Especially take care about overlaps with structural elements like junctions and crossovers
3. Generate Multi-Class Segmentation Maps from the refined polygons
   - Use: `python3 segmentation.py create`
   - Use the generated images for a visual inspection
   - After spotting problems, continue with Step 2

### Terminal Annotation Guidelines

```
labelme --labels "connector" --config "{shift_auto_shape_color: 1}" --nodata
```


## Licence
The entire content of this repository, including all image files, annotation files as well as sourcecode, metadata and documentation has been published under the [Creative Commons Attribution Share Alike Licence 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
