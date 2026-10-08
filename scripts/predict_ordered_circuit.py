"""Research-only ordered rectification; geometry acceptance is not independent verification."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import predict_contrast_edge as inference
from scripts.ordered_line_rectification_v2 import register_ordered


if __name__=='__main__':
    inference.register_guarded=register_ordered;inference.main()
