"""Multicontrast image-only guard with the fixed paired-model inference interface."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import predict_contrast_edge as inference
from scripts.circuit_geometry_guard_v2 import register_guarded_v2


if __name__=='__main__':
    inference.register_guarded=register_guarded_v2;inference.main()
