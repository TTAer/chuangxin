"""Use multiband image registration with the frozen reader and conservative output."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import predict_registered_circuit as inference
from scripts.circuit_registration_v2 import register_image_v2


if __name__=='__main__':
    inference.register_image=register_image_v2
    inference.main()
