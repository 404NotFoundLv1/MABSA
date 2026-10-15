#!/usr/bin/env python3
"""Independent frozen-baseline OCR pilot; Dev only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mact.ocr_residual_train import main

if __name__=='__main__':main()
