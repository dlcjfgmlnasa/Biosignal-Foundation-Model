#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
"""Phase 2 DDP launcher (V1/V2 자동 선택).

Usage (V1, 기본):
    python launch_phase2.py --device cuda:0 ...

Usage (V2):
    python launch_phase2.py --model_version v2 --device cuda:0 ...
"""
import runpy
import sys

if "" not in sys.path:
    sys.path.insert(0, "")

# --model_version 인자 추출 (argparse에 전달되기 전에 빼냄)
version = "v1"
if "--model_version" in sys.argv:
    idx = sys.argv.index("--model_version")
    version = sys.argv[idx + 1]
    del sys.argv[idx:idx + 2]

module = f"train.{version}_2_any_variate"
runpy.run_module(module, run_name="__main__")
