#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations

"""Phase 2 DDP launcher.

Usage (단일 GPU):
    python launch_phase2.py --device cuda:0 ...

Usage (멀티 GPU):
    torchrun --nproc_per_node=2 launch_phase2.py ...
"""
import runpy
import sys

if "" not in sys.path:
    sys.path.insert(0, "")

runpy.run_module("train.2_any_variate", run_name="__main__")
