#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
"""Phase 1 DDP launcher.

Usage (단일 GPU):
    python launch_phase1.py --device cuda:0 ...

Usage (멀티 GPU):
    torchrun --nproc_per_node=2 launch_phase1.py ...
"""
import runpy
import sys

if "" not in sys.path:
    sys.path.insert(0, "")

runpy.run_module("train.1_channel_independency", run_name="__main__")
