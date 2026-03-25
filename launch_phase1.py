#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Phase 1 DDP launcher.

torchrun은 -m (module) 실행을 지원하지 않으므로,
이 스크립트를 통해 train.1_channel_independency를 실행한다.

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
