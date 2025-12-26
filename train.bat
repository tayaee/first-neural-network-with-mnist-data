@echo off
set UV_CACHE_DIR=%CD%\..\..\..\..\..\.uv-cache
set TF_CPP_MIN_LOG_LEVEL=3
uv run --active python train.py
