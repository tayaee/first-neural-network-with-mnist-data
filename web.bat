@echo off
set UV_CACHE_DIR=%CD%\..\..\..\..\..\.uv-cache
set TF_ENABLE_ONEDNN_OPTS=0
uv run --active streamlit run web.py --server.port 80
