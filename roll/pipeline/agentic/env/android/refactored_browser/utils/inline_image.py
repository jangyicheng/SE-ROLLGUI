from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


@st.cache_data(show_spinner=False, max_entries=4096)
def _build_data_url(path_str: str, mtime_ns: int, size: int) -> str:
    _ = mtime_ns, size
    with open(path_str, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_image_inline(path: Path, use_container_width: bool = True):
    stat = path.stat()
    data_url = _build_data_url(str(path), stat.st_mtime_ns, stat.st_size)

    if use_container_width:
        style = "width:100%;height:auto;display:block;"
    else:
        style = "max-width:100%;height:auto;display:block;"

    st.markdown(
        f"<img src='{data_url}' style='{style}'/>",
        unsafe_allow_html=True,
    )
