"""Runtime diagnostics without loading vision dependencies during simulation."""
import os
import platform
import sys
import streamlit as st
from ui.theme import section_heading


def render_runtime():
    section_heading("COMPUTE RUNTIME", "Local workstation / Modal cloud")
    a, b, c = st.columns(3)
    a.metric("Environment", "MODAL" if os.environ.get("MODAL_TASK_ID") else "LOCAL")
    b.metric("Python", platform.python_version())
    c.metric("Platform", platform.system())
    if st.button("Probe GPU availability"):
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_properties(0)
                st.success(f"{gpu.name} · {gpu.total_memory / 1024**3:.1f} GiB · CUDA {torch.version.cuda}")
            else:
                st.info("CPU runtime. The autonomy simulator does not require a GPU.")
        except ImportError:
            st.info("PyTorch is not installed. Install the optional vision dependencies for inference.")
    st.caption(f"Interpreter: {sys.executable}")
    st.subheader("Modal deployment")
    st.code("python -m modal setup\npython -m modal deploy deployment/modal_app.py", language="bash")
    st.caption("The cloud image includes CUDA-enabled perception dependencies. Models and experiment artifacts persist in Modal volumes.")
