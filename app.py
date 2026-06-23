from __future__ import annotations

import importlib
from typing import Callable, Optional

import streamlit as st


try:
    from config import APP_TITLE
except Exception:
    APP_TITLE = "Compression-Aware DNA Storage Pipeline"


try:
    from ui_design_system.streamlit_style import apply_app_style
except Exception:
    try:
        from styles import apply_style as apply_app_style
    except Exception:
        def apply_app_style() -> None:
            return


APP_STEPS = [
    (1, "Input"),
    (2, "Compression"),
    (3, "Encoding"),
    (4, "Strand Design"),
    (5, "Decoding"),
    (6, "Summarization"),
]


def _has_any_session_key(*keys: str) -> bool:
    """Return True if any listed session_state key exists with meaningful content."""
    for key in keys:
        if key not in st.session_state:
            continue

        value = st.session_state.get(key)

        if value is None:
            continue

        try:
            if isinstance(value, (bytes, bytearray, str, list, tuple, dict, set)):
                if len(value) > 0:
                    return True
                continue
        except Exception:
            pass

        # DataFrame, dataclass, pathlib Path, compression candidate objects, etc.
        try:
            if bool(value):
                return True
        except Exception:
            return True

    return False


def _decode_successful() -> bool:
    """
    Decode is considered complete when decoded/restored output exists
    and no decode error is stored.
    """
    if st.session_state.get("decode_error"):
        return False

    return _has_any_session_key(
        "decoded_data",
        "restored_info",
        "restored_file_path",
        "decoded_file_path",
        "decoded_output_path",
        "decoded_bytes",
        "restored_bytes",
    )


def _pipeline_checks() -> dict[int, bool]:
    """
    Flexible session-state checks for the three-tab app.

    Step 6 is automatic. Once step 5 Decoding is complete,
    step 6 Summarization is also complete because the summary/report panel
    is generated from the decode result.
    """
    decoded_ok = _decode_successful()

    return {
        1: _has_any_session_key(
            "input_bytes",
            "uploaded_bytes",
            "original_bytes",
            "source_bytes",
            "file_bytes",
            "input_file_path",
            "uploaded_file_path",
            "original_file_path",
            "uploaded_file",
        ),
        2: _has_any_session_key(
            "stored_bytes",
            "compressed_bytes",
            "selected_candidate",
            "compression_candidates",
            "stored_file_path",
            "compressed_file_path",
            "storage_method",
            "compression_result",
        ),
        3: _has_any_session_key(
            "dna",
            "encoded_dna",
            "dna_payload",
            "dna_sequence",
            "bits",
            "codec_meta",
            "encoded_metadata",
        ),
        4: _has_any_session_key(
            "strand_rows",
            "strands",
            "designed_strands",
            "strand_table",
            "strand_csv",
            "ngs_fragments",
        ),
        5: decoded_ok,
        6: decoded_ok,
    }


def _step_state(step_no: int) -> tuple[str, str]:
    """
    Visible statuses:
    - Done: the stage output exists.
    - Next: the first incomplete stage after all previous stages are done.
    - Waiting: prerequisites are missing.
    - Review: decoding failed.
    """
    checks = _pipeline_checks()

    if step_no == 5 and st.session_state.get("decode_error"):
        return "review", "Review"

    if checks.get(step_no, False):
        return "done", "Done"

    previous_done = all(checks.get(i, False) for i in range(1, step_no))

    if previous_done:
        return "current", "Next"

    return "waiting", "Waiting"


def _render_compact_overrides() -> None:
    """
    Compact UI overrides for the tab version.
    This is safe even if your design system already has styling.
    """
    if st.session_state.get("_compact_overrides_applied"):
        return

    st.session_state["_compact_overrides_applied"] = True

    st.markdown(
        """
<style>
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

h1, h2, h3, h4 {
  letter-spacing: -0.02em;
}

[data-testid="stVerticalBlock"] {
  gap: 0.75rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _apply_pipeline_status_style() -> None:
    """Sticky six-step pipeline status style."""
    if st.session_state.get("_pipeline_status_style_applied"):
        return

    st.session_state["_pipeline_status_style_applied"] = True

    st.markdown(
        """
<style>
.pipeline-sticky {
  position: sticky;
  top: 0.55rem;
  z-index: 999;
  background: rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(10px);
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 0.55rem;
  margin: 0.55rem 0 1rem 0;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}

.pipeline-steps {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.45rem;
}

.pipeline-step {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 0.52rem 0.58rem;
  background: #f8fafc;
  min-height: 56px;
}

.step-main {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  min-width: 0;
}

.step-num {
  width: 22px;
  height: 22px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 0.78rem;
  background: #e2e8f0;
  color: #334155;
  flex: 0 0 auto;
}

.step-name {
  font-weight: 750;
  font-size: 0.83rem;
  color: #0f172a;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.step-state {
  margin-top: 0.32rem;
  font-size: 0.75rem;
  font-weight: 800;
  color: #64748b;
}

.pipeline-step.done {
  background: #ecfdf5;
  border-color: #86efac;
}

.pipeline-step.done .step-num {
  background: #22c55e;
  color: white;
}

.pipeline-step.done .step-state {
  color: #15803d;
}

.pipeline-step.current {
  background: #eff6ff;
  border-color: #93c5fd;
}

.pipeline-step.current .step-num {
  background: #2563eb;
  color: white;
}

.pipeline-step.current .step-state {
  color: #1d4ed8;
}

.pipeline-step.waiting {
  background: #f8fafc;
  border-color: #e5e7eb;
  opacity: 0.82;
}

.pipeline-step.review {
  background: #fff7ed;
  border-color: #fdba74;
}

.pipeline-step.review .step-num {
  background: #f97316;
  color: white;
}

.pipeline-step.review .step-state {
  color: #c2410c;
}

@media (max-width: 1100px) {
  .pipeline-steps {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .pipeline-sticky {
    position: static;
  }

  .pipeline-steps {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
<div style="
  border: 1px solid #e5e7eb;
  border-radius: 20px;
  padding: 1.1rem 1.25rem;
  margin-bottom: 0.7rem;
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
">
  <div style="font-size: 1.45rem; font-weight: 850; color: #0f172a;">
    🧬 DNA Storage Pipeline
  </div>
  <div style="font-size: 0.95rem; color: #475569; margin-top: 0.25rem;">
    Compression, DNA encoding, strand design, decoding, and summarization.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_pipeline_status() -> None:
    """Render sticky six-step status."""
    parts = ['<div class="pipeline-sticky"><div class="pipeline-steps">']

    for number, label in APP_STEPS:
        css_class, state_text = _step_state(number)
        parts.append(
            f"""
<div class="pipeline-step {css_class}">
  <div class="step-main">
    <span class="step-num">{number}</span>
    <span class="step-name">{label}</span>
  </div>
  <div class="step-state">{state_text}</div>
</div>
"""
        )

    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _find_callable(function_name: str, module_names: list[str]) -> Optional[Callable[[], None]]:
    """
    Try to load a branch-rendering function from common project modules.

    This prevents the app from crashing immediately if the function location
    differs between versions.
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name, None)
            if callable(func):
                return func
        except Exception:
            continue
    return None


def _render_default_pipeline_fallback() -> None:
    """
    Fallback for projects that use the six-panel panels.py layout instead of
    separate image/text/audio branch functions.
    """
    try:
        from panels import (
            render_panel_1_upload,
            render_panel_2_compression,
            render_panel_3_encoding,
            render_panel_4_experiment,
            render_panel_5_decoding,
            render_panel_6_analysis,
        )

        render_panel_1_upload()
        render_panel_2_compression()
        render_panel_3_encoding()
        render_panel_4_experiment()
        render_panel_5_decoding()
        render_panel_6_analysis()
    except Exception as exc:
        st.error("Could not load the default six-panel pipeline.")
        st.exception(exc)


def render_image_branch() -> None:
    func = _find_callable(
        "render_image_branch",
        ["panels", "image_panel", "image_branch", "tab_image", "image_compression_panel"],
    )
    if func is not None:
        func()
        return

    st.info("Image branch function was not found. Showing the default pipeline instead.")
    _render_default_pipeline_fallback()


def render_text_branch() -> None:
    func = _find_callable(
        "render_text_branch",
        ["panels", "text_panel", "text_branch", "tab_text", "text_compression_panel"],
    )
    if func is not None:
        func()
        return

    st.warning(
        "Text branch function was not found. "
        "Please make sure render_text_branch() is available in one of your project modules."
    )


def render_audio_dna_storage_panel() -> None:
    func = _find_callable(
        "render_audio_dna_storage_panel",
        ["panels", "audio_panel", "audio_branch", "tab_audio", "audio_dna_storage_panel"],
    )
    if func is not None:
        func()
        return

    st.warning(
        "Audio branch function was not found. "
        "Please make sure render_audio_dna_storage_panel() is available in one of your project modules."
    )


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")

    apply_app_style()
    _render_compact_overrides()
    _apply_pipeline_status_style()
    _render_hero()
    _render_pipeline_status()

    tab_image, tab_text, tab_audio = st.tabs(["🖼️ Image", "📝 Text", "🎧 Audio"])

    with tab_image:
        render_image_branch()

    with tab_text:
        render_text_branch()

    with tab_audio:
        render_audio_dna_storage_panel()


if __name__ == "__main__":
    render_app()
