from __future__ import annotations

from typing import Any

import streamlit as st

from config import APP_TITLE
from ui_design_system.streamlit_style import apply_app_style
from panels import (
    render_panel_1_upload,
    render_panel_2_compression,
    render_panel_3_encoding,
    render_panel_4_experiment,
    render_panel_5_decoding,
    render_panel_6_analysis,
)


# Build marker for checking that Streamlit is running the correct file.
APP_BUILD_ID = "single_page_no_tabs_fixed_status_decode_update_v5"


APP_STEPS = [
    (1, "Input"),
    (2, "Compression"),
    (3, "Encoding"),
    (4, "Strand Design"),
    (5, "Decoding"),
    (6, "Summarization"),
]


def _has_meaningful_value(key: str) -> bool:
    """
    Return True when a session_state key exists and contains useful data.

    This avoids errors from pandas DataFrame or custom objects whose boolean
    value can be ambiguous.
    """
    if key not in st.session_state:
        return False

    value = st.session_state.get(key)

    if value is None:
        return False

    if isinstance(value, (bytes, bytearray, str, list, tuple, dict, set)):
        return len(value) > 0

    try:
        return bool(value)
    except Exception:
        return True


def _decode_successful() -> bool:
    """
    Decode is successful when decoding has produced output and no decode error
    is currently stored.

    Different restore branches may save slightly different keys, so this accepts
    restored_info, raw_restore_info, or decoded_data as completion signals.
    """
    if st.session_state.get("decode_error"):
        return False

    decoded_data_exists = (
        "decoded_data" in st.session_state
        and st.session_state.get("decoded_data") is not None
    )
    restored_info_exists = _has_meaningful_value("restored_info")
    raw_restore_info_exists = _has_meaningful_value("raw_restore_info")

    return restored_info_exists or raw_restore_info_exists or decoded_data_exists


def _pipeline_checks() -> dict[int, bool]:
    """
    Check whether each visible pipeline step has completed.

    Step 6 is automatic. Summarization becomes Done when Decoding succeeds.
    """
    decoded_ok = _decode_successful()

    return {
        1: _has_meaningful_value("input_bytes"),
        2: _has_meaningful_value("stored_bytes"),
        3: _has_meaningful_value("dna"),
        4: _has_meaningful_value("strand_rows"),
        5: decoded_ok,
        6: decoded_ok,
    }


def _step_state(step_no: int) -> tuple[str, str]:
    """
    Return CSS class and visible status text.

    Done    = the step output exists.
    Next    = this is the first incomplete step after all previous steps are done.
    Waiting = previous requirements are missing.
    Review  = decoding failed and needs user attention.
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
    Small spacing overrides for the single-page app.

    This CSS is rendered on every Streamlit rerun.
    """
    st.markdown(
        """
<style>
.block-container {
  padding-top: 1.15rem;
  padding-bottom: 2rem;
}

h1, h2, h3, h4 {
  letter-spacing: -0.02em;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _apply_pipeline_status_style() -> None:
    """
    Fixed status bar style.

    This CSS is rendered on every Streamlit rerun. Do not cache this with
    session_state because Streamlit rebuilds the page after upload, compression,
    encoding, strand design, and decoding.
    """
    st.markdown(
        """
<style>
.pipeline-fixed {
  position: fixed;
  top: 0.55rem;
  left: 50%;
  transform: translateX(-50%);
  width: min(1280px, calc(100vw - 3rem));
  z-index: 999999;
  background: rgba(255, 255, 255, 0.97);
  backdrop-filter: blur(10px);
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 0.62rem 0.65rem 0.6rem 0.65rem;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
}

.pipeline-fixed-title {
  text-align: center;
  font-size: 1.05rem;
  font-weight: 850;
  letter-spacing: -0.02em;
  color: #0f172a;
  margin: 0 0 0.48rem 0;
  line-height: 1.15;
}

.pipeline-status-spacer {
  height: 112px;
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
  .pipeline-fixed {
    width: calc(100vw - 2rem);
  }

  .pipeline-steps {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .pipeline-status-spacer {
    height: 172px;
  }
}

@media (max-width: 700px) {
  .pipeline-fixed {
    width: calc(100vw - 1rem);
    top: 0.35rem;
  }

  .pipeline-fixed-title {
    font-size: 0.95rem;
  }

  .pipeline-steps {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .pipeline-status-spacer {
    height: 235px;
  }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _pipeline_status_html() -> str:
    """Build the fixed title and six-step status bar with latest session_state."""
    parts = [
        f"<!-- {APP_BUILD_ID} -->",
        '<div class="pipeline-fixed">',
        '<div class="pipeline-fixed-title">DNA Data Storage System</div>',
        '<div class="pipeline-steps">',
    ]

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
    return "".join(parts)


def _render_pipeline_status(target: Any | None = None) -> None:
    """
    Render the fixed title and six-step status bar.

    With a placeholder target, the status appears at the top of the page but is
    filled after all panels run. This makes step 5 and 6 update immediately
    after render_panel_5_decoding() changes session_state.
    """
    html = _pipeline_status_html()

    if target is None:
        st.markdown(html, unsafe_allow_html=True)
    else:
        target.markdown(html, unsafe_allow_html=True)


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")

    apply_app_style()
    _render_compact_overrides()
    _apply_pipeline_status_style()

    # The placeholder is created before the panels so the status stays at top.
    # It is filled after the panels so it uses the newest session_state.
    status_placeholder = st.empty()
    st.markdown("<div class='pipeline-status-spacer'></div>", unsafe_allow_html=True)

    render_panel_1_upload()
    render_panel_2_compression()
    render_panel_3_encoding()
    render_panel_4_experiment()
    render_panel_5_decoding()
    render_panel_6_analysis()

    _render_pipeline_status(status_placeholder)


if __name__ == "__main__":
    render_app()
