from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from compression_pipeline import CompressionCandidate, run_compression_benchmark
from config import WORK_ROOT, MAPPING_OPTIONS, DNA_PREVIEW_HEIGHT
from dna_codec import gc_content, homopolymer_stats
from dna_mapping import decode_dna_with_mapping, encode_bytes_to_dna, validate_container
from fragments import clean_dna, choose_auto_strand_design, prepare_dna_strands, strand_rows_to_csv
from restore_analysis import image_metrics, text_similarity, write_restored_file
from ui_helpers import download_bytes_button, fmt_bytes, get_domain, magic_dict, preview_file, save_upload, step_header, _write_preview_inner
from utils_bits_v2 import detect_magic, bytes_to_bitstring
from ui_design_system.ui_labels import (
    PANEL_TITLES, BUTTONS, METRICS, DATA_SOURCES, FIELDS, MESSAGES, DOWNLOAD_FILES, display_mapping
)
from ui_design_system.design_tokens import REGION_COLORS


# -----------------------------------------------------------------------------
# Small shared helpers.  Keep this file intentionally simple: each helper below
# is used by exactly one or more visible pipeline panels.
# -----------------------------------------------------------------------------

MAX_BINARY_PREVIEW_BYTES = 512
MAX_BITS_PREVIEW_CHARS = 4096


def _apply_panel_typography() -> None:
    if st.session_state.get("_panel_typography_applied"):
        return
    st.session_state["_panel_typography_applied"] = True
    st.markdown(
        """
<style>
html, body, [class*="css"] {
  font-family: Inter, "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stDataFrame, .stTable, [data-testid="stMetric"] {
  font-family: Inter, "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.84rem;
  letter-spacing: 0;
}
[data-testid="stMetricValue"] {
  font-size: 1.25rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _preview_seq(seq: str, n: int = 80) -> str:
    seq = clean_dna(seq)
    return seq[:n] + ("..." if len(seq) > n else "")


def _candidate_file(cand: CompressionCandidate) -> str:
    out_dir = WORK_ROOT / "selected_compression"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_method = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in cand.method)
    path = out_dir / f"selected_{safe_method}{cand.ext or '.bin'}"
    path.write_bytes(cand.data)
    return str(path)


def _selected_candidate_path(cand: CompressionCandidate) -> str:
    path = _candidate_file(cand)
    st.session_state["stored_file_path"] = path
    return path



def _display_mapping(mapping: str) -> str:
    return display_mapping(mapping)


def _decode_source() -> Tuple[str, str]:
    """Return (label, dna_text) for the currently selected decode source."""
    return "Original encoded DNA", st.session_state.get("dna", "")


def _dna_from_uploaded_strand_csv(data: bytes) -> Tuple[str, int, str]:
    text = bytes(data or b"").decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader]
    if not rows:
        return "", 0, "CSV has no strand rows."

    def row_no(row: Dict[str, Any]) -> int:
        try:
            return int(str(row.get("No.", row.get("No", "0")) or "0"))
        except Exception:
            return 0

    rows.sort(key=row_no)
    if "Payload" in rows[0]:
        dna = clean_dna("".join(str(row.get("Payload", "")) for row in rows))
        return dna, len(rows), "Loaded payload column from strand CSV."
    if "Full strand" in rows[0]:
        dna = clean_dna("".join(str(row.get("Full strand", "")) for row in rows))
        return dna, len(rows), "Loaded full-strand column; use Payload column for exact prepared-strand decode."
    return "", len(rows), "CSV must include a Payload or Full strand column."


def _bytes_to_bit_text(data: bytes, max_bytes: int = MAX_BINARY_PREVIEW_BYTES) -> str:
    raw = bytes(data or b"")
    shown = raw[:max_bytes]
    bits = bytes_to_bitstring(shown)
    if len(raw) > max_bytes:
        bits += f"\n\n... preview only: showing first {max_bytes:,} of {len(raw):,} bytes."
    return bits


def _bytes_to_full_bit_text(data: bytes) -> str:
    return bytes_to_bitstring(bytes(data or b""))


def _bits_preview_text(bits: str, max_chars: int = MAX_BITS_PREVIEW_CHARS) -> str:
    text = str(bits or "")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... preview only: showing first {max_chars:,} of {len(text):,} bits."
    return text


def _download_text_button(label: str, text: str, file_name: str) -> None:
    st.download_button(label, data=text.encode("utf-8"), file_name=file_name, mime="text/plain", use_container_width=True)


def _download_full_binary_button(label: str, data: bytes, file_name: str) -> None:
    st.download_button(
        label,
        data=_bytes_to_full_bit_text(data).encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        use_container_width=True,
    )


def _pipeline_file_metric_rows(path: str, data: bytes, *, compressed: bool = False) -> List[Dict[str, Any]]:
    rows = _file_info_rows(path, data)
    out: List[Dict[str, Any]] = []
    for row in rows:
        metric = str(row.get("Metric", ""))
        if metric in {"File name", "Container"}:
            continue
        if metric == "Size" and compressed:
            out.append({"Metric": "Compressed data", "Value": row.get("Value", "")})
        else:
            out.append(row)
    return out


def _rows_to_properties(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "Property": row.get("Metric", row.get("Property", "")),
            "Value": row.get("Value", ""),
        })
    return out


def _render_property_table(rows: List[Dict[str, Any]], height: int | None = None) -> None:
    if not rows:
        return
    kwargs = {"use_container_width": True, "hide_index": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(pd.DataFrame(_rows_to_properties(rows)), **kwargs)


def _render_stats_table(rows: List[Dict[str, Any]], height: int | None = None) -> None:
    if not rows:
        return
    kwargs = {"use_container_width": True, "hide_index": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(pd.DataFrame(rows), **kwargs)


def _format_chart_label(value: float, metric: str) -> str:
    if metric in {"Compression ratio"}:
        return f"{value:.2f}x"
    if metric in {"PSNR", "Keyframe PSNR", "SNR", "Signal-to-Noise Ratio"}:
        return f"{value:.2f} dB"
    if metric in {"SSIM", "Keyframe SSIM", "Waveform correlation", "Waveform Correlation", "Spectrogram similarity"}:
        return f"{value:.4f}"
    if metric in {"MAE", "Mean absolute error", "Duration difference"}:
        return f"{value:.3f}"
    if metric in {"Text accuracy"}:
        return f"{value:.2f}%"
    return f"{value:.2f}"


def _chart_scale_domain(metric: str) -> List[float] | None:
    if metric in {"SSIM", "Keyframe SSIM", "Waveform correlation", "Waveform Correlation", "Spectrogram similarity"}:
        return [0, 1]
    if metric in {"Text accuracy"}:
        return [0, 100]
    return None


def _chart_lollipop(
    rows: List[Dict[str, Any]],
    label_col: str,
    value_col: str,
    *,
    height: int | None = None,
) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if label_col not in df or value_col not in df:
        return
    df = df.copy()
    df["Value"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["Value"])
    if df.empty:
        return
    df["Order"] = range(len(df))
    df["Zero"] = 0.0
    df["Label"] = df["Value"].apply(lambda v: _format_chart_label(float(v), value_col))
    chart_height = height or max(120, min(260, 34 * len(df) + 24))
    domain = _chart_scale_domain(value_col)

    x_scale: Dict[str, Any] = {"zero": True}
    if domain:
        x_scale["domain"] = domain
    spec = {
        "height": chart_height,
        "encoding": {
            "y": {
                "field": label_col,
                "type": "nominal",
                "sort": {"field": "Order", "order": "ascending"},
                "axis": {"labelLimit": 180, "labelFontSize": 11, "title": None},
            },
            "x": {
                "field": "Value",
                "type": "quantitative",
                "title": value_col,
                "scale": x_scale,
                "axis": {"grid": True, "labelFontSize": 10, "titleFontSize": 11},
            },
        },
        "layer": [
            {
                "mark": {"type": "rule", "strokeWidth": 2, "color": "#94a3b8"},
                "encoding": {"x": {"field": "Zero", "type": "quantitative"}, "x2": {"field": "Value"}},
            },
            {"mark": {"type": "point", "filled": True, "size": 72, "color": "#2563eb", "opacity": 0.92}},
            {
                "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 8, "fontSize": 11, "color": "#0f172a"},
                "encoding": {"text": {"field": "Label", "type": "nominal"}},
            },
        ],
    }
    st.vega_lite_chart(df, spec, use_container_width=True)


def _chart_bar(rows: List[Dict[str, Any]], label_col: str, value_col: str) -> None:
    _chart_lollipop(rows, label_col, value_col)


def _format_method_name(method: str) -> str:
    text = str(method or "No compression").replace("_", " ").replace("-", " ")
    tokens = []
    for token in text.split():
        lower = token.lower()
        if lower in {"webp", "png", "jpeg", "jpg", "avif", "ogg", "opus", "mp4", "h264", "h265", "vp9", "gzip", "zlib", "zip", "bz2", "xz"}:
            tokens.append(lower.upper())
        elif lower.startswith("q") and lower[1:].isdigit():
            tokens.append(f"Q{lower[1:]}")
        elif lower.startswith("c") and lower[1:].isdigit():
            tokens.append(f"C{lower[1:]}")
        elif lower.startswith("lvl") and lower[3:].isdigit():
            tokens.append(f"LVL{lower[3:]}")
        elif lower.endswith("k") and lower[:-1].isdigit():
            tokens.append(lower.upper())
        else:
            tokens.append(token.capitalize())
    return " ".join(tokens)


def _quality_columns_for_domain(domain: str) -> List[str]:
    if domain == "image":
        return ["PSNR", "SSIM", "MAE"]
    if domain == "audio":
        return ["SNR", "Waveform correlation", "Spectrogram similarity"]
    if domain == "video":
        return ["Keyframe PSNR", "Keyframe SSIM", "Duration difference", "Resolution"]
    if domain == "text":
        return ["Text accuracy", "Exact match", "Length delta"]
    return ["Exact match"]


def _quality_rows_to_candidate_columns(rows: List[Dict[str, Any]], domain: str | None = None) -> Dict[str, str]:
    aliases = {
        "PSNR": "PSNR",
        "SSIM": "SSIM",
        "Mean absolute error": "MAE",
        "Signal-to-Noise Ratio": "SNR",
        "Waveform Correlation": "Waveform correlation",
        "Spectrogram similarity": "Spectrogram similarity",
        "Keyframe PSNR": "Keyframe PSNR",
        "Keyframe SSIM": "Keyframe SSIM",
        "Text accuracy": "Text accuracy",
        "Exact text match": "Exact match",
        "Length delta": "Length delta",
        "Duration difference": "Duration difference",
        "Resolution": "Resolution",
    }
    wanted = _quality_columns_for_domain(domain or "") if domain else list(dict.fromkeys(aliases.values()))
    out = {name: "—" for name in wanted}
    for row in rows:
        metric = str(row.get("Metric", ""))
        if metric in aliases and aliases[metric] in out:
            out[aliases[metric]] = str(row.get("Value", "—"))
    return out


def _candidate_quality_cache_key(input_path: str | None, input_bytes: bytes, cand: CompressionCandidate) -> str:
    raw = bytes(input_bytes or b"")
    digest = hashlib.sha256(raw[:8192] + cand.data[:8192]).hexdigest()
    return "|".join([
        str(input_path or ""),
        str(len(raw)),
        str(cand.method),
        str(cand.size_bytes),
        str(cand.ext),
        digest,
    ])


def _candidate_quality_rows(
    input_path: str | None,
    input_bytes: bytes,
    cand: CompressionCandidate,
) -> List[Dict[str, Any]]:
    cache = st.session_state.setdefault("candidate_quality_cache", {})
    key = _candidate_quality_cache_key(input_path, input_bytes, cand)
    if key not in cache:
        cache[key] = _compression_quality_rows(input_path, input_bytes, _candidate_file(cand), cand.data)
    return cache.get(key, [])


def _candidate_quality_table_rows(
    input_path: str | None,
    input_bytes: bytes,
    candidates: List[CompressionCandidate],
) -> List[Dict[str, Any]]:
    domain = get_domain(input_path or "", bytes(input_bytes or b""))
    rows: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: c.rank):
        q_rows = _candidate_quality_rows(input_path, input_bytes, cand)
        row = {
            "Rank": cand.rank,
            "Method": _format_method_name(cand.method),
            "Output": cand.ext or ".bin",
            "Compressed data": fmt_bytes(cand.size_bytes),
            "Compression ratio": f"{cand.compression_ratio:.2f}x",
            "Size reduction": f"{cand.saving_pct:.2f}%",
        }
        row.update(_quality_rows_to_candidate_columns(q_rows, domain))
        row["Estimated DNA"] = f"{cand.estimated_dna_nt:,} nt"
        rows.append(row)
    return rows


def _parse_chart_value(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text == "—" or text == "Unavailable":
        return None
    if text.startswith("∞"):
        return 100.0
    if text == "Yes":
        return 100.0
    if text == "No":
        return 0.0
    cleaned = (
        text.replace("dB", "")
        .replace("%", "")
        .replace("s", "")
        .replace("x", "")
        .replace("×", "")
        .strip()
    )
    if "->" in cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _render_quality_metric_charts(rows: List[Dict[str, Any]], metrics: List[str]) -> None:
    if not rows:
        return
    chart_metrics = [m for m in metrics if m != "Resolution"]
    for i in range(0, len(chart_metrics), 2):
        cols = st.columns(2)
        for col, metric in zip(cols, chart_metrics[i:i + 2]):
            chart_rows = []
            for row in rows:
                value = _parse_chart_value(row.get(metric))
                if value is not None:
                    chart_rows.append({"Method": row.get("Method", ""), metric: value})
            if chart_rows:
                with col:
                    st.markdown(f"##### {metric}")
                    _chart_lollipop(chart_rows, "Method", metric, height=max(120, min(210, 32 * len(chart_rows) + 20)))


def _quality_score_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []
    for row in rows:
        metric = str(row.get("Metric", ""))
        value = str(row.get("Value", ""))
        try:
            if metric in {"SSIM", "Keyframe SSIM", "Waveform Correlation", "Spectrogram similarity"}:
                scores.append({"Metric": metric, "Score": max(0.0, min(100.0, float(value) * 100.0))})
            elif metric in {"Text accuracy"}:
                scores.append({"Metric": metric, "Score": max(0.0, min(100.0, float(value.rstrip("%"))))})
            elif metric == "Exact text match":
                scores.append({"Metric": metric, "Score": 100.0 if value == "Yes" else 0.0})
        except Exception:
            pass
    return scores


def _quality_numeric_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    numeric_metrics = {
        "PSNR",
        "SSIM",
        "Mean absolute error",
        "Signal-to-Noise Ratio",
        "Waveform Correlation",
        "Spectrogram similarity",
        "Keyframe PSNR",
        "Keyframe SSIM",
        "Duration difference",
    }
    for row in rows:
        metric = str(row.get("Metric", ""))
        value = str(row.get("Value", ""))
        if metric not in numeric_metrics or value == "Unavailable":
            continue
        if value.startswith("∞"):
            parsed = 100.0
        else:
            cleaned = value.replace("dB", "").replace("s", "").strip()
            try:
                parsed = float(cleaned)
            except Exception:
                continue
        out.append({"Metric": metric, "Value": parsed})
    return out


def _candidate_list_label(cand: CompressionCandidate) -> str:
    return (
        f"{cand.rank}. {_format_method_name(cand.method)} | {cand.ext or '.bin'} | "
        f"{fmt_bytes(cand.size_bytes)} | {cand.compression_ratio:.2f}x | "
        f"{cand.estimated_dna_nt:,} nt"
    )


def _set_selected_candidate(cand: CompressionCandidate) -> None:
    st.session_state.update({
        "selected_candidate": cand,
        "stored_bytes": cand.data,
        "stored_file_path": _selected_candidate_path(cand),
        "storage_method": cand.method,
        "storage_kind": cand.kind,
        "storage_meta": {"kind": "compressed_file", "method": cand.method, "file_kind": cand.kind, "ext": cand.ext},
    })
    for key in ["dna", "bits", "codec_meta", "strand_rows", "decoded_data", "restored_info", "decode_error"]:
        st.session_state.pop(key, None)


def _render_candidate_list(candidates: List[CompressionCandidate], selected: CompressionCandidate | None) -> None:
    if not candidates:
        return

    st.markdown("##### 🧾 Compression candidates")
    current_idx = 0
    if selected is not None:
        for i, cand in enumerate(candidates):
            if cand.method == selected.method and cand.size_bytes == selected.size_bytes and cand.ext == selected.ext:
                current_idx = i
                break
    selected_idx = st.selectbox(
        "Selected compressed output",
        list(range(len(candidates))),
        index=current_idx,
        format_func=lambda i: _candidate_list_label(candidates[int(i)]),
        key="compression_choice",
    )
    cand = candidates[int(selected_idx)]
    if (
        selected is None
        or cand.method != selected.method
        or cand.size_bytes != selected.size_bytes
        or cand.ext != selected.ext
    ):
        _set_selected_candidate(cand)
        st.rerun()

    input_path = st.session_state.get("input_path")
    input_bytes = st.session_state.get("input_bytes", b"") or b""
    table_rows = _candidate_quality_table_rows(input_path, input_bytes, candidates)
    for row, c in zip(table_rows, sorted(candidates, key=lambda item: item.rank)):
        row["Selected"] = "Yes" if c.method == cand.method and c.size_bytes == cand.size_bytes and c.ext == cand.ext else ""
    ordered = []
    for row in table_rows:
        selected_flag = row.pop("Selected", "")
        ordered.append({"Selected": selected_flag, **row})
    _render_stats_table(ordered)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, float(seconds))
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins}:{secs:05.2f}"


def _previewable_payload_path(path: str | None, data: bytes | None, title: str) -> str | None:
    if not path:
        return None
    kind = magic_dict(bytes(data or b"")).get("kind", "unknown")
    if kind in {"zip", "gzip", "xz", "bz2", "zlib"}:
        inner = _write_preview_inner(path, kind, title)
        return inner or path
    return path


def _file_info_rows(path: str | None, data: bytes | None) -> List[Dict[str, Any]]:
    raw = bytes(data or b"")
    ext = os.path.splitext(path or "")[1].lower() or magic_dict(raw).get("ext", ".bin")
    domain = get_domain(path or "", raw) if raw else "unknown"
    m = magic_dict(raw)
    rows: List[Dict[str, Any]] = [
        {"Metric": "File name", "Value": os.path.basename(path or "—")},
        {"Metric": "Extension", "Value": ext or "—"},
        {"Metric": "Type", "Value": domain},
        {"Metric": "Size", "Value": fmt_bytes(len(raw))},
    ]
    if m.get("kind") and m.get("kind") != "unknown":
        rows.append({"Metric": "Container", "Value": m.get("kind")})

    if domain == "image" and Image is not None and raw:
        try:
            img = Image.open(io.BytesIO(raw))
            rows.append({"Metric": "Image size", "Value": f"{img.width} x {img.height} px"})
        except Exception:
            pass

    if domain in {"audio", "video"} and path:
        info = _run_ffprobe(path)
        duration = _duration_seconds(info)
        if duration is not None:
            rows.append({"Metric": "Duration", "Value": _format_duration(duration)})
        if domain == "video":
            stream = _first_stream(info, "video")
            if stream:
                rows.extend([
                    {"Metric": "Resolution", "Value": f"{stream.get('width', '?')} x {stream.get('height', '?')} px"},
                    {"Metric": "FPS", "Value": _fps_value(stream)},
                ])
    return rows


def _render_info_table(rows: List[Dict[str, Any]]) -> None:
    _render_property_table(rows)


def _render_metric_rows(rows: List[Dict[str, Any]], columns: int = 4) -> None:
    if not rows:
        return
    width = max(1, min(int(columns), 4))
    for start in range(0, len(rows), width):
        chunk = rows[start:start + width]
        cols = st.columns(len(chunk))
        for col, row in zip(cols, chunk):
            metric = str(row.get("Metric", ""))
            value = str(row.get("Value", ""))
            delta = row.get("Delta")
            col.metric(metric, value, delta=delta if delta else None)


def _render_workflow_overview() -> None:
    steps = [
        ("1", "Upload", "Input preview"),
        ("2", "Compress", "Stored bytes"),
        ("3", "DNA", "Mapping"),
        ("4", "Strands", "Design"),
        ("5", "Decode", "Restored file"),
        ("6", "Validate", "Compare"),
    ]
    html = ['<div class="workflow-strip">']
    for no, title, desc in steps:
        html.append(
            f'<div class="workflow-item"><div class="workflow-no">{no}</div>'
            f'<div class="workflow-title">{title}</div><div class="workflow-desc">{desc}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _clear_downstream_from_storage() -> None:
    for key in [
        "compression_candidates", "selected_candidate", "stored_bytes", "stored_file_path",
        "candidate_quality_cache",
        "storage_method", "storage_kind", "storage_meta",
        "dna", "bits", "codec_meta", "strand_rows",
        "decoded_data", "decoded_raw_pixels", "decoded_bits", "decoded_meta", "decoded_magic", "decoded_valid",
        "decoded_note", "raw_restore_info", "restored_info", "decode_error",
    ]:
        st.session_state.pop(key, None)


def _validate_and_write(data: bytes, preferred: str = "restored") -> Dict[str, Any]:
    out_dir = WORK_ROOT / "decode_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_restored_file(data, str(out_dir), preferred_name=preferred)


def _is_uploaded_image(path: str, data: bytes) -> bool:
    """Return True when the uploaded file can be handled as an image."""
    if Image is None or not data:
        return False
    try:
        domain = get_domain(path, data)
        if domain == "image":
            return True
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def _image_pixels_to_bytes(data: bytes, representation: str, threshold: int = 128) -> Tuple[bytes, Dict[str, Any], bytes]:
    """
    Convert an uploaded image to raw pixel bytes for no-compression storage.

    Returns: (pixel_bytes, metadata, preview_png_bytes).
    The bytes are not an image container; width/height/mode metadata are required
    to reconstruct them later.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image pixel conversion.")
    img = Image.open(io.BytesIO(data))
    if representation == "RGB pixels":
        out_img = img.convert("RGB")
        channels = 3
        raw_mode = "RGB"
        rep_label = "RGB pixels"
    elif representation == "Grayscale pixels":
        out_img = img.convert("L")
        channels = 1
        raw_mode = "L"
        rep_label = "Grayscale pixels"
    elif representation == "Binary image pixels":
        gray = img.convert("L")
        out_img = gray.point(lambda p: 255 if p >= int(threshold) else 0).convert("L")
        channels = 1
        raw_mode = "L"
        rep_label = "Binary image pixels"
    else:
        raise ValueError(f"Unknown image representation: {representation}")

    raw = out_img.tobytes()
    png = io.BytesIO()
    out_img.save(png, format="PNG")
    meta = {
        "kind": "raw_image_pixels",
        "representation": rep_label,
        "raw_mode": raw_mode,
        "width": int(out_img.width),
        "height": int(out_img.height),
        "channels": int(channels),
        "expected_bytes": int(len(raw)),
        "threshold": int(threshold),
        "output_ext": ".png",
    }
    return raw, meta, png.getvalue()


def _raw_image_bytes_to_png(data: bytes, meta: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """Build a PNG preview/output from decoded raw image pixel bytes."""
    if Image is None:
        raise RuntimeError("Pillow is required to restore raw image pixels.")
    width = int(meta.get("width", 0))
    height = int(meta.get("height", 0))
    mode = str(meta.get("raw_mode", "L"))
    expected = int(meta.get("expected_bytes", width * height * (3 if mode == "RGB" else 1)))
    raw = bytes(data or b"")
    note = "Exact raw-pixel length."
    if len(raw) < expected:
        raw = raw + bytes(expected - len(raw))
        note = f"Decoded bytes were shorter than expected; padded {expected - len(data or b'')} bytes."
    elif len(raw) > expected:
        raw = raw[:expected]
        note = f"Decoded bytes were longer than expected; truncated {len(data or b'') - expected} bytes."
    img = Image.frombytes(mode, (width, height), raw)
    png = io.BytesIO()
    img.save(png, format="PNG")
    return png.getvalue(), {"note": note, "width": width, "height": height, "mode": mode, "expected_bytes": expected}


def _byte_accuracy_bytes(a: bytes, b: bytes) -> float:
    """Position-wise byte accuracy with length difference counted as errors."""
    a = bytes(a or b"")
    b = bytes(b or b"")
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / denom


def _bit_error_rate_bytes(a: bytes, b: bytes) -> float:
    a = bytes(a or b"")
    b = bytes(b or b"")
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0

    min_len = min(len(a), len(b))
    error_bits = 0
    for x, y in zip(a[:min_len], b[:min_len]):
        error_bits += (x ^ y).bit_count()
    error_bits += abs(len(a) - len(b)) * 8
    return error_bits / max(1, max_len * 8)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _sci(value: float) -> str:
    return f"{float(value):.2e}"


def _validation_row(stage: str, metric: str, value: Any, meaning: str | None = None, delta: str | None = None) -> Dict[str, Any]:
    row = {
        "Stage": stage,
        "Metric": metric,
        "Value": value,
    }
    if delta:
        row["Delta"] = delta
    return row


def _render_validation_metric_cards(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("Stage", "Validation")), []).append(row)

    for stage, stage_rows in grouped.items():
        st.markdown(f"#### {stage}")
        _render_metric_rows(stage_rows, columns=4)


def _file_cache_signature(path: str | None) -> Tuple[int, int]:
    if not path or not os.path.exists(path):
        return (0, 0)
    try:
        stat = os.stat(path)
        return (int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return (0, 0)


@st.cache_data(show_spinner=False)
def _run_ffprobe_cached(path: str, mtime_ns: int, size: int) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
                "-of",
                "json",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if p.returncode != 0 or not p.stdout.strip():
            return {}
        return json.loads(p.stdout)
    except Exception:
        return {}


def _run_ffprobe(path: str | None) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    mtime_ns, size = _file_cache_signature(path)
    return _run_ffprobe_cached(path, mtime_ns, size)


def _duration_seconds(info: Dict[str, Any]) -> float | None:
    try:
        value = info.get("format", {}).get("duration")
        return float(value) if value is not None else None
    except Exception:
        return None


def _first_stream(info: Dict[str, Any], codec_type: str) -> Dict[str, Any]:
    for stream in info.get("streams", []) or []:
        if stream.get("codec_type") == codec_type:
            return stream
    return {}


def _fps_value(stream: Dict[str, Any]) -> str:
    raw = str(stream.get("avg_frame_rate") or "")
    if "/" not in raw:
        return raw or "unknown"
    try:
        num, den = raw.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return "unknown"
        return f"{float(num) / den_f:.2f}"
    except Exception:
        return "unknown"


@st.cache_data(show_spinner=False)
def _decode_audio_mono_cached(path: str, mtime_ns: int, size: int, sample_rate: int, seconds: int):
    try:
        p = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                path,
                "-t",
                str(int(seconds)),
                "-ac",
                "1",
                "-ar",
                str(int(sample_rate)),
                "-f",
                "f32le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        if p.returncode != 0 or not p.stdout:
            return None
        audio = np.frombuffer(p.stdout, dtype=np.float32)
        return audio if audio.size else None
    except Exception:
        return None


def _decode_audio_mono(path: str | None, *, sample_rate: int = 16000, seconds: int = 30):
    if np is None or not path or not os.path.exists(path):
        return None
    mtime_ns, size = _file_cache_signature(path)
    return _decode_audio_mono_cached(path, mtime_ns, size, int(sample_rate), int(seconds))


def _spectrogram_similarity(path_a: str | None, path_b: str | None) -> float | None:
    if np is None:
        return None
    a = _decode_audio_mono(path_a)
    b = _decode_audio_mono(path_b)
    if a is None or b is None:
        return None
    n = min(a.size, b.size)
    if n < 1024:
        return None
    a = a[:n]
    b = b[:n]
    n_fft = 512
    hop = 256
    window = np.hanning(n_fft).astype(np.float32)

    def spec(x):
        frames = []
        for start in range(0, max(1, len(x) - n_fft + 1), hop):
            frame = x[start:start + n_fft]
            if frame.size < n_fft:
                break
            frames.append(np.log1p(np.abs(np.fft.rfft(frame * window))))
        if not frames:
            return None
        return np.asarray(frames, dtype=np.float32)

    sa = spec(a)
    sb = spec(b)
    if sa is None or sb is None:
        return None
    m = min(sa.shape[0], sb.shape[0])
    va = sa[:m].reshape(-1)
    vb = sb[:m].reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return None
    return max(0.0, min(1.0, float(np.dot(va, vb) / denom)))


def _audio_waveform_metrics(path_a: str | None, path_b: str | None) -> Dict[str, float]:
    if np is None:
        return {}
    a = _decode_audio_mono(path_a)
    b = _decode_audio_mono(path_b)
    if a is None or b is None:
        return {}
    n = min(a.size, b.size)
    if n < 1024:
        return {}
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    noise = a - b
    signal_power = float(np.mean(a * a))
    noise_power = float(np.mean(noise * noise))
    if noise_power <= 1e-18:
        snr = float("inf")
    elif signal_power <= 1e-18:
        snr = 0.0
    else:
        snr = float(10.0 * np.log10(signal_power / noise_power))

    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        corr = 1.0 if noise_power <= 1e-18 else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {"snr": snr, "correlation": max(-1.0, min(1.0, corr))}


@st.cache_data(show_spinner=False)
def _extract_video_frame_cached(path: str, mtime_ns: int, size: int, at_millis: int):
    try:
        at_seconds = max(0.0, float(at_millis) / 1000.0)
        p = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{max(0.0, float(at_seconds)):.3f}",
                "-i",
                path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        if p.returncode != 0 or not p.stdout:
            return None
        return Image.open(io.BytesIO(p.stdout)).convert("RGB")
    except Exception:
        return None


def _extract_video_frame(path: str | None, at_seconds: float = 1.0):
    if Image is None or not path or not os.path.exists(path):
        return None
    mtime_ns, size = _file_cache_signature(path)
    at_millis = int(max(0.0, float(at_seconds)) * 1000)
    return _extract_video_frame_cached(path, mtime_ns, size, at_millis)


def _image_array_metrics(img_a, img_b) -> Dict[str, float]:
    if np is None or img_a is None or img_b is None:
        return {}
    try:
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size)
        arr_a = np.asarray(img_a).astype("float32")
        arr_b = np.asarray(img_b).astype("float32")
        mse = float(np.mean((arr_a - arr_b) ** 2))
        psnr = 99.0 if mse <= 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        vals = []
        x = arr_a.reshape(-1, 3)
        y = arr_b.reshape(-1, 3)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        for ch in range(3):
            xx = x[:, ch]
            yy = y[:, ch]
            mux, muy = float(xx.mean()), float(yy.mean())
            vx, vy = float(xx.var()), float(yy.var())
            cov = float(((xx - mux) * (yy - muy)).mean())
            vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
        return {"psnr": psnr, "ssim": float(np.mean(vals))}
    except Exception:
        return {}


def _keyframe_metrics(path_a: str | None, path_b: str | None, duration: float | None) -> Dict[str, float]:
    at = 1.0
    if duration is not None and duration > 2:
        at = min(duration / 2.0, 10.0)
    frame_a = _extract_video_frame(path_a, at)
    frame_b = _extract_video_frame(path_b, at)
    return _image_array_metrics(frame_a, frame_b)


def _media_quality_rows(stage: str, domain: str, original_path: str | None, restored_path: str | None) -> List[Dict[str, Any]]:
    original = _run_ffprobe(original_path)
    restored = _run_ffprobe(restored_path)
    if not original or not restored:
        return [_validation_row(stage, "Media comparison", "Unavailable")]

    rows: List[Dict[str, Any]] = []
    orig_duration = _duration_seconds(original)
    new_duration = _duration_seconds(restored)
    if orig_duration is not None and new_duration is not None:
        diff = abs(orig_duration - new_duration)
        rows.append(_validation_row(
            stage,
            "Duration difference",
            f"{diff:.3f} s",
        ))

    if domain == "audio":
        spec_sim = _spectrogram_similarity(original_path, restored_path)
        wave_metrics = _audio_waveform_metrics(original_path, restored_path)
        snr = wave_metrics.get("snr")
        corr = wave_metrics.get("correlation")
        if snr is not None:
            if snr == float("inf"):
                rows.append(_validation_row(stage, "Signal-to-Noise Ratio", "∞ dB", delta="Perfect"))
            else:
                rows.append(_validation_row(stage, "Signal-to-Noise Ratio", f"{snr:.2f} dB"))
        if corr is not None:
            rows.append(_validation_row(stage, "Waveform Correlation", f"{corr:.4f}", delta="Perfect" if corr >= 0.9999 else None))
        rows.append(_validation_row(stage, "Spectrogram similarity", f"{spec_sim:.4f}" if spec_sim is not None else "Unavailable"))
    elif domain == "video":
        ov = _first_stream(original, "video")
        rv = _first_stream(restored, "video")
        orig_res = f"{ov.get('width', '?')}x{ov.get('height', '?')}"
        new_res = f"{rv.get('width', '?')}x{rv.get('height', '?')}"
        kmetrics = _keyframe_metrics(original_path, restored_path, orig_duration)
        rows.extend([
            _validation_row(stage, "Resolution", f"{orig_res} -> {new_res}"),
            _validation_row(stage, "Keyframe PSNR", f"{kmetrics['psnr']:.2f} dB" if "psnr" in kmetrics else "Unavailable"),
            _validation_row(stage, "Keyframe SSIM", f"{kmetrics['ssim']:.4f}" if "ssim" in kmetrics else "Unavailable"),
        ])
    return rows


def _compression_quality_rows(
    input_path: str | None,
    input_bytes: bytes,
    stored_path: str | None,
    stored_bytes: bytes,
) -> List[Dict[str, Any]]:
    original = bytes(input_bytes or b"")
    stored = bytes(stored_bytes or b"")
    if not input_path or not original or not stored_path or not stored:
        return []

    output_path = _previewable_payload_path(stored_path, stored, "compression_quality")
    if not output_path:
        return []

    domain = get_domain(input_path, original)
    rows: List[Dict[str, Any]] = []
    if domain == "image" and Image is not None:
        metrics = image_metrics(input_path, output_path)
        if metrics.get("Validation"):
            rows.extend([
                _validation_row("Compression quality", "PSNR", f"{float(metrics.get('psnr', 0)):.2f} dB"),
                _validation_row("Compression quality", "SSIM", f"{float(metrics.get('ssim', 0)):.4f}"),
                _validation_row("Compression quality", "Mean absolute error", f"{float(metrics.get('mae', 0)):.2f}"),
            ])
    elif domain == "text":
        sim = text_similarity(input_path, output_path)
        if sim.get("Validation"):
            rows.extend([
                _validation_row("Compression quality", "Text accuracy", _pct(float(sim.get("char_position_accuracy", 0)))),
                _validation_row("Compression quality", "Exact text match", "Yes" if sim.get("exact") else "No"),
                _validation_row("Compression quality", "Length delta", f"{int(sim.get('len_b', 0)) - int(sim.get('len_a', 0)):+,} chars"),
            ])
    elif domain in {"audio", "video"}:
        rows.extend(_media_quality_rows("Compression quality", domain, input_path, output_path))

    if original and stored:
        rows.extend([
            _validation_row("Compression efficiency", "Original size", fmt_bytes(len(original))),
            _validation_row("Compression efficiency", "Compressed data", fmt_bytes(len(stored))),
            _validation_row("Compression efficiency", "Compression ratio", f"{len(original) / max(1, len(stored)):.2f}x"),
            _validation_row("Compression efficiency", "Size reduction", _pct(1.0 - (len(stored) / max(1, len(original))))),
        ])
    return rows


def _validation_rows(
    *,
    input_path: str | None,
    input_bytes: bytes,
    stored_file_path: str | None,
    stored_bytes: bytes,
    restored_preview_path: str | None,
    recovered_for_match: bytes,
    file_can_open: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stored = bytes(stored_bytes or b"")
    recovered = bytes(recovered_for_match or b"")
    original = bytes(input_bytes or b"")

    hash_match = bool(stored) and hashlib.sha256(stored).hexdigest() == hashlib.sha256(recovered).hexdigest()
    length_delta = len(recovered) - len(stored)

    rows.extend([
        _validation_row("DNA decode integrity", "Payload accuracy", _pct(_byte_accuracy_bytes(stored, recovered))),
        _validation_row("DNA decode integrity", "Bit error rate", _sci(_bit_error_rate_bytes(stored, recovered))),
        _validation_row("DNA decode integrity", "Length delta", f"{length_delta:+,} bytes"),
        _validation_row("DNA decode integrity", "Checksum", "Pass" if hash_match else "Fail"),
    ])

    return rows


def _compression_analysis_rows(
    input_path: str | None,
    input_bytes: bytes,
    stored_path: str | None,
    stored_bytes: bytes,
) -> List[Dict[str, Any]]:
    original = bytes(input_bytes or b"")
    stored = bytes(stored_bytes or b"")
    if not original or not stored:
        return []
    domain = get_domain(input_path or "", original)
    quality = _quality_rows_to_candidate_columns(_compression_quality_rows(input_path, original, stored_path, stored), domain)
    row = {
        "Method": _format_method_name(st.session_state.get("storage_method", "No compression")),
        "Original data": fmt_bytes(len(original)),
        "Original extension": os.path.splitext(input_path or "")[1].lower() or magic_dict(original).get("ext", ".bin"),
        "Compressed data": fmt_bytes(len(stored)),
        "Compressed extension": os.path.splitext(stored_path or "")[1].lower() or magic_dict(stored).get("ext", ".bin"),
        "Compression ratio": f"{len(original) / max(1, len(stored)):.2f}x",
        "Size reduction": _pct(1.0 - (len(stored) / max(1, len(original)))),
    }
    for key, value in quality.items():
        if value != "—":
            row[key] = value
    return [row]


def _encode_decode_analysis_rows(
    stored_bytes: bytes,
    recovered_bytes: bytes,
    dna: str,
    strand_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stored = bytes(stored_bytes or b"")
    recovered = bytes(recovered_bytes or b"")
    byte_accuracy = _byte_accuracy_bytes(stored, recovered)
    hash_match = bool(stored) and hashlib.sha256(stored).hexdigest() == hashlib.sha256(recovered).hexdigest()
    lengths = [len(clean_dna(r.get("Full strand", ""))) for r in strand_rows]
    hp_values = []
    gc_values = []
    for row in strand_rows:
        try:
            hp_values.append(int(row.get("Longest homopolymer", 0)))
            gc_values.append(float(row.get("GC content", 0)))
        except Exception:
            pass

    recovery = "Decoded exactly, every byte matches (100%)" if byte_accuracy >= 1.0 and hash_match else "Decoded with differences"
    return [
        {"Metric": "DNA design", "Value": _display_mapping(st.session_state.get("encoding_mapping", "—"))},
        {"Metric": "Strand format", "Value": "FBR + SI + payload + filler + RBR" if strand_rows else "Not prepared"},
        {"Metric": "DNA length", "Value": f"{len(dna or ''):,} nt"},
        {"Metric": "DNA GC content", "Value": f"{gc_content(dna):.3f}" if dna else "—"},
        {"Metric": "DNA longest homopolymer", "Value": homopolymer_stats(dna).get("longest", 0) if dna else "—"},
        {"Metric": "Strand count", "Value": len(strand_rows)},
        {"Metric": "Total strand length", "Value": f"{sum(lengths):,} nt" if lengths else "—"},
        {"Metric": "Average strand GC", "Value": f"{(sum(gc_values) / len(gc_values)):.4f}" if gc_values else "—"},
        {"Metric": "Strand longest homopolymer", "Value": max(hp_values) if hp_values else "—"},
        {"Metric": "Payload accuracy", "Value": _pct(byte_accuracy)},
        {"Metric": "Checksum", "Value": "Pass" if hash_match else "Fail"},
        {"Metric": "Recovery result", "Value": recovery},
    ]


# -----------------------------------------------------------------------------
# DNA Strand Prep visualization
# -----------------------------------------------------------------------------

_REGION_COLORS = REGION_COLORS


def _row_regions(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return ordered strand regions for the prepared strand view."""
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("SI", clean_dna(row.get("Strand index", ""))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _region_html(name: str, seq: str, error_positions: set[int] | None = None, *, start_pos: int = 1) -> str:
    """Render one region with optional red marking at 1-indexed full-strand positions."""
    bg, fg = _REGION_COLORS.get(name, ("#f8fafc", "#0f172a"))
    error_positions = error_positions or set()
    chars = []
    for i, ch in enumerate(clean_dna(seq), start=start_pos):
        if i in error_positions:
            ebg, efg = _REGION_COLORS["Error"]
            chars.append(f'<span class="error-base">{ch}</span>')
        else:
            chars.append(ch)
    body = "".join(chars) if chars else "—"
    return (
        f'<span class="region-tag" style="background:{bg};color:{fg};">'
        f'<b>{name}</b>: {body}</span>'
    )


def _render_segmented_strand(row: Dict[str, Any], title: str, *, error_positions: set[int] | None = None) -> None:
    """Show FBR/SI/Payload/Filler/RBR as colored chunks."""
    parts = []
    cursor = 1
    for name, seq in _row_regions(row):
        parts.append(_region_html(name, seq, error_positions, start_pos=cursor))
        cursor += len(seq)
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Panel 1 — Upload
# -----------------------------------------------------------------------------


def render_panel_1_upload() -> None:
    _apply_panel_typography()
    with st.container(border=True):
        step_header(1, PANEL_TITLES["input"])
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 📁 Input")
            uploaded = st.file_uploader("", type=None, key="upload_input_file")
            if uploaded is not None:
                # Streamlit re-runs the script after every button click. The uploaded
                # file object is still present on those re-runs, so we must not treat
                # it as a new upload every time; otherwise downstream results are
                # cleared immediately after Run Encode / Run DNA Strand Prep.
                data_now = uploaded.getvalue()
                upload_sig = f"{uploaded.name}|{len(data_now)}|{hashlib.sha256(data_now).hexdigest()}"
                if st.session_state.get("upload_signature") != upload_sig:
                    path, data = save_upload(uploaded)
                    st.session_state.update({
                        "upload_signature": upload_sig,
                        "input_path": path,
                        "input_bytes": data,
                        "input_name": os.path.basename(path),
                    })
                    _clear_downstream_from_storage()
                elif not st.session_state.get("input_bytes"):
                    # Defensive fallback for restored sessions.
                    path, data = save_upload(uploaded)
                    st.session_state.update({
                        "input_path": path,
                        "input_bytes": data,
                        "input_name": os.path.basename(path),
                    })
            data = st.session_state.get("input_bytes")
            path = st.session_state.get("input_path")
            if data and path:
                st.markdown("##### 📄 File properties")
                _render_info_table(_file_info_rows(path, data))
                with st.expander(FIELDS["input_binary"], expanded=False):
                    st.text_area("Binary preview only", _bytes_to_bit_text(data), height=120)
                    _download_full_binary_button(BUTTONS["download_input_binary"], data, DOWNLOAD_FILES["input_binary"])
        with right:
            st.markdown("#### 🖼️ Preview")
            data = st.session_state.get("input_bytes")
            path = st.session_state.get("input_path")
            if not data or not path:
                st.info(MESSAGES["upload_to_start"])
                return
            preview_file(path, FIELDS["input_preview"])


# -----------------------------------------------------------------------------
# Panel 2 — Encode: compression
# -----------------------------------------------------------------------------


def render_panel_2_compression() -> None:
    with st.container(border=True):
        step_header(2, PANEL_TITLES["data_encoding"])
        data = st.session_state.get("input_bytes")
        path = st.session_state.get("input_path")
        if not data or not path:
            st.info(MESSAGES["upload_first"])
            return

        controls, selected_output = st.columns([0.9, 1.1], gap="large")
        with controls:
            st.markdown("#### 🗂️ Encoding parameters")
            storage_mode = st.radio(FIELDS["storage_method"], [DATA_SOURCES["no_compression"], DATA_SOURCES["compression"]], horizontal=True, key="storage_mode")

            if st.button(BUTTONS["run_data_encoding"], key="run_compression", type="primary", use_container_width=True):
                _clear_downstream_from_storage()
                if storage_mode == DATA_SOURCES["no_compression"]:
                    st.session_state.update({
                        "compression_candidates": [],
                        "selected_candidate": None,
                        "stored_bytes": data,
                        "stored_file_path": path,
                        "storage_method": "No compression",
                        "storage_kind": magic_dict(data).get("kind", "unknown"),
                        "storage_meta": {"kind": "file_bytes"},
                    })
                else:
                    cand, candidates = run_compression_benchmark(path, data)
                    st.session_state.pop("candidate_quality_cache", None)
                    st.session_state.update({
                        "compression_candidates": candidates,
                        "selected_candidate": cand,
                        "stored_bytes": cand.data,
                        "stored_file_path": _selected_candidate_path(cand),
                        "storage_method": cand.method,
                        "storage_kind": cand.kind,
                        "storage_meta": {"kind": "compressed_file", "method": cand.method, "file_kind": cand.kind, "ext": cand.ext},
                    })

            candidates: List[CompressionCandidate] = st.session_state.get("compression_candidates", [])

        with selected_output:
            st.markdown("#### 🗜️ Selected compressed output")
            stored = st.session_state.get("stored_bytes")
            if not stored:
                st.info(MESSAGES["choose_storage"])
                return

            kind = st.session_state.get("storage_kind", magic_dict(stored).get("kind", "unknown"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Compressed data", fmt_bytes(len(stored)))
            c2.metric(METRICS["stored_type"], kind)
            c3.metric("Estimated DNA", f"{len(stored) * 4:,} nt")

            d1, d2 = st.columns(2)
            with d1:
                download_bytes_button(BUTTONS["download_stored_data"], stored, file_name=f"stored_data{magic_dict(stored).get('ext', '.bin')}")
            with d2:
                _download_full_binary_button(BUTTONS["download_stored_binary"], stored, DOWNLOAD_FILES["stored_binary"])

        candidates = st.session_state.get("compression_candidates", [])
        stored = st.session_state.get("stored_bytes")
        stored_path = st.session_state.get("stored_file_path")
        quality_rows = _compression_quality_rows(path, data, stored_path, stored or b"") if stored and stored_path else []
        if quality_rows:
            st.session_state["selected_compression_quality"] = _quality_rows_to_candidate_columns(quality_rows)
        else:
            st.session_state["selected_compression_quality"] = {}
        if storage_mode == DATA_SOURCES["compression"] and candidates:
            current = st.session_state.get("selected_candidate")
            _render_candidate_list(candidates, current)

        st.markdown("#### 🖼️ Before / after compression")
        before, after = st.columns(2, gap="large")
        with before:
            st.markdown("##### Before compression")
            preview_file(path, "Original preview")
            _render_property_table(_pipeline_file_metric_rows(path, data))
        with after:
            st.markdown("##### After compression")
            stored = st.session_state.get("stored_bytes")
            stored_path = st.session_state.get("stored_file_path")
            if stored and stored_path:
                preview_path = _previewable_payload_path(stored_path, stored, "after_compression_preview") or stored_path
                preview_file(preview_path, "Compressed output preview", show_caption=False)
                _render_property_table(_pipeline_file_metric_rows(stored_path, stored, compressed=True))
            else:
                st.info(MESSAGES["choose_storage"])

        if candidates:
            st.markdown("#### 📊 Compression analysis")
            ranked_candidates = sorted(candidates, key=lambda c: c.rank)
            _chart_bar(
                [{"Method": f"{c.rank}. {_format_method_name(c.method)}", "Compression ratio": float(c.compression_ratio)} for c in ranked_candidates],
                "Method",
                "Compression ratio",
            )
        if candidates:
            st.markdown("#### 📊 Compression quality")
            domain = get_domain(path, data)
            quality_table_rows = _candidate_quality_table_rows(path, data, candidates)
            metric_cols = _quality_columns_for_domain(domain)
            visible_cols = ["Rank", "Method"] + [m for m in metric_cols if any(str(row.get(m, "—")) != "—" for row in quality_table_rows)]
            visible_rows = [{k: row.get(k, "—") for k in visible_cols} for row in quality_table_rows]
            _render_stats_table(visible_rows)
            _render_quality_metric_charts(visible_rows, [m for m in visible_cols if m not in {"Rank", "Method"}])
        elif quality_rows:
            st.markdown("#### 📊 Compression quality")
            _render_property_table(quality_rows)


# -----------------------------------------------------------------------------
# Panel 3 — Encode: DNA mapping
# -----------------------------------------------------------------------------


def render_panel_3_encoding() -> None:
    with st.container(border=True):
        step_header(3, "Encoding")
        payload = st.session_state.get("stored_bytes")
        if not payload:
            st.info(MESSAGES["run_data_encoding_first"])
            return

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 🧬 Design")
            previous = st.session_state.get("encoding_mapping", "Simple Mapping")
            if previous not in MAPPING_OPTIONS:
                previous = "Simple Mapping"
            mapping = st.selectbox(
                "DNA design",
                MAPPING_OPTIONS,
                index=MAPPING_OPTIONS.index(previous),
                format_func=_display_mapping,
                key="encoding_mapping_select",
            )

            if st.button(BUTTONS["run_dna_encoding"], key="run_encoding", type="primary", use_container_width=True):
                dna, bits, meta = encode_bytes_to_dna(payload, mapping)
                st.session_state.update({
                    "encoding_mapping": mapping,
                    "dna": dna,
                    "bits": bits,
                    "codec_meta": meta,
                    "strand_rows": [],
                    "decoded_data": None,
                    "restored_info": None,
                    "decode_error": "",
                })

            st.markdown("##### 📄 Encoded data properties")
            _render_property_table([
                {"Metric": "Encoded data", "Value": fmt_bytes(len(payload))},
                {"Metric": "Estimated bits", "Value": f"{len(payload) * 8:,} bits"},
            ])
            _download_full_binary_button(BUTTONS["download_encoded_binary"], payload, DOWNLOAD_FILES["encoded_binary"])

        with right:
            st.markdown("#### 🧬 DNA output")
            dna = st.session_state.get("dna", "")
            if not dna:
                st.info(MESSAGES["run_data_encoding_first"])
                return

            _render_property_table([
                {"Metric": "DNA design", "Value": _display_mapping(st.session_state.get("encoding_mapping", mapping))},
                {"Metric": METRICS["dna_length"], "Value": f"{len(dna):,} nt"},
                {"Metric": "GC content", "Value": f"{gc_content(dna):.3f}"},
                {"Metric": "Longest homopolymer", "Value": homopolymer_stats(dna).get("longest", 0)},
            ])
            st.text_area("DNA payload preview", _preview_seq(dna, 600), height=DNA_PREVIEW_HEIGHT)
            _download_text_button(BUTTONS["download_encoded_dna"], dna, DOWNLOAD_FILES["encoded_dna"])


# -----------------------------------------------------------------------------
# Panel 4 — DNA Strand Prep
# -----------------------------------------------------------------------------



def render_panel_4_experiment() -> None:
    with st.container(border=True):
        step_header(4, PANEL_TITLES["strand_preparation"])
        dna = st.session_state.get("dna", "")
        if not dna:
            st.info(MESSAGES["run_dna_encoding_first"])
            return

        mapping = st.session_state.get("encoding_mapping", "")
        st.markdown(f"#### 🧵 {PANEL_TITLES['strand_preparation']}")
        with st.expander(FIELDS["strand_design"], expanded=not bool(st.session_state.get("strand_rows"))):
            target_len = st.number_input(FIELDS["total_strand_length"], min_value=80, max_value=250, value=125, step=1, key="std_total_len")
            index_len = st.number_input(FIELDS["si_length"], min_value=0, max_value=24, value=8, step=1, key="std_index_len")
            fbr = st.text_input(FIELDS["fbr"], value="ACACGACGCTCTTCCGATCT", key="std_fbr")
            rbr = st.text_input(FIELDS["rbr"], value="AGATCGGAAGAGCACACGTCT", key="std_rbr")
            if st.button(BUTTONS["run_strand_preparation"], key="build_standard_strands"):
                cfg = choose_auto_strand_design(
                    len(dna), len(clean_dna(fbr)), len(clean_dna(rbr)), int(index_len),
                    min_total_len=int(target_len), max_total_len=int(target_len),
                )
                rows = prepare_dna_strands(
                    dna,
                    fbr=clean_dna(fbr),
                    rbr=clean_dna(rbr),
                    index_len=int(index_len),
                    target_total_len=int(cfg.get("target_total_len", cfg.get("total_len", target_len))),
                    add_filler=True,
                )
                for r in rows:
                    r["Type"] = FIELDS["prepared_strand"]
                st.session_state.update({
                    "strand_rows": rows,
                    "decoded_data": None,
                    "restored_info": None,
                })

        rows: List[Dict[str, Any]] = st.session_state.get("strand_rows", [])
        if not rows:
            st.info(MESSAGES["run_strand_preparation"])
            return

        lengths = [len(clean_dna(r.get("Full strand", ""))) for r in rows]
        total_strand_len = sum(lengths)
        gc_values = []
        hp_values = []
        for r in rows:
            try:
                gc_values.append(float(r.get("GC content", 0)))
                hp_values.append(int(r.get("Longest homopolymer", 0)))
            except Exception:
                pass
        st.markdown("##### 📄 Strand statistics")
        _render_property_table([
            {"Metric": METRICS["prepared_strands"], "Value": len(rows)},
            {"Metric": METRICS["total_strand_length"], "Value": f"{total_strand_len:,} nt"},
            {"Metric": "Average GC", "Value": f"{(sum(gc_values) / len(gc_values)):.4f}" if gc_values else "—"},
            {"Metric": "Longest homopolymer", "Value": max(hp_values) if hp_values else "—"},
            {"Metric": "DNA design", "Value": _display_mapping(mapping or "—")},
        ])

        selected_index = int(st.number_input(
            "Strand ID",
            min_value=1,
            max_value=max(1, len(rows)),
            value=1,
            step=1,
            key="inspect_prepared_strand_no",
        ))
        selected_row = rows[selected_index - 1]
        _render_segmented_strand(selected_row, FIELDS["prepared_strand"])

        st.markdown("##### 🧾 Strand table")
        table_rows = []
        for r in rows[:500]:
            table_rows.append({
                "No.": r.get("No.", "—"),
                "SI": r.get("Strand index", "—"),
                "Payload length": r.get("Payload length", "—"),
                "Filler length": r.get("Filler length", "—"),
                "Total length": r.get("Total length", "—"),
                "GC content": r.get("GC content", "—"),
                "Longest homopolymer": r.get("Longest homopolymer", "—"),
                "Homo count": r.get("Homopolymer count", "—"),
            })
        _render_stats_table(table_rows, height=260)
        if len(rows) > 500:
            st.caption(f"Showing first 500 of {len(rows):,} strands.")

        st.download_button(BUTTONS["download_prepared_strands"], data=strand_rows_to_csv(rows), file_name=DOWNLOAD_FILES["prepared_strands"], mime="text/csv", use_container_width=True)


# -----------------------------------------------------------------------------
# Panel 5 — Decode
# -----------------------------------------------------------------------------


def render_panel_5_decoding() -> None:
    with st.container(border=True):
        step_header(5, "Decoding")
        mapping = st.session_state.get("encoding_mapping")
        if not mapping:
            st.info(MESSAGES["run_dna_encoding_first"])
            return

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 📁 DNA input")
            source = st.radio(
                "DNA source",
                ["Current encoded DNA", "Upload full payload TXT", "Upload strand CSV"],
                horizontal=True,
                key="decode_dna_source",
            )
            if source == "Upload full payload TXT":
                uploaded_dna = st.file_uploader("Upload full DNA payload .txt", type=["txt"], key="decode_encoded_upload")
                if uploaded_dna is not None:
                    raw_text = uploaded_dna.getvalue().decode("utf-8", errors="ignore")
                    dna_text = clean_dna(raw_text)
                    st.session_state["decode_uploaded_dna"] = dna_text
                    source_label = f"Uploaded DNA payload ({uploaded_dna.name})"
                else:
                    dna_text = st.session_state.get("decode_uploaded_dna", "")
                    source_label = "Uploaded DNA payload"
            elif source == "Upload strand CSV":
                uploaded_csv = st.file_uploader("Upload prepared strand CSV", type=["csv"], key="decode_strand_csv_upload")
                if uploaded_csv is not None:
                    dna_text, row_count, note = _dna_from_uploaded_strand_csv(uploaded_csv.getvalue())
                    st.session_state["decode_uploaded_csv_dna"] = dna_text
                    st.session_state["decode_uploaded_csv_note"] = note
                    source_label = f"Uploaded strand CSV ({row_count:,} strands)"
                else:
                    dna_text = st.session_state.get("decode_uploaded_csv_dna", "")
                    source_label = "Uploaded strand CSV"
                note = st.session_state.get("decode_uploaded_csv_note")
                if note:
                    st.caption(note)
            else:
                source_label, dna_text = _decode_source()

            c1, c2 = st.columns(2)
            c1.metric("DNA design", _display_mapping(mapping))
            c2.metric(METRICS["input_dna"], source_label)
            st.text_area("Input DNA payload preview", _preview_seq(dna_text, 600), height=120)

            if st.button(BUTTONS["run_decode"], key="run_decode", type="primary", use_container_width=True):
                try:
                    if not clean_dna(dna_text):
                        raise ValueError("No valid encoded DNA sequence was provided.")
                    data, bits, meta = decode_dna_with_mapping(
                        dna_text,
                        mapping,
                        codec_meta=st.session_state.get("codec_meta", {}) or {},
                    )
                    storage_meta = st.session_state.get("storage_meta", {}) or {}
                    decoded_output = data
                    decoded_raw_pixels = None
                    raw_restore_info: Dict[str, Any] = {}
                    if storage_meta.get("kind") == "raw_image_pixels":
                        decoded_raw_pixels = data
                        decoded_output, raw_restore_info = _raw_image_bytes_to_png(data, storage_meta)
                        m = detect_magic(decoded_output)
                        valid = True
                        note = f"Raw image pixels restored as PNG. {raw_restore_info.get('note', '')}"
                        info = _validate_and_write(decoded_output, preferred="restored_raw_image")
                    else:
                        m = detect_magic(decoded_output)
                        valid = False
                        note = "No recognizable file signature"
                        if m:
                            valid, note = validate_container(decoded_output, m.kind)
                        info = _validate_and_write(decoded_output, preferred="restored")
                    st.session_state.update({
                        "decoded_data": decoded_output,
                        "decoded_raw_pixels": decoded_raw_pixels,
                        "decoded_bits": bits,
                        "decoded_meta": meta,
                        "decoded_magic": m,
                        "decoded_valid": valid,
                        "decoded_note": note,
                        "raw_restore_info": raw_restore_info,
                        "restored_info": info,
                        "decode_source_label": source_label,
                        "decode_error": "",
                    })
                except Exception as exc:
                    st.session_state["decode_error"] = str(exc)
                    st.session_state["restored_info"] = None

            if st.session_state.get("decode_error"):
                st.error(st.session_state["decode_error"])

        with right:
            st.markdown("#### 🖼️ Decoded output")
            data = st.session_state.get("decoded_data")
            if data is None:
                st.info(MESSAGES["run_decode_first"])
                return
            m = st.session_state.get("decoded_magic")
            info = st.session_state.get("restored_info") or {}
            preview_path = info.get("preview_path") or info.get("file_path")
            if preview_path:
                preview_file(preview_path, "Decoded preview")
            _render_property_table(_pipeline_file_metric_rows(preview_path or "", data or b""))
            _render_property_table([
                {"Metric": METRICS["decoded_size"], "Value": fmt_bytes(len(data))},
                {"Metric": METRICS["restored_type"], "Value": m.kind if m else "unknown"},
            ])

            d1, d2 = st.columns(2)
            with d1:
                download_bytes_button(BUTTONS["download_decoded_file"], data, file_name=f"decoded{m.ext if m else '.bin'}")
            with d2:
                _download_full_binary_button(BUTTONS["download_decoded_binary"], data, DOWNLOAD_FILES["decoded_binary"])
            raw_pixels = st.session_state.get("decoded_raw_pixels")
            if raw_pixels is not None:
                r1, r2 = st.columns(2)
                with r1:
                    download_bytes_button("Download decoded raw pixels", raw_pixels, file_name=DOWNLOAD_FILES["decoded_raw_pixels"])
                with r2:
                    _download_full_binary_button("Download decoded raw-pixel binary", raw_pixels, DOWNLOAD_FILES["decoded_raw_pixel_binary"])


# -----------------------------------------------------------------------------
# Panel 6 — Validate
# -----------------------------------------------------------------------------




def render_panel_6_analysis() -> None:
    with st.container(border=True):
        step_header(6, PANEL_TITLES["validation"])
        info = st.session_state.get("restored_info")
        if not info:
            st.info(MESSAGES["run_decode_first"])
            return

        path = info.get("file_path")
        preview_path = info.get("preview_path") or path
        data = st.session_state.get("decoded_data", b"")

        storage_meta = st.session_state.get("storage_meta", {}) or {}
        stored_bytes = st.session_state.get("stored_bytes", b"") or b""
        if storage_meta.get("kind") == "raw_image_pixels":
            recovered_for_match = st.session_state.get("decoded_raw_pixels", b"") or b""
        else:
            recovered_for_match = data or b""
        stored_path = st.session_state.get("stored_file_path")
        input_path = st.session_state.get("input_path")
        input_bytes = st.session_state.get("input_bytes", b"") or b""
        dna = st.session_state.get("dna", "")
        strand_rows = st.session_state.get("strand_rows", [])

        st.markdown("#### 📊 Summary")
        original_col, compressed_col, decoded_col = st.columns(3, gap="large")
        with original_col:
            st.markdown("##### Original")
            if input_path and input_bytes:
                preview_file(input_path, "Original preview")
                _render_property_table(_pipeline_file_metric_rows(input_path, input_bytes))
            else:
                st.info(MESSAGES["upload_first"])
        with compressed_col:
            st.markdown("##### Compressed")
            if stored_path and stored_bytes:
                stored_preview = _previewable_payload_path(stored_path, stored_bytes, "summary_compressed") or stored_path
                preview_file(stored_preview, "Compressed preview", show_caption=False)
                _render_property_table(_pipeline_file_metric_rows(stored_path, stored_bytes, compressed=True))
            else:
                st.info(MESSAGES["run_data_encoding_first"])
        with decoded_col:
            st.markdown("##### Decoded")
            if preview_path:
                preview_file(preview_path, "Decoded preview")
            _render_property_table(_pipeline_file_metric_rows(path, data or b""))

        st.markdown("#### 🧾 Compression analysis")
        _render_stats_table(_compression_analysis_rows(input_path, input_bytes, stored_path, stored_bytes))

        st.markdown("#### 🧾 Encode-decode analysis")
        _render_property_table(_encode_decode_analysis_rows(stored_bytes, recovered_for_match, dna, strand_rows))
