from __future__ import annotations

import io
import os
import base64
import gzip
import bz2
import lzma
import hashlib
import subprocess
import shutil
import uuid
import zipfile
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mimetypes
import base64

import streamlit as st
import streamlit.components.v1 as components

try:
    from PIL import Image
except Exception:
    Image = None

import dna_codec
from utils_bits_v2 import detect_magic, safe_basename

try:
    from compressors_v2 import detect_domain
except Exception:
    detect_domain = None

from config import (
    WORK_ROOT,
    IMAGE_KINDS,
    AUDIO_KINDS,
    VIDEO_KINDS,
    IMAGE_PREVIEW_USE_CONTAINER_WIDTH,
    IMAGE_PREVIEW_WIDTH,
    TEXT_PREVIEW_HEIGHT,
)

_PREVIEW_FILE_CALL_COUNTER = 0
_OFFICE_KINDS = {"docx", "pptx", "xlsx", "epub"}
_VIEWABLE_MEMBER_EXTS = {".xml", ".rels", ".txt", ".html", ".htm", ".css", ".json", ".csv"}
_MAX_INTERNAL_PREVIEW_BYTES = 256 * 1024
_BROWSER_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}
_VIDEO_EXTS = _BROWSER_VIDEO_EXTS | {".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp", ".ts", ".mts", ".m2ts"}
_BROWSER_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac"}
_OFFICE_EXTS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp"}
_MAX_INLINE_PDF_BYTES = 8 * 1024 * 1024


def fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "—"
    try:
        x = float(n)
    except Exception:
        return "—"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TB"

def step_header(number: int, title: str) -> None:
    st.markdown(
        f"""
<div class="step-heading">
  <span class="step-badge">{number}</span>
  <span class="step-title">{title}</span>
</div>
""",
        unsafe_allow_html=True,
    )

def safe_write_bytes(path: str | Path, data: bytes) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)

def save_upload(uploaded_file) -> Tuple[str, bytes]:
    run_dir = WORK_ROOT / "uploads" / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    name = safe_basename(uploaded_file.name or "upload.bin")
    data = uploaded_file.getvalue()
    path = run_dir / name
    path.write_bytes(data)
    return str(path), data

def magic_dict(data: bytes) -> Dict[str, Any]:
    m = detect_magic(data)
    if not m:
        return {"kind": "unknown", "ext": ".bin", "mime": "application/octet-stream", "confidence": 0.0, "note": ""}
    return {
        "kind": m.kind,
        "ext": m.ext,
        "mime": m.mime,
        "confidence": m.confidence,
        "note": getattr(m, "note", ""),
    }

def get_domain(path: str, data: bytes) -> str:
    ext = Path(path).suffix.lower()
    document_exts = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp"}
    if detect_domain is not None:
        try:
            detected = detect_domain(path, data)
            if detected not in {"unknown", "other", "binary", None}:
                return detected
            if ext in document_exts:
                return "document"
            if detected:
                return detected
        except Exception:
            pass
    m = detect_magic(data)
    if not m:
        if ext in document_exts:
            return "document"
        if ext in _VIDEO_EXTS:
            return "video"
        if ext in _BROWSER_AUDIO_EXTS:
            return "audio"
        return "unknown"
    if m.kind in IMAGE_KINDS:
        return "image"
    if m.kind in AUDIO_KINDS:
        return "audio"
    if m.kind in VIDEO_KINDS:
        return "video"
    if ext in _VIDEO_EXTS:
        return "video"
    if m.kind in {"pdf", "docx", "pptx", "xlsx", "epub"}:
        return "document"
    if m.kind in {"zip", "gzip", "xz", "bz2", "zlib"}:
        return "archive"
    if m.kind == "text":
        return "text"
    return "other"


def _write_preview_inner(path: str, kind: str, title: str) -> Optional[str]:
    try:
        data = Path(path).read_bytes()
        if kind == "gzip":
            inner = gzip.decompress(data)
            suffix = "gunzip"
        elif kind == "xz":
            inner = lzma.decompress(data, format=lzma.FORMAT_XZ)
            suffix = "unxz"
        elif kind == "bz2":
            inner = bz2.decompress(data)
            suffix = "bunzip2"
        elif kind == "zlib":
            inner = zlib.decompress(data)
            suffix = "unzlib"
        elif kind == "zip":
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                if not members:
                    return None
                member = members[0]
                inner = zf.read(member)
                suffix = safe_basename(os.path.basename(member) or "unzip")
        else:
            return None

        inner_magic = detect_magic(inner)
        ext = inner_magic.ext if inner_magic else ".bin"
        preview_dir = WORK_ROOT / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256((title + os.path.abspath(path)).encode("utf-8", errors="ignore") + inner[:4096]).hexdigest()[:16]
        out_path = preview_dir / f"{digest}_{suffix}{ext}"
        out_path.write_bytes(inner)
        return str(out_path)
    except Exception:
        return None


def _preview_cache_path(path: str, suffix: str) -> Path:
    src = Path(path)
    stat = src.stat()
    digest = hashlib.sha256(f"{src.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{suffix}".encode("utf-8", errors="ignore")).hexdigest()[:16]
    out_dir = WORK_ROOT / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{digest}{suffix}"


def _run_preview_command(cmd: List[str], timeout_sec: int = 60) -> bool:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
        return p.returncode == 0
    except Exception:
        return False


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def _preview_button_key(path: str, suffix: str, action: str) -> str:
    src = Path(path)
    try:
        stat = src.stat()
        seed = f"{src.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{suffix}|{action}"
    except Exception:
        seed = f"{path}|{suffix}|{action}"
    return "preview_" + hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _video_mp4_preview(path: str) -> Optional[str]:
    out_path = _preview_cache_path(path, "_preview.mp4")
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-t", "45",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    if _run_preview_command(cmd, timeout_sec=90) and out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)
    return None


def _office_pdf_preview(path: str) -> Optional[str]:
    out_path = _preview_cache_path(path, "_office.pdf")
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)
    out_dir = out_path.parent
    tmp_dir = WORK_ROOT / "office_convert" / out_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(tmp_dir),
        path,
    ]
    if not _run_preview_command(cmd, timeout_sec=90):
        cmd[0] = "soffice"
        if not _run_preview_command(cmd, timeout_sec=90):
            return None
    pdfs = sorted(tmp_dir.glob("*.pdf"))
    if not pdfs:
        return None
    out_path.write_bytes(pdfs[0].read_bytes())
    return str(out_path) if out_path.exists() and out_path.stat().st_size > 0 else None


def _pdf_page_preview(path: str) -> Optional[str]:
    out_prefix = _preview_cache_path(path, "_pdf_page")
    out_png = out_prefix.with_suffix(".png")
    if out_png.exists() and out_png.stat().st_size > 0:
        return str(out_png)
    cmd = ["pdftoppm", "-png", "-f", "1", "-singlefile", "-r", "120", path, str(out_prefix)]
    if _run_preview_command(cmd, timeout_sec=45) and out_png.exists() and out_png.stat().st_size > 0:
        return str(out_png)
    return None


def _preview_pdf(path: str) -> None:
    size = os.path.getsize(path)
    if size <= _MAX_INLINE_PDF_BYTES:
        try:
            data = Path(path).read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            components.html(
                f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="640" type="application/pdf"></iframe>',
                height=660,
            )
        except Exception:
            pass
    else:
        st.info("PDF is large, so inline embed is skipped to keep the Streamlit page responsive.")

    page_cache = _preview_cache_path(path, "_pdf_page").with_suffix(".png")
    page = str(page_cache) if page_cache.exists() and page_cache.stat().st_size > 0 else None
    if page:
        st.caption("First page preview")
        st.image(page, use_container_width=True)
    elif _tool_available("pdftoppm"):
        if st.button("Generate first-page image preview", key=_preview_button_key(path, "_pdf_page", "pdf_page")):
            page = _pdf_page_preview(path)
            if page:
                st.image(page, use_container_width=True)
            else:
                st.info("Could not render the PDF first page.")


def _binary_preview(path: str, size: int, kind: str, ext: str) -> None:
    data = Path(path).read_bytes()[:2048]
    text = data.decode("utf-8", errors="ignore")
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
    ratio = printable / max(1, len(text))
    key_base = hashlib.sha256(f"{os.path.abspath(path)}|{size}|{kind}|{ext}".encode("utf-8", errors="ignore")).hexdigest()[:16]
    if text.strip() and ratio > 0.70:
        st.text_area("Text-like preview", value=text[:4000], height=TEXT_PREVIEW_HEIGHT, label_visibility="collapsed", key=f"text_like_{key_base}")
        return
    hex_rows = []
    for offset in range(0, len(data), 16):
        chunk = data[offset:offset + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        hex_rows.append(f"{offset:08x}  {hex_part:<47}  {ascii_part}")
    st.text_area("Binary preview", value="\n".join(hex_rows), height=TEXT_PREVIEW_HEIGHT, label_visibility="collapsed", key=f"binary_{key_base}")


def _xml_text(xml_bytes: bytes, text_tag_suffix: str = "}t") -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return ""
    chunks: List[str] = []
    for el in root.iter():
        tag = str(el.tag)
        if (tag.endswith(text_tag_suffix) or tag == "t") and el.text:
            value = el.text.strip()
            if value:
                chunks.append(value)
    return " ".join(chunks)


def _office_text_preview(zf: zipfile.ZipFile, kind: str) -> str:
    try:
        if kind == "docx" and "word/document.xml" in zf.namelist():
            return _xml_text(zf.read("word/document.xml"))
        if kind == "pptx":
            slide_names = sorted(n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml"))
            slides = []
            for i, name in enumerate(slide_names[:12], start=1):
                text = _xml_text(zf.read(name))
                if text:
                    slides.append(f"Slide {i}: {text}")
            return "\n\n".join(slides)
        if kind == "xlsx" and "xl/sharedStrings.xml" in zf.namelist():
            return _xml_text(zf.read("xl/sharedStrings.xml"))
        if kind == "epub":
            html_names = sorted(n for n in zf.namelist() if Path(n).suffix.lower() in {".xhtml", ".html", ".htm"})
            chunks = []
            for name in html_names[:5]:
                raw = zf.read(name)[:_MAX_INTERNAL_PREVIEW_BYTES]
                text = raw.decode("utf-8", errors="ignore")
                if text.strip():
                    chunks.append(f"{name}\n{text[:4000]}")
            return "\n\n".join(chunks)
    except Exception:
        return ""
    return ""


def _preview_office_document(path: str, kind: str, title: str) -> None:
    try:
        pdf_cache = _preview_cache_path(path, "_office.pdf")
        pdf_path = str(pdf_cache) if pdf_cache.exists() and pdf_cache.stat().st_size > 0 else None
        if pdf_path:
            st.caption("Office preview rendered as PDF.")
            _preview_pdf(pdf_path)
        elif _tool_available("libreoffice") or _tool_available("soffice"):
            if st.button("Generate PDF preview", key=_preview_button_key(path, "_office.pdf", "office_pdf")):
                pdf_path = _office_pdf_preview(path)
                if pdf_path:
                    st.caption("Office preview rendered as PDF.")
                    _preview_pdf(pdf_path)
                else:
                    st.info("Could not generate an Office PDF preview.")

        with zipfile.ZipFile(path, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            viewable = [
                info for info in infos
                if Path(info.filename).suffix.lower() in _VIEWABLE_MEMBER_EXTS
                and info.file_size <= _MAX_INTERNAL_PREVIEW_BYTES
            ]

            st.caption(f"{kind.upper()} is a ZIP-based Office container.")

            summary_text = _office_text_preview(zf, kind)
            if summary_text:
                st.text_area(
                    "Extracted text preview",
                    value=summary_text[:15000],
                    height=TEXT_PREVIEW_HEIGHT,
                    label_visibility="collapsed",
                    key="office_text_" + hashlib.sha256((title + os.path.abspath(path) + kind).encode("utf-8", errors="ignore")).hexdigest()[:16],
                )

            if infos:
                rows = [
                    {
                        "File": info.filename,
                        "Size": fmt_bytes(info.file_size),
                        "Compressed": fmt_bytes(info.compress_size),
                    }
                    for info in infos[:200]
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

            if viewable:
                key_src = f"{title}|{os.path.abspath(path)}|{kind}|member"
                select_key = "office_member_" + hashlib.sha256(key_src.encode("utf-8", errors="ignore")).hexdigest()[:16]
                selected = st.selectbox(
                    "Reviewable internal file",
                    list(range(len(viewable))),
                    format_func=lambda i: f"{viewable[i].filename} ({fmt_bytes(viewable[i].file_size)})",
                    key=select_key,
                )
                info = viewable[int(selected)]
                raw = zf.read(info.filename)
                text = raw.decode("utf-8", errors="ignore")
                st.text_area(
                    "Internal file preview",
                    value=text[:15000],
                    height=TEXT_PREVIEW_HEIGHT,
                    label_visibility="collapsed",
                    key=select_key + "_text",
                )
            else:
                st.info("No small XML/text files are available for inline review.")
    except Exception as exc:
        st.info(f"Document preview is not available: {exc}")

def _render_natural_image_preview(path: str, ext: str) -> None:
    raw = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"

    # TIFF thường browser không hiển thị trực tiếp tốt, nên để Streamlit xử lý.
    if ext.lower() in {".tif", ".tiff"}:
        st.image(path, use_container_width=False)
        return

    encoded = base64.b64encode(raw).decode("ascii")
    st.markdown(
        f"""
<div style="width: 100%; display: flex; justify-content: center;">
  <img
    src="data:{mime};base64,{encoded}"
    style="
      width: auto;
      height: auto;
      min-width: 50%;
      max-width: 100%;
      object-fit: contain;
      display: block;
    "
  />
</div>
""",
        unsafe_allow_html=True,
    )

def preview_file(path: Optional[str], title: str = "Preview", show_caption: bool = False) -> None:
    st.markdown(f"#### {title}")
    if not path or not os.path.exists(path):
        st.info("No file available.")
        return
    size = os.path.getsize(path)
    data_head = Path(path).read_bytes()[:4096]
    m = detect_magic(Path(path).read_bytes()[: min(size, 1024 * 1024)]) if size <= 512 * 512 else detect_magic(data_head)
    ext = (m.ext if m else Path(path).suffix).lower()
    kind = m.kind if m else "unknown"
    if show_caption:
        st.caption(f"{os.path.basename(path)} · {fmt_bytes(size)} · {kind} · {ext}")

    try:
        if kind in IMAGE_KINDS or ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            _render_natural_image_preview(path, ext)
        elif kind in AUDIO_KINDS or ext in _BROWSER_AUDIO_EXTS:
            st.audio(path)
        elif kind in VIDEO_KINDS or ext in _VIDEO_EXTS:
            if ext in _BROWSER_VIDEO_EXTS:
                st.video(path)
            else:
                mp4_cache = _preview_cache_path(path, "_preview.mp4")
                mp4_path = str(mp4_cache) if mp4_cache.exists() and mp4_cache.stat().st_size > 0 else None
                if mp4_path:
                    st.caption("Browser-compatible MP4 preview")
                    st.video(mp4_path)
                elif _tool_available("ffmpeg"):
                    st.info("This video format is not browser-native. Generate a short MP4 preview only if needed.")
                    if st.button("Generate MP4 preview", key=_preview_button_key(path, "_preview.mp4", "video_mp4")):
                        mp4_path = _video_mp4_preview(path)
                        if mp4_path:
                            st.video(mp4_path)
                        else:
                            st.info("Could not generate an MP4 preview.")
                            _binary_preview(path, size, kind, ext)
                    else:
                        _binary_preview(path, size, kind, ext)
                else:
                    st.info("This video format is not browser-native, and ffmpeg is not available for preview conversion.")
                    _binary_preview(path, size, kind, ext)
        elif kind == "pdf" or ext == ".pdf":
            _preview_pdf(path)
        elif kind in _OFFICE_KINDS:
            _preview_office_document(path, kind, title)
        elif ext in _OFFICE_EXTS:
            pdf_cache = _preview_cache_path(path, "_office.pdf")
            pdf_path = str(pdf_cache) if pdf_cache.exists() and pdf_cache.stat().st_size > 0 else None
            if pdf_path:
                st.caption("Office preview rendered as PDF.")
                _preview_pdf(pdf_path)
            elif _tool_available("libreoffice") or _tool_available("soffice"):
                if st.button("Generate PDF preview", key=_preview_button_key(path, "_office.pdf", "legacy_office_pdf")):
                    pdf_path = _office_pdf_preview(path)
                    if pdf_path:
                        st.caption("Office preview rendered as PDF.")
                        _preview_pdf(pdf_path)
                    else:
                        _binary_preview(path, size, kind, ext)
                else:
                    _binary_preview(path, size, kind, ext)
            else:
                _binary_preview(path, size, kind, ext)
        elif kind == "text" or ext in {".txt", ".md", ".json", ".csv", ".tsv", ".log", ".xml", ".yaml", ".yml", ".html", ".py", ".js"}:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            global _PREVIEW_FILE_CALL_COUNTER
            _PREVIEW_FILE_CALL_COUNTER += 1
            preview_key_src = f"{title}|{os.path.abspath(path)}|{size}|{_PREVIEW_FILE_CALL_COUNTER}"
            preview_key = "text_preview_" + hashlib.sha256(preview_key_src.encode("utf-8", errors="ignore")).hexdigest()[:16]
            st.text_area("Text preview", value=text[:15000], height=TEXT_PREVIEW_HEIGHT, label_visibility="collapsed", key=preview_key)
        elif kind in {"gzip", "xz", "bz2", "zlib", "zip"}:
            inner_path = _write_preview_inner(path, kind, title)
            if inner_path:
                st.caption(f"Previewing decompressed content from {kind}.")
                preview_file(inner_path, "Decompressed preview")
            else:
                _binary_preview(path, size, kind, ext)
        else:
            _binary_preview(path, size, kind, ext)
    except Exception as e:
        st.warning(f"Preview failed: {e}")

def download_bytes_button(label: str, data: bytes, file_name: str, mime: str = "application/octet-stream") -> None:
    st.download_button(label, data=data, file_name=file_name, mime=mime, use_container_width=True)
