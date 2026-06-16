from __future__ import annotations

import gzip
import io
import lzma
import bz2
import zipfile
import zlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from PIL import Image
except Exception:
    Image = None

import dna_codec
from utils_bits_v2 import bytes_to_bitstring, bitstring_to_bytes, detect_magic
from config import MAPPING_OPTIONS, IMAGE_KINDS

_PAIR_TO_BASE = ("A", "C", "G", "T")
_BASE_TO_PAIR = {"A": 0, "C": 1, "G": 2, "T": 3}
_BYTE_TO_DNA = tuple(
    "".join(_PAIR_TO_BASE[(b >> shift) & 0b11] for shift in (6, 4, 2, 0))
    for b in range(256)
)
_RINF_NIBBLE_TO_DIMER = tuple(dna_codec.DIMERS)
_RINF_DIMER_TO_NIBBLE = {d: i for i, d in enumerate(_RINF_NIBBLE_TO_DIMER)}
_BIT_PREVIEW_BYTES = 512


def _bits_preview_from_bytes(data: bytes) -> str:
    raw = bytes(data or b"")
    shown = raw[:_BIT_PREVIEW_BYTES]
    bits = bytes_to_bitstring(shown)
    if len(raw) > _BIT_PREVIEW_BYTES:
        bits += f"\n\n... preview only: showing first {_BIT_PREVIEW_BYTES:,} of {len(raw):,} bytes."
    return bits


def _simple_encode_bytes_direct(data: bytes) -> str:
    raw = bytes(data or b"")
    if not raw:
        return "A"
    return "A" + "".join(_BYTE_TO_DNA[b] for b in raw)


def _simple_decode_bytes_direct(dna_text: str) -> bytes:
    dna = dna_codec.clean_dna_text(dna_text)
    if not dna:
        return b""
    payload = dna[1:] if dna[0] == "A" else dna
    if len(payload) % 4 != 0:
        raise ValueError("Simple Mapping DNA length is not byte-aligned.")
    out = bytearray()
    for i in range(0, len(payload), 4):
        val = 0
        for base in payload[i:i + 4]:
            val = (val << 2) | _BASE_TO_PAIR[base]
        out.append(val)
    return bytes(out)


def _rinf_encode_bytes_direct(data: bytes) -> str:
    raw = bytes(data or b"")
    out = [_RINF_NIBBLE_TO_DIMER[1]]
    for b in raw:
        out.append(_RINF_NIBBLE_TO_DIMER[(b >> 4) & 0x0F])
        out.append(_RINF_NIBBLE_TO_DIMER[b & 0x0F])
    return "".join(out)


def _rinf_decode_bytes_direct(dna_text: str) -> bytes:
    dna = dna_codec.clean_dna_text(dna_text)
    if not dna:
        return b""
    if len(dna) % 2 != 0:
        raise ValueError("RINF_B16 DNA length must be even.")
    digits = []
    for i in range(0, len(dna), 2):
        dimer = dna[i:i + 2]
        if dimer not in _RINF_DIMER_TO_NIBBLE:
            raise ValueError(f"Dimer {dimer} invalid for RINF_B16.")
        digits.append(_RINF_DIMER_TO_NIBBLE[dimer])
    if not digits or digits[0] != 1:
        raise ValueError("Corrupted RINF_B16 stream: leading digit missing.")
    payload = digits[1:]
    if len(payload) % 2 != 0:
        raise ValueError("RINF_B16 payload is not byte-aligned.")
    return bytes((payload[i] << 4) | payload[i + 1] for i in range(0, len(payload), 2))


def mapping_to_config(mapping_name: str) -> Dict[str, Any]:
    if mapping_name == "Simple Mapping":
        return {
            "mode": "SIMPLE",
            "scheme_name": "RINF_B16",
            "init_dimer": "TA",
            "whiten": False,
        }
    return {
        "mode": "TABLE",
        "scheme_name": mapping_name,
        "init_dimer": "TA",
        "whiten": False,
    }

def encode_bytes_to_dna(data: bytes, mapping_name: str) -> Tuple[str, str, Dict[str, Any]]:
    if mapping_name == "Simple Mapping":
        dna = _simple_encode_bytes_direct(data)
        bits_preview = _bits_preview_from_bytes(data)
        return dna, bits_preview, {
            "mapping": mapping_name,
            "mode": "SIMPLE",
            "scheme_name": "RINF_B16",
            "init_dimer": "TA",
            "bits_len": len(data) * 8,
            "digits_len": len(dna),
            "bytes_len": len(data),
            "bits_preview_only": True,
        }
    if mapping_name == "RINF_B16":
        dna = _rinf_encode_bytes_direct(data)
        bits_preview = _bits_preview_from_bytes(data)
        return dna, bits_preview, {
            "mapping": mapping_name,
            "mode": "TABLE_FAST",
            "scheme_name": "RINF_B16",
            "init_dimer": "TA",
            "bits_len": len(data) * 8,
            "digits_len": len(dna) // 2,
            "bytes_len": len(data),
            "bits_preview_only": True,
        }

    bits = bytes_to_bitstring(data)
    if bits == "":
        bits = "0"

    cfg = mapping_to_config(mapping_name)
    dna, digits = dna_codec.encode_bits_to_dna(
        bits,
        scheme_name=cfg["scheme_name"],
        mode=cfg["mode"],
        seed="rn",
        init_dimer=cfg["init_dimer"],
        prepend_one=True,
        whiten=cfg["whiten"],
        target_gc=0.50,
        w_gc=0.0,
        w_motif=0.0,
        ks=(4, 6),
    )
    meta = {
        "mapping": mapping_name,
        "mode": cfg["mode"],
        "scheme_name": cfg["scheme_name"],
        "init_dimer": "TA",
        "bits_len": len(bits),
        "digits_len": len(digits) if isinstance(digits, list) else None,
        "bytes_len": len(data),
    }
    return dna, bits, meta

def decode_dna_with_mapping(dna: str, mapping_name: str, codec_meta: Optional[Dict[str, Any]] = None) -> Tuple[bytes, str, Dict[str, Any]]:
    if mapping_name == "Simple Mapping":
        data = _simple_decode_bytes_direct(dna)
        bits_preview = _bits_preview_from_bytes(data)
        return data, bits_preview, {
            "mapping": mapping_name,
            "mode": "SIMPLE",
            "scheme_name": "RINF_B16",
            "bits_len": len(data) * 8,
            "bytes_len": len(data),
            "pad_bits_to_byte": 0,
            "digits_len": len(dna_codec.clean_dna_text(dna)),
            "bits_preview_only": True,
        }
    if mapping_name == "RINF_B16":
        data = _rinf_decode_bytes_direct(dna)
        bits_preview = _bits_preview_from_bytes(data)
        return data, bits_preview, {
            "mapping": mapping_name,
            "mode": "TABLE_FAST",
            "scheme_name": "RINF_B16",
            "bits_len": len(data) * 8,
            "bytes_len": len(data),
            "pad_bits_to_byte": 0,
            "digits_len": len(dna_codec.clean_dna_text(dna)) // 2,
            "bits_preview_only": True,
        }

    cfg = mapping_to_config(mapping_name)
    bits, digits = dna_codec.decode_dna_to_bits(
        dna,
        scheme_name=cfg["scheme_name"],
        mode=cfg["mode"],
        seed="rn",
        init_dimer=cfg["init_dimer"],
        remove_leading_one=True,
        whiten=cfg["whiten"],
        target_gc=0.50,
        w_gc=0.0,
        w_motif=0.0,
        ks=(4, 6),
    )
    data, pad_bits = bitstring_to_bytes(bits, pad_to_byte=True)
    meta = {
        "mapping": mapping_name,
        "bits_len": len(bits),
        "bytes_len": len(data),
        "pad_bits_to_byte": pad_bits,
        "digits_len": len(digits) if isinstance(digits, list) else None,
    }
    return data, bits, meta

def validate_container(data: bytes, magic_kind: str) -> Tuple[bool, str]:
    """Lightweight validation beyond magic signature."""
    try:
        if magic_kind in {"docx", "pptx", "xlsx", "epub"}:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                names = set(zf.namelist())
            if magic_kind == "docx" and "word/document.xml" not in names:
                return False, "DOCX is missing word/document.xml"
            if magic_kind == "pptx" and not any(n.startswith("ppt/slides/slide") and n.endswith(".xml") for n in names):
                return False, "PPTX is missing slide XML files"
            if magic_kind == "xlsx" and "xl/workbook.xml" not in names:
                return False, "XLSX is missing xl/workbook.xml"
            if magic_kind == "epub" and "mimetype" not in names:
                return False, "EPUB is missing mimetype"
            return True, f"{magic_kind.upper()} container structure opened successfully"
        if magic_kind == "zip":
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    return False, f"ZIP test failed at {bad}"
            return True, "ZIP container opened successfully"
        if magic_kind == "gzip":
            gzip.decompress(data)
            return True, "GZIP decompressed successfully"
        if magic_kind == "xz":
            lzma.decompress(data, format=lzma.FORMAT_XZ)
            return True, "XZ decompressed successfully"
        if magic_kind == "bz2":
            bz2.decompress(data)
            return True, "BZ2 decompressed successfully"
        if magic_kind == "zlib":
            zlib.decompress(data)
            return True, "ZLIB decompressed successfully"
        if magic_kind in IMAGE_KINDS and Image is not None:
            img = Image.open(io.BytesIO(data))
            img.verify()
            return True, "Image verified successfully"
        return True, "Valid file signature"
    except Exception as e:
        return False, str(e)

def blind_decode_dna(dna_text: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Decode DNA by trying the five visible mapping methods.
    """
    dna = dna_codec.clean_dna_text(dna_text)
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for mapping in MAPPING_OPTIONS:
        row: Dict[str, Any] = {"Mapping": mapping}
        try:
            data, bits, meta = decode_dna_with_mapping(dna, mapping)
            m = detect_magic(data)
            score = 0.0
            valid = False
            if data:
                score += 1.0
            if m:
                score += 10.0 * float(m.confidence)
                ok, note = validate_container(data, m.kind)
                valid = bool(ok)
                if ok:
                    score += 5.0
                else:
                    score -= 2.0
                row.update({
                    "Status": "Valid" if ok else "Weak",
                    "Magic": m.kind,
                    "Ext": m.ext,
                    "Confidence": m.confidence,
                    "Bytes": len(data),
                    "Score": score,
                    "Note": note,
                })
            else:
                row.update({
                    "Status": "No magic",
                    "Magic": "—",
                    "Ext": "—",
                    "Confidence": 0.0,
                    "Bytes": len(data),
                    "Score": score,
                    "Note": "No recognizable file/container signature",
                })

            candidate = {
                "mapping": mapping,
                "data": data,
                "bits": bits,
                "meta": meta,
                "magic": m,
                "score": score,
                "row": row,
            }
            if m is not None and (best is None or candidate["score"] > best["score"]):
                best = candidate


        except Exception as e:
            row.update({
                "Status": "Failed",
                "Magic": "—",
                "Ext": "—",
                "Confidence": 0.0,
                "Bytes": 0,
                "Score": -1.0,
                "Note": str(e)[:160],
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    if best is None:
        raise ValueError("Auto-detection failed: no mapping produced a recognizable self-describing byte stream.")
    return best, df
