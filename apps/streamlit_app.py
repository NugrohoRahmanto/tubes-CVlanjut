from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from math import ceil, floor
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import DocSAM_GT
from models.DocSAM import DocSAM
from test import (
    CustomSubset,
    count_parameters,
    inference,
    load_para_weights,
    parse_tuple,
)


DEFAULT_CLASSES = ["text", "table", "list", "title", "figure", "_background_"]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "pretrained_model" / "docsam_large_all_dataset.pth"
RUNTIME_ROOT = PROJECT_ROOT / "data" / "streamlit_runtime"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "streamlit"
OCR_SEGMENT_TYPES = {"list", "table", "text", "title"}
LOCAL_TESSERACT_CMD = PROJECT_ROOT / "tools" / "tesseract" / "tesseract.AppImage"
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "eng")
TESSERACT_PSM = os.environ.get("TESSERACT_PSM", "6")


st.set_page_config(
    page_title="DocSAM Local Inference",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --docsam-black: #111111;
        --docsam-charcoal: #2a2a2a;
        --docsam-red: #d71920;
        --docsam-red-dark: #ad1118;
        --docsam-red-soft: #fff1f1;
        --docsam-line: #d7d7d7;
        --docsam-muted: #5a5a5a;
        --docsam-surface: #f6f6f6;
      }
      html,
      body,
      [data-testid="stAppViewContainer"],
      .stApp {
        background: #ffffff;
        color: var(--docsam-black);
      }
      [data-testid="stHeader"],
      [data-testid="stToolbar"] {
        background: #ffffff;
      }
      .block-container {
        max-width: 1320px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
      }
      [data-testid="stSidebar"] {
        display: none;
      }
      .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        border-top: 5px solid var(--docsam-red);
        border-bottom: 1px solid var(--docsam-black);
        margin-bottom: 1.2rem;
        padding: 1rem 0 1.15rem;
      }
      .app-title {
        color: var(--docsam-black);
        font-size: 2.85rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: 0;
        margin: 0;
      }
      .app-subtitle {
        color: var(--docsam-muted);
        font-size: 1rem;
        margin-top: .55rem;
      }
      .status-cluster {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: .55rem;
      }
      .status-pill {
        align-items: center;
        background: #ffffff;
        border: 1px solid var(--docsam-black);
        border-radius: 8px;
        color: var(--docsam-black);
        display: inline-flex;
        gap: .45rem;
        padding: .48rem .72rem;
        white-space: nowrap;
        font-size: .88rem;
        font-weight: 650;
      }
      .status-pill::before {
        background: var(--docsam-red);
        border-radius: 999px;
        content: "";
        height: .58rem;
        width: .58rem;
      }
      .upload-shell {
        background: linear-gradient(90deg, var(--docsam-red) 0 92px, var(--docsam-black) 92px 100%) top / 100% 3px no-repeat,
          var(--docsam-surface);
        border: 1px solid var(--docsam-black);
        border-radius: 8px;
        padding: 1rem 1rem .45rem;
        margin-bottom: 1rem;
      }
      div[data-testid="stFileUploader"] section {
        border: 1px dashed var(--docsam-charcoal);
        background: #ffffff;
        border-radius: 8px;
        min-height: 170px;
      }
      div[data-testid="stFileUploader"] section:hover {
        border-color: var(--docsam-red);
        background: var(--docsam-red-soft);
      }
      div[data-testid="stFileUploader"] button,
      div[data-testid="stDownloadButton"] button {
        background: #ffffff;
        border: 1px solid var(--docsam-black);
        color: var(--docsam-black);
      }
      div[data-testid="stFileUploader"] button:hover,
      div[data-testid="stDownloadButton"] button:hover {
        border-color: var(--docsam-red);
        color: var(--docsam-red-dark);
      }
      .control-shell {
        background: #ffffff;
        border-bottom: 1px solid var(--docsam-line);
        border-top: 1px solid var(--docsam-line);
        margin-top: .6rem;
        padding: 1rem 0 .8rem;
      }
      div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--docsam-red);
        border: 1px solid var(--docsam-red);
        color: #ffffff;
        min-height: 2.75rem;
        font-weight: 700;
      }
      div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--docsam-red-dark);
        border-color: var(--docsam-red-dark);
        color: #ffffff;
      }
      div[data-testid="stButton"] > button[kind="primary"]:disabled {
        background: #efefef;
        border-color: #c7c7c7;
        color: #777777;
      }
      div[data-testid="stSegmentedControl"] button {
        border-radius: 6px;
        font-weight: 650;
      }
      div[data-testid="stSlider"] [role="slider"] {
        box-shadow: 0 0 0 2px #ffffff, 0 0 0 3px var(--docsam-black);
      }
      .result-title {
        border-left: 4px solid var(--docsam-red);
        color: var(--docsam-black);
        font-size: 1rem;
        font-weight: 800;
        margin: 1.35rem 0 .65rem;
        padding-left: .65rem;
        text-transform: uppercase;
      }
      [data-testid="stImage"] img {
        border: 1px solid var(--docsam-black);
        border-radius: 8px;
        background: #ffffff;
      }
      button[data-baseweb="tab"] {
        color: var(--docsam-charcoal);
        font-weight: 650;
      }
      button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--docsam-black);
      }
      [data-baseweb="tab-highlight"] {
        background-color: var(--docsam-red);
      }
      [data-testid="stExpander"] details {
        background: var(--docsam-surface);
        border: 1px solid var(--docsam-line);
        border-radius: 8px;
      }
      [data-testid="stExpander"] summary {
        color: var(--docsam-black);
        font-weight: 650;
      }
      [data-testid="stAlert"] {
        border-radius: 8px;
      }
      @media (max-width: 720px) {
        .app-header {
          display: block;
          padding-top: .9rem;
        }
        .app-title {
          font-size: 2.1rem;
        }
        .status-cluster {
          justify-content: flex-start;
          margin-top: .9rem;
        }
        .status-pill {
          font-size: .84rem;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_filename(name: str) -> str:
    name = Path(name).name
    stem = Path(name).stem or "document"
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{stem or 'document'}{suffix or '.png'}"


def parse_classes(raw_value: str) -> list[str]:
    classes = [item.strip() for item in raw_value.splitlines() if item.strip()]
    if not classes:
        classes = DEFAULT_CLASSES[:]
    if classes[-1] != "_background_":
        classes.append("_background_")
    return classes


def ensure_runtime_dataset(uploaded_file, classes: list[str]) -> tuple[Path, str]:
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    dataset_dir = RUNTIME_ROOT / run_id
    image_dir = dataset_dir / "image"
    class_dir = dataset_dir / "class_name"
    image_dir.mkdir(parents=True, exist_ok=True)
    class_dir.mkdir(parents=True, exist_ok=True)

    filename = clean_filename(uploaded_file.name)
    image_path = image_dir / filename
    image_path.write_bytes(uploaded_file.getbuffer())

    (dataset_dir / "list.txt").write_text(filename + "\n", encoding="utf-8")
    (class_dir / f"{Path(filename).stem}.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    return dataset_dir, Path(filename).stem


def build_args(
    dataset_dir: Path,
    save_path: Path,
    checkpoint_path: Path,
    model_size: str,
    short_range: str,
    patch_size: str,
    gpu_ids: str,
    score_threshold: float,
    mask_alpha: float,
    keep_size: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        stage="inference",
        model_size=model_size,
        eval_path=[str(dataset_dir)],
        save_path=str(save_path),
        short_range=parse_tuple(short_range),
        patch_size=parse_tuple(patch_size),
        patch_num=1,
        keep_size=keep_size,
        max_num=1,
        batch_size=1,
        predict_batch_size=2,
        restore_from=str(checkpoint_path),
        gpus=gpu_ids,
        visual_score_threshold=score_threshold,
        visual_mask_alpha=mask_alpha,
        visual_use_original=True,
    )


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    model = DocSAM(model_size=model_size)
    model = load_para_weights(model, str(checkpoint))
    return model


def run_inference(args: SimpleNamespace, model: DocSAM):
    dataset = DocSAM_GT(
        args.eval_path,
        short_range=args.short_range,
        patch_size=args.patch_size,
        patch_num=args.patch_num,
        keep_size=args.keep_size,
        stage=args.stage,
    )
    subset = CustomSubset(dataset, range(0, min(args.max_num, len(dataset))))
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=subset.collate_fn,
    )
    inference(args, model, loader, gpu_id=0, save_num=1, stage=args.stage)


def find_result_files(save_path: Path, image_stem: str) -> dict[str, Path | None]:
    matches = list(save_path.rglob(f"{image_stem}*"))
    return {
        "original": next((p for p in matches if p.name == f"{image_stem}.png"), None),
        "instance": next((p for p in matches if p.name == f"{image_stem}_instance_dt.png"), None),
        "category": next((p for p in matches if p.name == f"{image_stem}_category_dt.png"), None),
        "semantic": next((p for p in matches if p.name == f"{image_stem}_semantic_dt.png"), None),
        "jsonl": next((p for p in matches if p.name == f"{image_stem}_instance_dt.jsonl"), None),
    }


def read_jsonl(path: Path | None, limit: int | None = None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def compact_bbox(raw_bbox) -> dict[str, float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x, y, width, height = [round(float(item), 2) for item in raw_bbox]
    except (TypeError, ValueError):
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def normalize_ocr_text(raw_text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines()]
    return "\n".join(line for line in lines if line)


def resolve_tesseract_cmd() -> str | None:
    override = os.environ.get("TESSERACT_CMD")
    if override:
        return override
    if LOCAL_TESSERACT_CMD.is_file():
        return str(LOCAL_TESSERACT_CMD)
    return shutil.which("tesseract")


def crop_bbox(image: Image.Image, bbox: dict[str, float]) -> Image.Image | None:
    left = max(0, floor(bbox["x"]))
    top = max(0, floor(bbox["y"]))
    right = min(image.width, ceil(bbox["x"] + bbox["width"]))
    bottom = min(image.height, ceil(bbox["y"] + bbox["height"]))
    if left >= right or top >= bottom:
        return None
    return image.crop((left, top, right, bottom))


def run_tesseract_ocr(image: Image.Image, tesseract_cmd: str, lang: str) -> str:
    with tempfile.TemporaryDirectory(prefix="docsam_ocr_") as temp_dir:
        crop_path = Path(temp_dir) / "segment.png"
        image.save(crop_path)
        result = subprocess.run(
            [
                tesseract_cmd,
                str(crop_path),
                "stdout",
                "--oem",
                "1",
                "--psm",
                TESSERACT_PSM,
                "-l",
                lang,
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
    return normalize_ocr_text(result.stdout)


def build_chunking_json(
    rows: list[dict],
    document_name: str,
    score_threshold: float,
    image: Image.Image | None = None,
    tesseract_cmd: str | None = None,
    ocr_lang: str = TESSERACT_LANG,
) -> dict:
    segments = []
    ocr_errors = []
    for row in rows:
        category = str(row.get("category_id", "unknown"))
        bbox = compact_bbox(row.get("bbox"))
        try:
            raw_score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            raw_score = 0.0
        if category == "_background_" or bbox is None or raw_score < score_threshold:
            continue
        text = ""
        if image is not None and tesseract_cmd and not ocr_errors and category in OCR_SEGMENT_TYPES:
            crop = crop_bbox(image, bbox)
            if crop is not None:
                try:
                    text = run_tesseract_ocr(crop, tesseract_cmd, ocr_lang)
                except (OSError, subprocess.SubprocessError) as exc:
                    ocr_errors.append(str(exc))
        segments.append(
            {
                "type": category,
                "score": round(raw_score, 4),
                "bbox": bbox,
                "text": text,
            }
        )

    segments.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    for index, segment in enumerate(segments, start=1):
        segment["segment_id"] = f"segment_{index:03d}"

    chunk_groups = []
    active_title = None
    active_segments = []
    for segment in segments:
        if segment["type"] == "title":
            if active_segments:
                chunk_groups.append((active_title, active_segments))
            active_title = segment
            active_segments = [segment]
            continue
        active_segments.append(segment)

    if active_segments:
        chunk_groups.append((active_title, active_segments))

    chunks = []
    for index, (title_segment, chunk_segments) in enumerate(chunk_groups, start=1):
        chunk_text = "\n\n".join(segment["text"] for segment in chunk_segments if segment["text"])
        chunks.append(
            {
                "chunk_id": f"chunk_{index:03d}",
                "title_segment_id": title_segment["segment_id"] if title_segment else None,
                "title_text": title_segment["text"] if title_segment else "",
                "segment_count": len(chunk_segments),
                "text": chunk_text,
                "segments": chunk_segments,
            }
        )

    if image is None:
        ocr_status = "skipped"
        ocr_message = "Gambar sumber OCR tidak tersedia."
    elif not tesseract_cmd:
        ocr_status = "unavailable"
        ocr_message = "Executable tesseract tidak ditemukan di PATH atau TESSERACT_CMD."
    elif ocr_errors:
        ocr_status = "error"
        ocr_message = ocr_errors[0]
    else:
        ocr_status = "ok"
        ocr_message = ""

    return {
        "document_name": document_name,
        "strategy": "detected_title_segments",
        "score_threshold": round(score_threshold, 2),
        "segment_count": len(segments),
        "title_segment_count": sum(segment["type"] == "title" for segment in segments),
        "ocr": {
            "engine": "tesseract",
            "language": ocr_lang,
            "status": ocr_status,
            "message": ocr_message,
        },
        "chunks": chunks,
    }


def save_chunking_jsonl(jsonl_path: Path | None, image_stem: str, payload: dict) -> Path | None:
    if jsonl_path is None:
        return None
    chunk_path = jsonl_path.with_name(f"{image_stem}_chunks.jsonl")
    lines = []
    for chunk in payload["chunks"]:
        lines.append(
            json.dumps(
                {
                    "document_name": payload["document_name"],
                    "strategy": payload["strategy"],
                    "score_threshold": payload["score_threshold"],
                    "ocr": payload["ocr"],
                    **chunk,
                },
                ensure_ascii=False,
            )
        )
    chunk_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return chunk_path


def show_image(path: Path | None, caption: str):
    if path and path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} belum tersedia.")


@st.fragment
def show_download_button(label: str, path: Path, mime: str, key: str):
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=key,
    )


def clear_inference_result():
    st.session_state.pop("inference_result", None)


def accelerator_status() -> tuple[bool, str]:
    if torch.cuda.is_available():
        return True, "CUDA ready"
    if torch.backends.mps.is_available():
        return True, "Apple MPS ready"
    return False, "GPU unavailable"


def show_inference_result(result: dict):
    result_files = result["files"]
    rows = result["rows"]
    chunking_json = result["chunking_json"]
    chunking_path = result["chunking_path"]

    st.success(f"Inference selesai. Output disimpan di {result['save_path']}")
    st.markdown('<div class="result-title">Result</div>', unsafe_allow_html=True)
    tab_instance, tab_category, tab_semantic, tab_chunks, tab_json = st.tabs(
        ["Instance", "Category", "Semantic", "Chunking JSONL", "Detection JSONL"]
    )
    with tab_instance:
        show_image(result_files["instance"], "Instance mask + bbox")
    with tab_category:
        show_image(result_files["category"], "Category mask")
    with tab_semantic:
        show_image(result_files["semantic"], "Semantic mask")
    with tab_chunks:
        if chunking_json["ocr"]["status"] != "ok":
            st.warning(chunking_json["ocr"]["message"])
        st.json(chunking_json, expanded=2)
        if chunking_path and chunking_path.is_file():
            show_download_button(
                "Download Chunking JSONL",
                chunking_path,
                "application/jsonl",
                key=f"download-chunking-{chunking_path}",
            )
    with tab_json:
        if rows:
            compact_rows = [
                {
                    "category": row.get("category_id"),
                    "score": round(float(row.get("score", 0.0)), 4),
                    "bbox": [round(float(x), 2) for x in row.get("bbox", [])],
                }
                for row in rows[:100]
            ]
            st.dataframe(compact_rows, use_container_width=True)
            jsonl_path = result_files["jsonl"]
            if jsonl_path and jsonl_path.is_file():
                show_download_button(
                    "Download JSONL",
                    jsonl_path,
                    "application/jsonl",
                    key=f"download-detections-{jsonl_path}",
                )
        else:
            st.info("JSONL belum tersedia.")

    instance_path = result_files["instance"]
    if instance_path and instance_path.is_file():
        show_download_button(
            "Download Result PNG",
            instance_path,
            "image/png",
            key=f"download-result-{instance_path}",
        )


def main():
    checkpoint_path = DEFAULT_CHECKPOINT
    gpu_ids = os.environ.get("GPU_IDS", "0")
    mask_alpha = 0.28
    short_range = "704,896"
    patch_size = "640,640"
    keep_size = False

    accelerator_ready, accelerator_label = accelerator_status()
    checkpoint_label = "model ready" if checkpoint_path.is_file() else "model missing"
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <h1 class="app-title">DocSAM Inference</h1>
            <div class="app-subtitle">Upload dokumen, pilih model, atur threshold, lalu lihat hasil segmentasi lokal.</div>
          </div>
          <div class="status-cluster">
            <div class="status-pill">{accelerator_label}</div>
            <div class="status-pill">{checkpoint_label}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop atau pilih gambar dokumen",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        label_visibility="visible",
        key="document_upload",
        on_change=clear_inference_result,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    can_run = uploaded_file is not None and checkpoint_path.is_file() and accelerator_ready
    if uploaded_file is not None and not checkpoint_path.is_file():
        st.error(f"Checkpoint tidak ditemukan: {checkpoint_path}")
    if uploaded_file is not None and not accelerator_ready:
        st.error("GPU tidak tersedia. Gunakan GPU CUDA atau Apple Silicon MPS.")

    st.markdown('<div class="control-shell">', unsafe_allow_html=True)
    control_left, control_mid, control_right = st.columns([1.1, 1.4, 1])
    with control_left:
        model_size = st.segmented_control("Model size", ["large", "base"], default="large")
    with control_mid:
        score_threshold = st.slider("Threshold label", 0.05, 0.95, 0.50, 0.05)
    with control_right:
        st.write("")
        st.write("")
        run_button = st.button("Run inference", type="primary", disabled=not can_run, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="result-title">Input</div>', unsafe_allow_html=True)
        preview = Image.open(uploaded_file).convert("RGB")
        st.image(preview, caption=uploaded_file.name, use_container_width=True)

    if run_button and uploaded_file is not None:
        classes = DEFAULT_CLASSES[:]
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        dataset_dir, image_stem = ensure_runtime_dataset(uploaded_file, classes)
        save_path = OUTPUT_ROOT / dataset_dir.name
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            args = build_args(
                dataset_dir=dataset_dir,
                save_path=save_path,
                checkpoint_path=checkpoint_path,
                model_size=model_size,
                short_range=short_range,
                patch_size=patch_size,
                gpu_ids=gpu_ids,
                score_threshold=score_threshold,
                mask_alpha=mask_alpha,
                keep_size=keep_size,
            )
        except Exception as exc:
            st.error(f"Config tidak valid: {exc}")
            return

        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        with st.spinner("Loading model dan menjalankan inference..."):
            try:
                model = load_model(model_size, str(checkpoint_path))
                run_inference(args, model)
            except Exception as exc:
                st.error(f"Inference gagal: {exc}")
                return

        result_files = find_result_files(save_path, image_stem)
        rows = read_jsonl(result_files["jsonl"])
        tesseract_cmd = resolve_tesseract_cmd()
        chunking_json = build_chunking_json(
            rows,
            uploaded_file.name,
            score_threshold,
            image=preview,
            tesseract_cmd=tesseract_cmd,
        )
        chunking_path = save_chunking_jsonl(result_files["jsonl"], image_stem, chunking_json)
        st.session_state["inference_result"] = {
            "files": result_files,
            "rows": rows,
            "chunking_json": chunking_json,
            "chunking_path": chunking_path,
            "save_path": save_path,
        }

    inference_result = st.session_state.get("inference_result")
    if inference_result:
        show_inference_result(inference_result)

    with st.expander("Run command"):
        st.code("uv run streamlit run apps/streamlit_app.py", language="bash")
        st.code(
            "BIND_ADDRESS=0.0.0.0 PORT=8501 uv run sh scripts/run_streamlit_app.sh",
            language="bash",
        )


if __name__ == "__main__":
    main()
