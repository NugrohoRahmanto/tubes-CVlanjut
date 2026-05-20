from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import uuid
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


st.set_page_config(
    page_title="DocSAM Local Inference",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
      }
      [data-testid="stSidebar"] {
        display: none;
      }
      .app-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.25rem;
      }
      .app-title {
        font-size: 2.25rem;
        line-height: 1.05;
        font-weight: 750;
        letter-spacing: 0;
        margin: 0;
      }
      .app-subtitle {
        color: #667085;
        font-size: 1rem;
        margin-top: .45rem;
      }
      .status-pill {
        border: 1px solid #d0d5dd;
        border-radius: 999px;
        padding: .45rem .8rem;
        color: #344054;
        background: #fff;
        white-space: nowrap;
        font-size: .9rem;
      }
      .upload-shell {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 1rem 1rem .35rem;
        background: #fcfcfd;
        margin-bottom: 1rem;
      }
      div[data-testid="stFileUploader"] section {
        border: 1px dashed #98a2b3;
        background: #ffffff;
        border-radius: 8px;
        min-height: 170px;
      }
      .control-shell {
        border-top: 1px solid #eaecf0;
        padding-top: 1rem;
        margin-top: .5rem;
      }
      .result-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.25rem 0 .5rem;
      }
      @media (max-width: 720px) {
        .app-header {
          display: block;
        }
        .status-pill {
          display: inline-block;
          margin-top: .8rem;
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


def read_jsonl(path: Path | None, limit: int = 100) -> list[dict]:
    if path is None or not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def show_image(path: Path | None, caption: str):
    if path and path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} belum tersedia.")


def main():
    checkpoint_path = DEFAULT_CHECKPOINT
    gpu_ids = os.environ.get("GPU_IDS", "0")
    mask_alpha = 0.28
    short_range = "704,896"
    patch_size = "640,640"
    keep_size = False

    cuda_label = "CUDA ready" if torch.cuda.is_available() else "CUDA unavailable"
    checkpoint_label = "model ready" if checkpoint_path.is_file() else "model missing"
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <h1 class="app-title">DocSAM Inference</h1>
            <div class="app-subtitle">Upload dokumen, pilih model, atur threshold, lalu lihat hasil segmentasi lokal.</div>
          </div>
          <div class="status-pill">{cuda_label} · {checkpoint_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop atau pilih gambar dokumen",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    can_run = uploaded_file is not None and checkpoint_path.is_file() and torch.cuda.is_available()
    if uploaded_file is not None and not checkpoint_path.is_file():
        st.error(f"Checkpoint tidak ditemukan: {checkpoint_path}")
    if uploaded_file is not None and not torch.cuda.is_available():
        st.error("CUDA tidak tersedia. Pipeline DocSAM ini membutuhkan GPU CUDA.")

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

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        with st.spinner("Loading model dan menjalankan inference..."):
            try:
                model = load_model(model_size, str(checkpoint_path))
                run_inference(args, model)
            except Exception as exc:
                st.error(f"Inference gagal: {exc}")
                return

        result_files = find_result_files(save_path, image_stem)
        st.success(f"Inference selesai. Output disimpan di {save_path}")

        st.markdown('<div class="result-title">Result</div>', unsafe_allow_html=True)
        tab_instance, tab_category, tab_semantic, tab_json = st.tabs(
            ["Instance", "Category", "Semantic", "JSONL"]
        )
        with tab_instance:
            show_image(result_files["instance"], "Instance mask + bbox")
        with tab_category:
            show_image(result_files["category"], "Category mask")
        with tab_semantic:
            show_image(result_files["semantic"], "Semantic mask")
        with tab_json:
            rows = read_jsonl(result_files["jsonl"])
            if rows:
                compact_rows = [
                    {
                        "category": row.get("category_id"),
                        "score": round(float(row.get("score", 0.0)), 4),
                        "bbox": [round(float(x), 2) for x in row.get("bbox", [])],
                    }
                    for row in rows
                ]
                st.dataframe(compact_rows, use_container_width=True)
                st.download_button(
                    "Download JSONL",
                    data=result_files["jsonl"].read_bytes(),
                    file_name=result_files["jsonl"].name,
                    mime="application/jsonl",
                )
            else:
                st.info("JSONL belum tersedia.")

        if result_files["instance"] and result_files["instance"].is_file():
            st.download_button(
                "Download Result PNG",
                data=result_files["instance"].read_bytes(),
                file_name=result_files["instance"].name,
                mime="image/png",
            )

    with st.expander("Run command"):
        st.code("uv run streamlit run apps/streamlit_app.py", language="bash")
        st.code(
            "BIND_ADDRESS=0.0.0.0 PORT=8501 uv run sh scripts/run_streamlit_app.sh",
            language="bash",
        )


if __name__ == "__main__":
    main()
