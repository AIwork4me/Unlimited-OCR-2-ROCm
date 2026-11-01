"""Real-document GPU benchmark runner: predeclared MPDocBench subset -> official scorer inputs.

One request per document processes the ordered pages of that complete document
through the verified native path (infer_multi). Production limits stay
unchanged: at most five pages and 8192 new tokens per request. Every document,
including failures, stays in the denominator. Results are labeled
REAL_DATA_INTEGRATION_ONLY and must not be published as full MPDocBench
accuracy or leaderboard parity.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from unlimited_ocr_2_rocm._upstream.process_unlimitedOCR2 import convert
from unlimited_ocr_2_rocm.config import file_sha256
from unlimited_ocr_2_rocm.doctor import memory_snapshot
from unlimited_ocr_2_rocm.inference import run_native
from unlimited_ocr_2_rocm.model import load_model


def load_ground_truth(gt_path: Path) -> dict[str, dict]:
    gt = json.loads(gt_path.read_text("utf-8"))
    by_id = {}
    for document in gt:
        info = document["page_info"]
        by_id[info["image_path"].removesuffix(".pdf")] = document
    return by_id


def document_pages(document: dict) -> list[str]:
    return list(document["page_info"]["images_list"])


def verify_page_hashes(data_root: Path, pages: list[str], page_index: dict[str, dict]) -> None:
    for rel in pages:
        expected = page_index.get(rel)
        if expected is None:
            raise ValueError(f"page not present in frozen index: {rel}")
        actual = file_sha256(data_root / rel)
        if actual != expected["sha256"]:
            raise ValueError(f"page hash mismatch against frozen index: {rel}")


def load_page_index(repo_root: Path) -> dict[str, dict]:
    index = json.loads(
        (repo_root / "eval/benchmarks/g05/datasets/MPDocBench-page-index.json").read_text("utf-8")
    )
    return {page["path"]: page for page in index["pages"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args(argv)

    selection = json.loads(args.selection.read_text("utf-8"))
    gt = load_ground_truth(args.gt)
    page_index = load_page_index(args.repo_root)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = args.output / "predictions"
    predictions.mkdir()

    started = time.time()
    model = load_model(args.model_cache, offline=True)
    records = []
    for doc_id, chosen in selection.items():
        document = gt.get(doc_id)
        if document is None:
            raise ValueError(f"selection document absent from ground truth: {doc_id}")
        pages = document_pages(document)
        if len(pages) > 5:
            raise ValueError(f"production five-image limit exceeded by {doc_id}")
        verify_page_hashes(args.data_root, pages, page_index)
        doc_out = args.output / doc_id
        record = {
            "document": doc_id,
            "selection_reason": chosen["reason"],
            "expected_pages": len(pages),
            "pages": [{"path": rel, "sha256": page_index[rel]["sha256"]} for rel in pages],
        }
        t0 = time.time()
        try:
            report = run_native(
                model,
                [args.data_root / rel for rel in pages],
                doc_out,
                multi=True,
                max_new_tokens=args.max_new_tokens,
            )
            record.update(
                status=report["status"],
                finish_reason=report.get("finish_reason"),
                output_token_count=report.get("output_token_count"),
                raw_sha256=report.get("raw_sha256"),
                native_kernel_count=report.get("native_kernel_count"),
                synchronized_wall_seconds=report.get("synchronized_wall_seconds"),
                upstream_timing=report.get("upstream_timing"),
                memory_snapshot=report.get("memory_after"),
            )
            if report.get("status") not in {"INFERENCE_ONLY", "PASS"}:
                record["error"] = report.get("traceback", "").splitlines()[-1:]
            raw_path = doc_out / "raw.txt"
            if raw_path.is_file():
                markdown = convert(raw_path.read_text("utf-8"))
                (predictions / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
                record["prediction_sha256"] = file_sha256(predictions / f"{doc_id}.md")
        except KeyboardInterrupt:
            record.update(status="FAIL", error="INTERRUPTED")
            (args.output / "INTERRUPTED").write_text(doc_id, encoding="utf-8")
            records.append(record)
            break
        except Exception as error:  # noqa: BLE001 - keep the document in the denominator
            record.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        record["document_wall_seconds"] = round(time.time() - t0, 3)
        records.append(record)
        print(f"{doc_id}: {record.get('status')} finish={record.get('finish_reason')}")

    failed = [r["document"] for r in records if r.get("status") not in {"INFERENCE_ONLY", "PASS"}]
    receipt = {
        "schema": 1,
        "label": "REAL_DATA_INTEGRATION_ONLY",
        "claim_limit": (
            "Small predeclared MPDocBench subset; NOT full MPDocBench accuracy, "
            "NOT leaderboard parity, NOT full dataset completion"
        ),
        "host": {"node": platform.node(), "system": platform.system()},
        "model": model.evidence,
        "max_new_tokens": args.max_new_tokens,
        "documents_selected": len(selection),
        "documents_completed": len(records) - len(failed),
        "failed_documents": failed,
        "records": records,
        "memory_final": memory_snapshot(),
        "total_wall_seconds": round(time.time() - started, 3),
    }
    (args.output / "run-receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Scorer input: ground-truth slice restricted to the selected documents,
    # original annotation bytes unmodified.
    gt_slice = [gt[doc_id] for doc_id in selection if doc_id in gt]
    (args.output / "ground-truth-slice.json").write_text(
        json.dumps(gt_slice, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"done: {receipt['documents_completed']}/{len(selection)} completed, "
        f"failed={failed}, label=REAL_DATA_INTEGRATION_ONLY"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
