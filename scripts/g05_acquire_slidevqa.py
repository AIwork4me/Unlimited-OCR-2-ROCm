"""Acquire authorized frozen SlideVQA shards and reproduce Qwen's JPEG saves.

No remote dataset code is executed. The user must first make an authorized HF
token available through a standard local token file or process environment.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from g05_acquire_datasets import DATA, ROOT, digest, download_file, require, write_json

REVISION = "e780ea258d734958befbfaed19fa059584642ca8"
VERSIONS = {"datasets": "5.1.0", "pyarrow": "25.0.1", "Pillow": "10.4.0"}


def active_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if value := os.environ.get(name):
            return value.strip()
    default = Path.home() / ".cache/huggingface"
    token_path = Path(
        os.environ.get("HF_TOKEN_PATH", str(Path(os.getenv("HF_HOME", default)) / "token"))
    )
    for path in (token_path, Path.home() / ".huggingface/token"):
        if path.is_file() and (value := path.read_text("utf-8").strip()):
            return value
    return None


def extract_frozen_decks(
    shards: list[Path], deck_urls: dict[str, str], page_counts: dict[str, int], output_root: Path
) -> dict:
    """Iterate official test order, preserving official last-match overwrite behavior."""
    import datasets
    import pyarrow.parquet as pq

    ids_by_url = {url: document for document, url in deck_urls.items()}
    require(len(ids_by_url) == len(deck_urls), "ambiguous frozen deck URL")
    decoder = datasets.Image()
    pages = {}
    matching_rows = 0
    matched_decks = set()
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        columns = ["deck_url"] + [
            f"page_{number}" for number in range(1, max(page_counts.values()) + 1)
        ]
        for row_index, batch in enumerate(parquet.iter_batches(batch_size=1, columns=columns)):
            url = batch.column(batch.schema.get_field_index("deck_url"))[0].as_py()
            if url not in ids_by_url:
                continue
            row = batch.to_pylist()[0]
            document = ids_by_url[url]
            matching_rows += 1
            matched_decks.add(document)
            for number in range(1, page_counts[document] + 1):
                payload = row.get(f"page_{number}")
                require(
                    payload is not None and payload.get("bytes") is not None,
                    f"missing expected page: {document}/page_{number}.jpg",
                )
                relative = f"images/{document}/page_{number}.jpg"
                output = output_root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".partial.jpg")
                image = decoder.decode_example(payload)
                # Exact official semantics: no mode conversion, resize, quality or format override.
                image.save(temporary)
                record = {
                    "path": relative,
                    "input_image_bytes_sha256": hashlib.sha256(payload["bytes"]).hexdigest(),
                    "output_sha256": digest(temporary),
                    "output_bytes": temporary.stat().st_size,
                    "source_shard": shard.name,
                    "source_row_index": row_index,
                    "decoded_width": image.width,
                    "decoded_height": image.height,
                    "decoded_mode": image.mode,
                }
                image.close()
                temporary.replace(output)
                pages[relative] = record
    require(matched_decks == set(deck_urls), "not all frozen decks were present")
    require(len(pages) == sum(page_counts.values()), "expected page count mismatch")
    for path, record in pages.items():
        require(digest(output_root / path) == record["output_sha256"], "output reread mismatch")
    return {"matching_rows": matching_rows, "pages": [pages[path] for path in sorted(pages)]}


def frozen_inputs() -> tuple[dict, dict, dict, list[dict]]:
    expected = json.loads((ROOT / "eval/benchmarks/benchmark-lock.json").read_text("utf-8"))[
        "datasets"
    ]["MPDocBench"]
    slide = expected["slidevqa"]
    require(slide["revision"] == REVISION, "SlideVQA revision mismatch")
    downloader = ROOT / ".b/benchmark-audit/qwen/MPDocBench/tools/download_data_from_slidevqa.py"
    require(
        digest(downloader) == slide["downloader_sha256"], "official downloader identity mismatch"
    )
    syntax = ast.parse(downloader.read_text("utf-8"))
    deck_urls = ast.literal_eval(syntax.body[0].value)
    require(set(deck_urls) == set(slide["document_ids"]), "frozen deck scope mismatch")
    gt_path = ROOT / ".b/benchmark-audit/qwen/MPDocBench/MPDocBench.json"
    require(digest(gt_path) == expected["ground_truth_sha256"], "ground truth identity mismatch")
    page_counts = {
        doc["page_info"]["image_path"].removesuffix(".pdf"): len(doc["page_info"]["images_list"])
        for doc in json.loads(gt_path.read_text("utf-8"))
        if doc["page_info"]["image_path"].removesuffix(".pdf") in deck_urls
    }
    require(
        len(page_counts) == 11 and sum(page_counts.values()) == 84, "11-deck/84-page scope mismatch"
    )
    metadata = json.loads((ROOT / ".b/benchmark-audit/slidevqa-meta.json").read_text("utf-8"))
    require(metadata["sha"] == REVISION, "metadata revision mismatch")
    by_path = {item["rfilename"]: item for item in metadata["siblings"]}
    shards = [by_path[name] for name in sorted(slide["test_shards"])]
    for item in shards:
        require(
            item["lfs"]["sha256"] == slide["test_shards"][item["rfilename"]],
            "frozen shard hash mismatch",
        )
    return slide, deck_urls, page_counts, shards


def main() -> None:
    token = active_token()
    require(token is not None, "No active authorized local Hugging Face token is available")
    # Resolve the user's active token first, then confine library caches to this root.
    for name, relative in {
        "HF_HOME": "cache/huggingface",
        "HF_DATASETS_CACHE": "cache/datasets",
        "TMP": "tmp",
        "TEMP": "tmp",
    }.items():
        directory = DATA / "SlideVQA" / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    installed = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(
        installed == VERSIONS, "serialization dependency versions differ from reviewed versions"
    )
    slide, deck_urls, counts, shards = frozen_inputs()
    receipt = {
        "schema": 1,
        "dataset": "SlideVQA",
        "revision": REVISION,
        "expected_decks": 11,
        "expected_pages": 84,
        "authenticated": True,
        "credentials_in_receipt": False,
        "signed_urls_in_receipt": False,
        "serialization_versions": installed,
        "official_downloader_sha256": slide["downloader_sha256"],
        "producer_sha256": digest(Path(__file__)),
        "source_constraint_receipt": "SlideVQA-attempt-1.json",
        "source_constraint_receipt_sha256": digest(
            ROOT / "eval/benchmarks/g05/datasets/SlideVQA-attempt-1.json"
        ),
    }
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for item in shards:
                name = item["rfilename"]
                url = f"https://huggingface.co/datasets/NTT-hil-insight/SlideVQA/resolve/{REVISION}/{name}"
                futures.append(
                    pool.submit(
                        download_file,
                        url,
                        DATA / "SlideVQA" / name,
                        item["lfs"]["size"],
                        item["lfs"]["sha256"],
                        token=token,
                    )
                )
            for number, future in enumerate(as_completed(futures), 1):
                future.result()
                print(
                    json.dumps({"verified_test_shards": number, "expected": len(shards)}),
                    flush=True,
                )
        for item in shards:
            require(
                digest(DATA / "SlideVQA" / item["rfilename"]) == item["lfs"]["sha256"],
                "independent shard reread mismatch",
            )
        result = extract_frozen_decks(
            [DATA / "SlideVQA" / item["rfilename"] for item in shards],
            deck_urls,
            counts,
            DATA / "MPDocBench",
        )
        index = ROOT / "eval/benchmarks/g05/datasets/SlideVQA-page-index.json"
        write_json(index, {"revision": REVISION, **result})
        receipt.update(
            {
                "status": "VERIFIED",
                "verified_decks": len(counts),
                "verified_pages": len(result["pages"]),
                "verified_shards": len(shards),
                "test_shard_sha256": slide["test_shards"],
                "page_hash_index": "SlideVQA-page-index.json",
                "page_hash_index_sha256": digest(index),
                "serialization": (
                    "datasets.Image.decode_example then PIL.Image.save to .jpg with defaults"
                ),
                "independent_verification": (
                    "All shard bytes reread against pinned hashes; "
                    "all output JPEGs reread against serialization hashes"
                ),
            }
        )
    except Exception as exc:
        receipt.update({"status": "BLOCKED_ACQUISITION_ERROR", "error_type": type(exc).__name__})
        if isinstance(exc, urllib.error.HTTPError):
            receipt["http_status"] = exc.code
        write_json(ROOT / "eval/benchmarks/g05/datasets/SlideVQA-attempt-2.json", receipt)
        raise SystemExit("Authorized acquisition failed; see sanitized attempt-2 receipt") from None
    write_json(ROOT / "eval/benchmarks/g05/datasets/SlideVQA-attempt-2.json", receipt)
    write_json(ROOT / "eval/benchmarks/g05/datasets/SlideVQA.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
