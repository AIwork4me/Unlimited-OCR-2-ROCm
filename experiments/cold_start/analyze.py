"""Exact token, stage and canonical bbox comparisons; no tolerance acceptance."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from worker import ROOT, digest, load, sha, write

COORDS = r"\[\s*-?\d+(?:\s*[,，]\s*-?\d+){3}\s*\]"
PREFIX = r"<BLOCK>\s*[A-Za-z_]+\s*(?:<LEVEL\d+>\s*)?"
SUFFIX = r"\s*(?:<LEVEL\d+>\s*)?<FORK>"
HEADER = re.compile("(" + PREFIX + ")(" + COORDS + ")(" + SUFFIX + ")")


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def boxes(text):
    for block in text.split("<BLOCK>")[1:]:
        header, fork, _ = block.partition("<FORK>")
        if not fork or HEADER.fullmatch("<BLOCK>" + header + fork) is None:
            raise ValueError("Noncanonical or lossy bbox header")
    parser = load(
        "canonical_bbox_parser",
        ROOT / "src/unlimited_ocr_2_rocm/_upstream/process_unlimitedOCR2.py",
    )
    result = parser._parse_blocks(text)
    if sum(b["label"] != "__raw__" for b in result) != text.count("<BLOCK>"):
        raise ValueError("Unparsed canonical block")
    if any(b["bbox"] is None for b in result):
        raise ValueError("Missing bbox or unparsed content")
    return result


def valid(b):
    return 0 <= b[0] < b[2] <= 999 and 0 <= b[1] < b[3] <= 999


def iou(a, b):
    if not valid(a) or not valid(b):
        return None
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def compare(a, b):
    left, right = boxes(a["text"]), boxes(b["text"])
    alignment = [
        (v["page"], v["label"], v["level"], v["content"], v["merge_prev"]) for v in left
    ] == [(v["page"], v["label"], v["level"], v["content"], v["merge_prev"]) for v in right]
    same_nonbbox = HEADER.sub(r"\1[BBOX]\3", a["text"]) == HEADER.sub(r"\1[BBOX]\3", b["text"])
    coords, changes = [], []
    if alignment:
        for index, (x, y) in enumerate(zip(left, right, strict=True)):
            deltas = [right - left for left, right in zip(x["bbox"], y["bbox"], strict=True)]
            coords.extend(deltas)
            changes.append(
                dict(
                    index=index,
                    page=x["page"],
                    label=x["label"],
                    left=x["bbox"],
                    right=y["bbox"],
                    signed_deltas=deltas,
                    iou=iou(x["bbox"], y["bbox"]),
                    valid_left=valid(x["bbox"]),
                    valid_right=valid(y["bbox"]),
                    boundary_touch=any(v in (0, 999) for v in (*x["bbox"], *y["bbox"])),
                )
            )
    ids_a, ids_b = a["token_ids"], b["token_ids"]
    differing = next(
        (i for i in range(max(len(ids_a), len(ids_b))) if ids_a[i : i + 1] != ids_b[i : i + 1]),
        None,
    )
    return dict(
        raw_exact=a["text"] == b["text"],
        output_ids_exact=ids_a == ids_b,
        prompt_ids_exact=a["prompt_token_ids"] == b["prompt_token_ids"],
        raw_hashes=[__import__("hashlib").sha256(v["text"].encode()).hexdigest() for v in (a, b)],
        token_hashes=[digest(v["token_ids"]) for v in (a, b)],
        first_differing_token=differing,
        token_context=None
        if differing is None
        else [v[max(0, differing - 5) : differing + 6] for v in (ids_a, ids_b)],
        alignment_content_labels_reading_order_exact=alignment,
        exact_nonbbox_bytes=same_nonbbox,
        page_marker_counts=[v["text"].count("<PAGE>") for v in (a, b)],
        signed_delta_distribution=dict(Counter(coords)),
        boxes=changes,
        invalid_boxes=[
            dict(arm=i, index=j, bbox=x["bbox"])
            for i, items in enumerate((left, right))
            for j, x in enumerate(items)
            if not valid(x["bbox"])
        ],
        acceptance="EXACT_PASS"
        if a["text"] == b["text"]
        and ids_a == ids_b
        and a["prompt_token_ids"] == b["prompt_token_ids"]
        else "EXACT_FAIL",
    )


def main():
    out = ROOT / ".b/cold-start-rca"
    pairs = {}
    for case in (
        "english",
        "chinese",
        "mixed",
        "table",
        "math",
        "repeat_unicode_recovery",
        "pages_2",
        "pages_3",
        "pages_5",
        "table_2_pages",
    ):
        root = ROOT / ".b/p240/ab-attempt1/diagnostic"
        paths = [
            root / f"diagnostic_{case}_0_{mode}/private-native.json" for mode in ("math", "auto")
        ]
        pairs[case] = compare(*(read(p) for p in paths))
    write(
        out / "historical-bbox-analysis.json",
        dict(
            scope="New analysis only; PR15 exact parity FAIL preserved",
            pairs=pairs,
            source_hashes={
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "src/unlimited_ocr_2_rocm/_upstream/process_unlimitedOCR2.py",
                )
            },
            coordinate_scale=dict(
                renderer_denominator=999,
                converter_clamp=1000,
                pixel_delta_formula="signed_delta * image_axis_pixels / 999",
            ),
        ),
    )


if __name__ == "__main__":
    main()
