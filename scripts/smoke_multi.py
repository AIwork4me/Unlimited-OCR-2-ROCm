"""Use native infer_multi on ordered pages, including repeated runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlimited_ocr_2_rocm.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["smoke-multi", *sys.argv[1:]]))
