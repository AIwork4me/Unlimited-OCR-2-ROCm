"""Write an installer creation witness bound to the actual isolated interpreter."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlimited_ocr_2_rocm.doctor import environment_identity, write_json  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--created-new", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Creation witness requires an isolated interpreter")
    write_json(
        args.output,
        {
            "environment_id": environment_identity(),
            "run_id": args.run_id,
            "created_new": args.created_new,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "executable": sys.executable,
        },
    )
