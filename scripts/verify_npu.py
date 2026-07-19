"""Standalone NPU / execution-provider verification.

    python scripts/verify_npu.py

Exit code 0 when the QNN (Snapdragon NPU) execution provider is available,
1 otherwise. Handy as a boot check and as evidence for the judges.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.npu import verify_npu  # noqa: E402


def main() -> int:
    info = verify_npu()
    print(json.dumps(info, indent=2))
    if not info.get("onnxruntime_available"):
        print("\n[FAIL] onnxruntime is not installed. Install requirements-npu.txt.", file=sys.stderr)
        return 1
    if not info.get("qnn_available"):
        print("\n[WARN] QNN execution provider not found -- NPU offload unavailable.", file=sys.stderr)
        return 1
    print("\n[OK] QNN execution provider available -- NPU ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
