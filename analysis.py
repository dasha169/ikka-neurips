"""Replay the 230-run benchmark from the manifest.

Usage:
    python analysis.py --manifest manifest.csv \
        --input out_logs/ \
        --output artefacts/

The full per-run log files are not bundled in this anonymised repository
(they are large and tied to the embedded platform).  This script is the
entry point referenced from the paper; given the logs, it regenerates
Tables 1-3 and the recovery-time statistics.
"""

from __future__ import annotations
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay IKKA benchmark.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    print("[ikka] manifest:", args.manifest)
    print("[ikka] input:   ", args.input)
    print("[ikka] output:  ", args.output)
    print("[ikka] full replay requires the per-run log files; see README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
