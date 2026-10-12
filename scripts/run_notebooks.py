"""
Execute the tutorial notebooks in order, each in a fresh kernel.

    python scripts/run_notebooks.py            # check that they run
    python scripts/run_notebooks.py --write    # also save the new outputs

The archived notebooks in notebooks/archive/ are not run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="save the executed notebooks back to disk")
    args = parser.parse_args()

    failed = []
    for path in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        start = time.time()
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        try:
            client.execute()
            status = "ok"
        except Exception as exc:  # report every notebook, then fail at the end
            status = f"FAILED ({type(exc).__name__})"
            failed.append(path.name)
            print(f"{path.name}: {exc}", file=sys.stderr, flush=True)
        print(f"{path.name}: {status} in {time.time() - start:.0f} s", flush=True)
        if args.write and status == "ok":
            nbformat.write(notebook, path)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
