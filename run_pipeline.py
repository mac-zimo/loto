#!/usr/bin/env python3
"""Backward-compatible wrapper for the package CLI."""

import sys
from pathlib import Path

# Keep this legacy script directly runnable from a src-layout checkout, even
# before the project has been installed. Package/module entry points are
# intentionally unaffected.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from loto.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
