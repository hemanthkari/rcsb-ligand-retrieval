"""
Build Colab notebooks from markdown cell definitions.

Parses "## Cell N -- ..." headers and fenced python code blocks from each
markdown source in scripts/notebook_sources/ and emits the matching
Jupyter notebook under notebooks/.

Usage:
    python scripts/build_notebook.py                    # build every notebook
    python scripts/build_notebook.py --name 02_train_adapter
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SRC_DIR = Path("scripts/notebook_sources")
DST_DIR = Path("notebooks")


def discover_notebooks() -> list:
    """Return sorted list of (stem, src_path, dst_path)."""
    out = []
    for md in sorted(SRC_DIR.glob("*.md")):
        stem = md.stem
        out.append((stem, md, DST_DIR / (stem + ".ipynb")))
    return out


def parse_cells(md_text: str) -> list[tuple[str, str, str]]:
    """Return list of (cell_type, header_markdown, body_source) tuples."""
    lines = md_text.splitlines()
    cells: list[tuple[str, str, str]] = []

    i = 0
    current_header_md: list[str] = []
    in_code_block = False
    code_lines: list[str] = []
    pending_cell_title = ""

    def flush_header():
        nonlocal current_header_md
        if any(ln.strip() for ln in current_header_md):
            cells.append(("markdown", "", "\n".join(current_header_md).rstrip() + "\n"))
        current_header_md = []

    while i < len(lines):
        line = lines[i]

        if line.strip() == "---":
            i += 1
            continue

        if line.startswith("## Cell "):
            flush_header()
            pending_cell_title = line.strip()
            current_header_md.append(line)
            i += 1
            continue

        m = re.match(r"^```(\w+)?\s*$", line)
        if m and not in_code_block:
            in_code_block = True
            code_lines = []
            i += 1
            continue
        if line.strip() == "```" and in_code_block:
            in_code_block = False
            if current_header_md:
                header_text = "\n".join(current_header_md).rstrip() + "\n"
                cells.append(("markdown", pending_cell_title, header_text))
                current_header_md = []
            body = "\n".join(code_lines).rstrip() + "\n"
            cells.append(("code", pending_cell_title, body))
            code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
        else:
            current_header_md.append(line)
        i += 1

    flush_header()
    return cells


def to_notebook(cells: list[tuple[str, str, str]]) -> dict:
    nb_cells = []
    for cell_type, _header, src in cells:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": _split_source(src),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _split_source(src: str) -> list[str]:
    if not src:
        return []
    return src.splitlines(keepends=True)


def build_one(src: Path, dst: Path) -> None:
    if not src.exists():
        print("  [SKIP] %s not found" % src)
        return
    md = src.read_text(encoding="utf-8")
    cells = parse_cells(md)
    notebook = to_notebook(cells)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    print("  Wrote %s: %d code cells, %d markdown cells" % (dst, n_code, n_md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Build only this notebook (stem, e.g. '02_train_adapter').",
    )
    args = parser.parse_args()

    pairs = discover_notebooks()
    if not pairs:
        raise SystemExit("No markdown sources under %s" % SRC_DIR)

    if args.name:
        wanted = args.name.replace(".md", "").replace(".ipynb", "")
        matched = [(s, src, dst) for s, src, dst in pairs if s == wanted]
        if not matched:
            names = ", ".join(s for s, _, _ in pairs)
            raise SystemExit("Unknown notebook: %s. Available: %s" % (wanted, names))
        for _, src, dst in matched:
            build_one(src, dst)
    else:
        print("Building all notebooks:")
        for _, src, dst in pairs:
            build_one(src, dst)

    print("Done.")


if __name__ == "__main__":
    main()
