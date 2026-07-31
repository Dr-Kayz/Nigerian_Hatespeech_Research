#!/usr/bin/env bash
# Build reports/REPORT.docx from reports/REPORT.md using pandoc.
#
# Requires:
#   brew install pandoc
#
# Usage:
#   bash scripts/build_report_docx.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/reports/REPORT.md"
OUT="$ROOT/reports/REPORT.docx"

if [ ! -f "$SRC" ]; then
    echo "Source not found: $SRC" >&2
    exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
    echo "pandoc not installed. Install with: brew install pandoc" >&2
    exit 1
fi

echo "Building $OUT from $SRC ..."

pandoc "$SRC" \
    --from=gfm \
    --to=docx \
    --output="$OUT" \
    --resource-path="$ROOT" \
    --toc \
    --toc-depth=3 \
    --standalone \
    --metadata title="Cross-Lingual Hate Speech Detection in Nigerian Languages"

echo "Done."
ls -la "$OUT"
