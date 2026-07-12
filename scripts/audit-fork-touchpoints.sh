#!/usr/bin/env bash
# audit-fork-touchpoints.sh
#
# Walks fork commits on top of the upstream fork-point and aggregates
# per-file fork LOC. Emits:
#   - <out>/fork-file-inventory.tsv       : all files touched by fork,
#     with added/removed LOC and whether the file existed at FORK_POINT.
#   - <out>/upstream-file-migration-queue.tsv : subset limited to files
#     that existed at FORK_POINT (i.e. upstream files), sorted by added
#     LOC descending.
#
# Usage:
#   FORK_POINT=<sha> ./scripts/audit-fork-touchpoints.sh [output-dir]

set -euo pipefail

if [[ -z "${FORK_POINT:-}" ]]; then
  echo "ERROR: FORK_POINT environment variable must be set." >&2
  exit 1
fi

OUT_DIR="${1:-audit-output}"
mkdir -p "$OUT_DIR"

INVENTORY="$OUT_DIR/fork-file-inventory.tsv"
QUEUE="$OUT_DIR/upstream-file-migration-queue.tsv"
TMP="$OUT_DIR/.inventory.tmp"

echo -e "file\tadded\tremoved\tupstream_at_fork_point" > "$INVENTORY"

# Aggregate per-file add/remove across all fork commits.
git log --format="%H" "${FORK_POINT}..HEAD" | while read -r SHA; do
  git show --numstat --format="" "$SHA"
done | awk -F'\t' '
  NF == 3 && $1 != "-" && $2 != "-" {
    added[$3]   += $1
    removed[$3] += $2
  }
  END {
    for (f in added) printf "%s\t%d\t%d\n", f, added[f], removed[f]
  }
' | sort -t $'\t' -k2,2 -nr > "$TMP"

# Mark upstream-at-fork-point vs fork-introduced.
while IFS=$'\t' read -r file added removed; do
  if git cat-file -e "${FORK_POINT}:${file}" 2>/dev/null; then
    upstream="yes"
  else
    upstream="no"
  fi
  printf "%s\t%s\t%s\t%s\n" "$file" "$added" "$removed" "$upstream"
done < "$TMP" >> "$INVENTORY"

rm "$TMP"

# Emit upstream-only subset sorted by added desc.
{
  head -n 1 "$INVENTORY"
  tail -n +2 "$INVENTORY" | awk -F'\t' '$4 == "yes"' | sort -t $'\t' -k2,2 -nr
} > "$QUEUE"

echo "Audit output:"
echo "  $INVENTORY"
echo "  $QUEUE"
