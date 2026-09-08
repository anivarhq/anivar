#!/usr/bin/env bash
#
# Print one version's section from CHANGELOG.md, for use as a release body.
#
#   changelog-section.sh 0.1.0 [CHANGELOG.md]
#   changelog-section.sh v0.1.0            # a leading v is tolerated
#
# Exits 1 when the version has no section, so the caller can fall back to
# something rather than publishing an empty release body.
#
# Headings are matched LITERALLY, not by regex: a version is full of dots, and
# `1.2.3` as a pattern also matches `1x2x3`. Accepts both Keep a Changelog's
# `## [1.2.3] - 2026-09-08` and a bare `## 1.2.3`.
set -euo pipefail

version="${1:?usage: changelog-section.sh <version> [file]}"
version="${version#v}"
file="${2:-CHANGELOG.md}"

[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

section=$(awk -v ver="$version" '
  BEGIN { bracketed = "## [" ver "]"; bare = "## " ver }
  !inside && (index($0, bracketed) == 1 || index($0, bare) == 1) { inside = 1; next }
  inside && index($0, "## ") == 1 { exit }
  inside { print }
' "$file")

# Trim leading and trailing blank lines; keep the ones in the middle.
section=$(printf '%s\n' "$section" | sed -e '/./,$!d' | sed -e ':a' -e '/^\n*$/{$d;N;ba' -e '}')

[ -n "$section" ] || { echo "no section for version $version in $file" >&2; exit 1; }
printf '%s\n' "$section"
