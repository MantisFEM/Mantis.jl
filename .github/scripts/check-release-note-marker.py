#!/usr/bin/env python3
"""Reject misspelt release-note markers in a commit message or PR description.

Usage: check-release-note-marker.py <file>

The only accepted marker is a line reading exactly `Release-note: yes`
(case-insensitive, as in release-notes.py). Any other line that looks like an
attempt at it (a `key: value` line whose key is close to "Release-note") is an
error, because release-notes.py would silently ignore it.
"""
import re
import sys

MARKER = re.compile(r"^Release-note:\s*yes\s*$", re.I)
KEY_VALUE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]*?)\s*:\s*(.*)$")
TARGET = "releasenote"


def distance(a, b):
    """Levenshtein edit distance between strings a and b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as f:
        lines = f.read().splitlines()
    bad = []
    for n, line in enumerate(lines, 1):
        if MARKER.match(line.strip()):
            continue
        m = KEY_VALUE.match(line)
        if m and distance(re.sub(r"[^a-z]", "", m.group(1).lower()), TARGET) <= 2:
            bad.append(f"line {n}: {line!r}")
    if bad:
        print("Misspelt release-note marker. The only accepted form is the line:\n"
              "    Release-note: yes\n" + "\n".join(bad), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
