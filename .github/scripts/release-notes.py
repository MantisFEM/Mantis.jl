#!/usr/bin/env python3
"""Draft release notes from explicitly marked commits on a git range.

Usage: release-notes.py <previous_tag> [<ref>]   (ref defaults to HEAD)

Only commits whose message contains the line

    Release-note: yes

are listed (case-insensitive, anywhere in the message body), together with
all commits authored by Dependabot (`dependabot[bot]`), which cannot carry the
marker. All other commits are ignored. Listed commits are grouped by their
conventional type, using the same sections as TagBot's `conventional`
changelog format; commits of type `doc` are listed under "Documentation".
"""
import re
import subprocess
import sys

MARKER = re.compile(r"^Release-note:\s*yes\s*$", re.I | re.M)
TITLE = re.compile(r"^(\w+)(\([^)]*\))?(!)?: (.+)$")
SECTIONS = [
    ("breaking", "Breaking Changes"),
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("perf", "Performance Improvements"),
    ("refactor", "Code Refactoring"),
    ("docs", "Documentation"),
    ("test", "Tests"),
    ("build", "Build System"),
    ("ci", "CI/CD"),
    ("style", "Code Style"),
    ("chore", "Chores"),
    ("revert", "Reverts"),
    ("other", "Other Changes"),
]
ALIASES = {"doc": "docs"}
DEPENDABOT = "dependabot[bot]"


def commits(rev_range):
    sep, end = "\x1f", "\x1e"
    out = subprocess.run(
        ["git", "log", "--reverse",
         f"--format=%h{sep}%an{sep}%s{sep}%b{end}", rev_range],
        check=True, capture_output=True, text=True,
    ).stdout
    for record in out.split(end):
        record = record.strip("\n")
        if record:
            yield record.split(sep, 3)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rev_range = f"{sys.argv[1]}..{sys.argv[2] if len(sys.argv) > 2 else 'HEAD'}"
    groups = {key: [] for key, _ in SECTIONS}
    for sha, author, title, body in commits(rev_range):
        if not MARKER.search(body) and author != DEPENDABOT:
            continue
        m = TITLE.match(title)
        line = f"- {title} ({sha}, {author})"
        # As in TagBot, a breaking commit is listed both under
        # "Breaking Changes" and under its own type.
        if m and (m.group(3) or re.search(r"^BREAKING[ -]CHANGE:", body, re.M)):
            groups["breaking"].append(line)
        ctype = ALIASES.get(m.group(1).lower(), m.group(1).lower()) if m else None
        if ctype in groups and ctype not in ("breaking", "other"):
            groups[ctype].append(line)
        else:
            groups["other"].append(line)
    blocks = [f"### {heading}\n" + "\n".join(groups[key])
              for key, heading in SECTIONS if groups[key]]
    print("\n\n".join(blocks) if blocks else "No changes marked for the release notes.")


if __name__ == "__main__":
    main()
