#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)" # Mantis top-level directory.

AI_DIR="$ROOT/.ai"
GITHUB_DIR="$ROOT/.github"
CLAUDE_DIR="$ROOT/.claude"

echo "Syncing AI configurations... "


############################################################################################
#                                     Main instructions                                    #
############################################################################################

mkdir -p "$GITHUB_DIR"
mkdir -p "$CLAUDE_DIR"

# Create read-only copies
install -m 644 "$AI_DIR/instructions.md" "$GITHUB_DIR/copilot-instructions.md"
install -m 644 "$AI_DIR/instructions.md" "$ROOT/CLAUDE.md"

############################################################################################
#                                         Skills                                           #
############################################################################################

# Copy skills
mkdir -p "$GITHUB_DIR/skills"
mkdir -p "$CLAUDE_DIR/skills"
for skill in "$AI_DIR"/skills/*.md; do
    name="$(basename "$skill" .md)"
    for agent in "$GITHUB_DIR" "$CLAUDE_DIR"; do
        mkdir -p "$agent/skills/$name"
        install -m 644 "$skill" "$agent/skills/$name/SKILL.md"
    done
done

# Remove stale skills
for dir in "$GITHUB_DIR" "$CLAUDE_DIR"; do
    for skilldir in "$dir"/skills/*; do
        [ -d "$skilldir" ] || continue

        name="$(basename "$skilldir")"

        if [[ ! -f "$AI_DIR/skills/$name.md" ]]; then
            rm -r "$skilldir"
        fi
    done
done

echo "Done!"
