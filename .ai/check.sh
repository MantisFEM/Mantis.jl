#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)" # Mantis top-level directory.

AI_DIR="$ROOT/.ai"
GITHUB_DIR="$ROOT/.github"
CLAUDE_DIR="$ROOT/.claude"

echo "Checking AI configurations... "


############################################################################################
#                                     Main instructions                                    #
############################################################################################

diff "$AI_DIR/instructions.md" "$GITHUB_DIR/copilot-instructions.md"
diff "$AI_DIR/instructions.md" "$ROOT/CLAUDE.md"

############################################################################################
#                                         Skills                                           #
############################################################################################

# Check modified skills
for skill in "$AI_DIR"/skills/*.md; do
    name="$(basename "$skill" .md)"
    for agent in "$GITHUB_DIR" "$CLAUDE_DIR"; do
        diff "$skill" "$agent/skills/$name/SKILL.md"
    done
done

# Check for stale skills
for dir in "$GITHUB_DIR" "$CLAUDE_DIR"; do
    for skilldir in "$dir"/skills/*; do
        [ -d "$skilldir" ] || continue

        name="$(basename "$skilldir")"

        if [[ ! -f "$AI_DIR/skills/$name.md" ]]; then
            echo "The skill \`$skilldir\` does not exist in \`.ai/skills\`."
            echo "Please remove it."
            exit 1
        fi
    done
done

echo "Done!"
