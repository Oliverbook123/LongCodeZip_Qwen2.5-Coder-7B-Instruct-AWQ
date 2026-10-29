#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
session_name="lcz-staged"
if tmux has-session -t "=$session_name" 2>/dev/null; then
    echo "Session $session_name already exists. Open it with: tmux attach -t $session_name"
    exit 1
fi
tmux new-session -d -s "$session_name" -c "$repo_root" bash experiments/long-code-completion/run_staged_awq.sh "$@"
echo "Started $session_name. Open it with: tmux attach -t $session_name"
