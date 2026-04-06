#!/bin/bash
# ClawTheSpire bot launcher
#
# Usage:
#   bash play.sh                        # single game (interactive TUI, profile A)
#   bash play.sh --profile b            # single game using challenger config
#   bash play.sh --step                 # step through each action
#   bash play.sh batch                  # continuous games (loops forever)
#   bash play.sh batch --once           # one game via batch runner
#   bash play.sh batch --profile b      # batch using challenger config
#
# The --profile flag sets which config to use:
#   a = champion (default, known-good baseline)
#   b = challenger (experimental changes)

if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
export PATH="$HOME/.local/bin:$PATH"

export STS2_API_BASE_URL=http://127.0.0.1:8080

# ---------------------------------------------------------------------------
# Parse --profile flag before passing remaining args to Python.
# This sets the STS2_CONFIG_PROFILE env var that config.py reads.
# ---------------------------------------------------------------------------
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--profile" ]]; then
        # Next argument is the profile letter — grab it on the next iteration
        GRAB_PROFILE=1
        continue
    fi
    if [[ "$GRAB_PROFILE" == "1" ]]; then
        export STS2_CONFIG_PROFILE="$arg"
        GRAB_PROFILE=0
        continue
    fi
    ARGS+=("$arg")
done

# Show which profile is active so there's no confusion in the terminal
if [[ -n "$STS2_CONFIG_PROFILE" ]]; then
    echo ">> Config profile: $STS2_CONFIG_PROFILE ($([ "$STS2_CONFIG_PROFILE" == "a" ] && echo 'champion' || echo 'challenger'))"
else
    echo ">> Config profile: a (champion, default)"
fi

cd "$(dirname "$0")/sts2-solver"

if [[ "${ARGS[0]}" == "batch" ]]; then
    # Remove "batch" from the front, pass the rest to batch_runner
    exec uv run python -m sts2_solver.batch_runner "${ARGS[@]:1}"
else
    exec uv run python ../run.py "${ARGS[@]}"
fi
