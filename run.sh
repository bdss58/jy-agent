#!/bin/bash
set -e

# Capture the directory where the user launched from
export LAUNCH_DIR="$(pwd)"

# Get the project directory (resolve symlinks)
SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
    SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
    [[ $SCRIPT_PATH != /* ]] && SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH"
done
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run the agent
python -m jyagent "$@"
