#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "not in a git repo"; exit 1; }
cd "$REPO_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE=".pack.conf"
STAGING_DIR=".packs"

# Windows 用 python，macOS/Linux 用 python3
case "$(uname -s)" in
    MINGW*|CYGWIN*|MSYS*) PYTHON="python" ;;
    *) PYTHON="python3" ;;
esac

usage() {
    cat <<'USAGE'
Usage: pack.sh [options]

Options:
  --full          full archive (all history)
  --branch NAME   specify branch (default: current)
  --no-copy       create only, skip copy step
  -h, --help      show help
USAGE
    exit 0
}

FULL_MODE=false
NO_COPY=false
BRANCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --full) FULL_MODE=true; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --no-copy) NO_COPY=true; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1"; usage ;;
    esac
done

if [ -z "$BRANCH" ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree has uncommitted changes:"
    git status --short
    echo ""
    echo "please commit first."
    exit 1
fi

mkdir -p "$STAGING_DIR"

REPO_NAME=$(basename "$REPO_ROOT")
SHORT_HASH=$(git rev-parse --short HEAD)
TS=$(date +%Y%m%d-%H%M%S)
PKG_NAME="${REPO_NAME}-${BRANCH}-${SHORT_HASH}-${TS}.bundle"
PKG_PATH="${STAGING_DIR}/${PKG_NAME}"

LAST_HASH_FILE="${STAGING_DIR}/.last-pack-hash"
LAST_HASH=$(cat "$LAST_HASH_FILE" 2>/dev/null || echo "")

if [ "$FULL_MODE" = true ] || [ -z "$LAST_HASH" ] || ! git cat-file -e "$LAST_HASH" 2>/dev/null; then
    if [ "$FULL_MODE" = false ] && [ -z "$LAST_HASH" ]; then
        echo "first time, creating full archive..."
    else
        echo "creating full archive..."
    fi
    git bundle create "$PKG_PATH" "$BRANCH"
    PKG_TYPE="full"
else
    NEW_COUNT=$(git rev-list --count "${LAST_HASH}..${BRANCH}")
    if [ "$NEW_COUNT" -eq 0 ]; then
        echo "nothing new since last pack (${LAST_HASH:0:7})"
        exit 0
    fi
    echo "creating incremental archive (${NEW_COUNT} new commits)..."
    git bundle create "$PKG_PATH" "${LAST_HASH}..${BRANCH}"
    PKG_TYPE="incremental"
fi

echo "verifying..."
git bundle verify "$PKG_PATH" >/dev/null 2>&1 || {
    echo "error: verification failed"
    rm -f "$PKG_PATH"
    exit 1
}

PKG_SIZE=$(du -h "$PKG_PATH" | cut -f1)
RANGE_INFO=""
if [ "$PKG_TYPE" = "incremental" ]; then
    RANGE_INFO="${LAST_HASH:0:7}..${SHORT_HASH}"
else
    TOTAL=$(git rev-list --count "$BRANCH")
    RANGE_INFO="${TOTAL} commits total"
fi

echo ""
echo "========================================="
echo "  Pack complete"
echo "========================================="
echo "  file:   ${PKG_NAME}"
echo "  size:   ${PKG_SIZE}"
echo "  type:   ${PKG_TYPE}"
echo "  branch: ${BRANCH}"
echo "  range:  ${RANGE_INFO}"
echo "========================================="

if [ "$NO_COPY" = true ]; then
    git rev-parse HEAD > "$LAST_HASH_FILE"
    echo ""
    echo "path: $(cd "$STAGING_DIR" && pwd)/${PKG_NAME}"
    exit 0
fi

SHARED_PATH=""
if [ -f "$CONF_FILE" ]; then
    source "$CONF_FILE"
fi

STORE_PY="${SCRIPT_DIR}/store.py"
PACK_METHOD="${PACK_METHOD:-auto}"
copied=false

# method 1: shared folder
if [ -n "$SHARED_PATH" ] && [ -d "$SHARED_PATH" ]; then
    cp "$PKG_PATH" "${SHARED_PATH}/"
    echo ""
    echo "copied to: ${SHARED_PATH}/${PKG_NAME}"
    copied=true
fi

# method 2: store.py (if token exists)
if [ "$copied" = false ] || [ "$PACK_METHOD" = "both" ]; then
    if [ -f "${SCRIPT_DIR}/token.json" ] && [ -f "$STORE_PY" ]; then
        echo ""
        echo "putting to remote..."
        $PYTHON "$STORE_PY" put "$PKG_PATH" --folder "${REMOTE_FOLDER:-packs}" && copied=true
    fi
fi

# fallback: manual
if [ "$copied" = false ]; then
    FULL_PATH="$(cd "$STAGING_DIR" && pwd)/${PKG_NAME}"
    echo ""
    echo "output: ${FULL_PATH}"
    echo "(configure SHARED_PATH in .pack.conf or set up store.py)"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$STAGING_DIR"
    fi
fi

if [ "$copied" = true ]; then
    git rev-parse HEAD > "$LAST_HASH_FILE"
fi

echo ""
echo "on the other end, run:"
echo "  bash .claude/skills/local-dev/unpack.sh --file ${PKG_NAME}"
