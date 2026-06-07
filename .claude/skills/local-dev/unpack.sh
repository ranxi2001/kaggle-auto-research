#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "not in a git repo"; exit 1; }
cd "$REPO_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE=".pack.conf"
STAGING_DIR=".packs"
REPO_NAME="$(basename "$REPO_ROOT")"

# Windows 用 python，macOS/Linux 用 python3
case "$(uname -s)" in
    MINGW*|CYGWIN*|MSYS*) PYTHON="python" ;;
    *) PYTHON="python3" ;;
esac

usage() {
    cat <<'USAGE'
Usage: unpack.sh [options]

Options:
  --file FILE     specify bundle file path
  --list          list available bundles only
  --no-merge      fetch only, skip merge
  -h, --help      show help
USAGE
    exit 0
}

PKG_FILE=""
LIST_ONLY=false
NO_MERGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --file) PKG_FILE="$2"; shift 2 ;;
        --list) LIST_ONLY=true; shift ;;
        --no-merge) NO_MERGE=true; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1"; usage ;;
    esac
done

mkdir -p "$STAGING_DIR"

SHARED_PATH=""
if [ -f "$CONF_FILE" ]; then
    source "$CONF_FILE"
fi

STORE_PY="${SCRIPT_DIR}/store.py"
REMOTE_FOLDER="${REMOTE_FOLDER:-packs}"

is_repo_bundle_name() {
    local name
    name="$(basename "$1")"
    case "$name" in
        "${REPO_NAME}"-*.bundle) return 0 ;;
        *) return 1 ;;
    esac
}

remote_ls() {
    local err_file
    err_file="$(mktemp)"
    if "$PYTHON" "$STORE_PY" ls --folder "$REMOTE_FOLDER" 2>"$err_file"; then
        rm -f "$err_file"
        return 0
    fi

    echo "warning: failed to list remote bundles." >&2
    tail -n 5 "$err_file" | sed 's/^/  /' >&2
    rm -f "$err_file"
    return 1
}

list_available() {
    if [ -n "$SHARED_PATH" ] && [ -d "$SHARED_PATH" ]; then
        echo "available in shared folder:"
        ls -lt "${SHARED_PATH}/${REPO_NAME}-"*.bundle 2>/dev/null || echo "  (none)"
    fi
    if [ -f "${SCRIPT_DIR}/token.json" ] && [ -f "$STORE_PY" ]; then
        echo ""
        echo "available on remote:"
        remote_found=false
        remote_list_failed=false
        remote_output=""
        if remote_output="$(remote_ls)"; then
            while IFS= read -r name; do
                [ -z "$name" ] && continue
                if is_repo_bundle_name "$name"; then
                    echo "  $name"
                    remote_found=true
                fi
            done < <(printf '%s\n' "$remote_output" | awk '{print $NF}')
        else
            remote_list_failed=true
        fi
        if [ "$remote_found" = true ]; then
            :
        elif [ "$remote_list_failed" = true ]; then
            echo "  (remote listing failed)"
        else
            echo "  (none)"
        fi
    fi
    echo ""
    echo "available locally:"
    ls -lt "${STAGING_DIR}/${REPO_NAME}-"*.bundle 2>/dev/null || echo "  (none)"
}

collect_remote_found() {
    local remote_output
    remote_output=""
    if ! remote_output="$(remote_ls)"; then
        return 1
    fi

    while IFS= read -r name; do
        [ -z "$name" ] && continue
        if is_repo_bundle_name "$name"; then
            REMOTE_FOUND+=("$name")
        fi
    done < <(printf '%s\n' "$remote_output" | awk '{print $NF}')
}

STASHED=false

stash_local_changes() {
    if [ -n "$(git diff --name-only HEAD 2>/dev/null)" ] || [ -n "$(git diff --cached --name-only 2>/dev/null)" ]; then
        echo "stashing local changes..."
        git stash push -m "unpack-auto-stash $(date +%Y%m%d-%H%M%S)"
        STASHED=true
        echo "stashed OK."
        echo ""
    fi
}

unstash_local_changes() {
    if [ "$STASHED" = true ]; then
        echo ""
        echo "restoring stashed changes..."
        if git stash pop; then
            STASHED=false
            echo "stash restored OK."
        else
            echo "warning: stash pop had conflicts. your changes are in 'git stash list'."
            echo "  resolve conflicts, then run: git stash drop"
            return 1
        fi
    fi
}

if [ "$LIST_ONLY" = true ]; then
    list_available
    exit 0
fi

apply_bundle() {
    local bundle="$1"
    local branch
    branch=$(git rev-parse --abbrev-ref HEAD)
    local fetch_ref
    fetch_ref=$(git bundle list-heads "$bundle" 2>/dev/null | grep "refs/heads/" | head -1 | awk '{print $2}' | sed 's|refs/heads/||')
    [ -z "$fetch_ref" ] && fetch_ref="HEAD"

    git branch -D pack-incoming 2>/dev/null || true
    echo "fetching from bundle..."
    git fetch "$bundle" "${fetch_ref}:pack-incoming"

    echo ""
    echo "incoming commits:"
    git log --oneline "${branch}..pack-incoming" 2>/dev/null || echo "(cannot diff)"
    echo ""

    local incoming
    incoming=$(git rev-list --count "${branch}..pack-incoming" 2>/dev/null || echo "?")
    echo "${incoming} new commit(s)."
    echo ""

    echo "merging..."
    if git merge pack-incoming --ff-only 2>/dev/null; then
        echo "fast-forward merge done."
    elif git merge pack-incoming -m "merge from bundle ($(basename "$bundle"))"; then
        echo "merge done (created merge commit)."
    else
        echo ""
        echo "conflict detected. resolve manually, then run:"
        echo "  git add ."
        echo "  git commit"
        echo "  git branch -d pack-incoming"
        return 1
    fi

    git branch -d pack-incoming 2>/dev/null || true
    git rev-parse HEAD > "${STAGING_DIR}/.last-pack-hash"
}

if [ -n "$PKG_FILE" ]; then
    # single-file mode
    if [ ! -f "$PKG_FILE" ]; then
        echo "error: file not found: ${PKG_FILE}"
        exit 1
    fi
    echo ""
    echo "verifying..."
    if ! git bundle verify "$PKG_FILE" 2>&1; then
        echo "verification failed."
        exit 1
    fi
    echo "verified OK."
    echo ""
    echo "refs in bundle:"
    git bundle list-heads "$PKG_FILE"
    echo ""
    if [ "$NO_MERGE" = true ]; then
        echo "skipping merge (--no-merge)."
        exit 0
    fi
    stash_local_changes
    if ! apply_bundle "$PKG_FILE"; then
        unstash_local_changes || true
        exit 1
    fi
    MERGED_COUNT=1
    LAST_PKG="$PKG_FILE"
    unstash_local_changes
else
    # auto-chain mode: collect all candidates, apply in order
    echo "looking for bundles..."

    # gather remote bundles (oldest-first) into staging
    REMOTE_BUNDLES=()
    REMOTE_DOWNLOAD_ERRORS=()
    REMOTE_LIST_FAILED=false
    if [ -f "${SCRIPT_DIR}/token.json" ] && [ -f "$STORE_PY" ]; then
        echo "checking remote..."
        REMOTE_FOUND=()
        collect_remote_found || REMOTE_LIST_FAILED=true
        if [ "${#REMOTE_FOUND[@]}" -gt 0 ]; then
            while IFS= read -r name; do
                [ -z "$name" ] && continue
                REMOTE_BUNDLES+=("$name")
            done < <(printf '%s\n' "${REMOTE_FOUND[@]}" | awk '{a[NR]=$0} END{for(i=NR;i>=1;i--) print a[i]}')
        fi
    fi

    # gather shared-folder bundles
    SHARED_BUNDLES=()
    if [ -n "$SHARED_PATH" ] && [ -d "$SHARED_PATH" ]; then
        while IFS= read -r f; do
            SHARED_BUNDLES+=("$f")
        done < <(ls -tr "${SHARED_PATH}/${REPO_NAME}-"*.bundle 2>/dev/null || true)
    fi

    MERGED_COUNT=0
    LAST_PKG=""

    stash_local_changes

    # download all remote bundles to staging first
    if [ -f "${SCRIPT_DIR}/token.json" ] && [ -f "$STORE_PY" ]; then
        for name in "${REMOTE_BUNDLES[@]:-}"; do
            [ -z "$name" ] && continue
            local_path="${STAGING_DIR}/${name}"
            if [ -f "$local_path" ]; then
                continue
            fi
            if ! output=$($PYTHON "$STORE_PY" get "$name" --folder "$REMOTE_FOLDER" --dest "$STAGING_DIR" 2>&1); then
                echo "warning: failed to download remote bundle: $name"
                echo "$output" | tail -n 3 | sed 's/^/  /'
                REMOTE_DOWNLOAD_ERRORS+=("$name")
            fi
        done
    fi
    for f in "${SHARED_BUNDLES[@]:-}"; do
        [ -z "$f" ] && continue
        dest="${STAGING_DIR}/$(basename "$f")"
        [ -f "$dest" ] || cp "$f" "$dest"
    done

    # apply bundles oldest-first, skip already-merged, stop when nothing new
    while true; do
        APPLIED_THIS_ROUND=false
        while IFS= read -r candidate; do
            [ -z "$candidate" ] || [ ! -f "$candidate" ] && continue
            # skip if bundle's target commit is already in our history
            bundle_commit=$(git bundle list-heads "$candidate" 2>/dev/null | grep "refs/heads/" | head -1 | awk '{print $1}')
            if [ -n "$bundle_commit" ] && git merge-base --is-ancestor "$bundle_commit" HEAD 2>/dev/null; then
                continue  # already have it
            fi
            if git bundle verify "$candidate" >/dev/null 2>&1; then
                echo ""
                echo "applying: $(basename "$candidate")"
                if apply_bundle "$candidate"; then
                    LAST_PKG="$candidate"
                    MERGED_COUNT=$((MERGED_COUNT + 1))
                    APPLIED_THIS_ROUND=true
                    break  # restart scan after HEAD moved
                else
                    unstash_local_changes || true
                    exit 1
                fi
            fi
        done < <(ls -tr "${STAGING_DIR}/${REPO_NAME}-"*.bundle 2>/dev/null || true)
        [ "$APPLIED_THIS_ROUND" = false ] && break
    done

    if [ "$MERGED_COUNT" -eq 0 ]; then
        echo "no applicable bundles found."
        if [ "${#REMOTE_DOWNLOAD_ERRORS[@]}" -gt 0 ]; then
            echo "remote download failed for:"
            printf '  %s\n' "${REMOTE_DOWNLOAD_ERRORS[@]}"
            echo "re-run after fixing remote connectivity, or use --file to specify a downloaded bundle."
        elif [ "$REMOTE_LIST_FAILED" = true ]; then
            echo "remote listing failed; re-run after fixing remote connectivity, or use --file to specify a downloaded bundle."
        else
            echo "use --file to specify path, or set up store.py / shared folder."
        fi
        unstash_local_changes || true
        exit 1
    fi
    unstash_local_changes
fi

echo ""
echo "========================================="
echo "  Unpack complete"
echo "========================================="
echo "  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  HEAD:   $(git rev-parse --short HEAD)"
[ -n "$LAST_PKG" ] && echo "  last:   $(basename "$LAST_PKG")"
[ "${MERGED_COUNT:-1}" -gt 1 ] && echo "  merged: ${MERGED_COUNT} bundles"
echo "========================================="
