#!/usr/bin/env bash
#
# Build the tiny-count model and register it with Ollama.
#
# Usage:
#   ./build.sh                 # setup + train + convert + ollama create
#   ./build.sh convert model   # reuse the existing HF model
#   ./build.sh run             # chat with the registered model
#
# See ./build.sh --help for all stages and environment variables.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-}"
HF_MODEL_DIR="${HF_MODEL_DIR:-$ROOT_DIR/tiny-count-model}"
GGUF_PATH="${GGUF_PATH:-$ROOT_DIR/tiny-count.gguf}"
MODEL_NAME="${MODEL_NAME:-tiny-count}"
OUTTYPE="${OUTTYPE:-f16}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTHON="${PYTHON:-python3}"
PROMPT="${PROMPT:-count to 5}"

PY=""
OLLAMA_STARTED_PID=""
MODEFILE_TMP=""

log() {
    printf '==> %s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "$OLLAMA_STARTED_PID" ]]; then
        kill "$OLLAMA_STARTED_PID" 2>/dev/null || true
    fi
    if [[ -n "$MODEFILE_TMP" ]]; then
        rm -f "$MODEFILE_TMP"
    fi
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
Usage: ./build.sh [stage ...]

Stages:
  setup    Clone llama.cpp and create .venv with the Python dependencies
  train    Train the tiny count model into tiny-count-model/
  convert  Convert the HF model into tiny-count.gguf
  model    Register the GGUF with Ollama
  run      Run the registered model with a prompt
  all      setup + train + convert + model (default)

Environment:
  MODEL_NAME       Ollama model name                 (default: tiny-count)
  PROMPT           Prompt used by the run stage      (default: count to 5)
  OUTTYPE          GGUF output type                  (default: f16)
  PYTHON_VERSION   Python version for the virtualenv (default: 3.12)
  VENV_DIR         Virtualenv path                   (default: .venv)
  LLAMA_CPP_DIR    llama.cpp checkout                (default: ./llama.cpp)
  LLAMA_CPP_REF    Optional llama.cpp git ref to clone
  HF_MODEL_DIR     Trained HF model directory        (default: tiny-count-model)
  GGUF_PATH        Output GGUF path                  (default: tiny-count.gguf)

Examples:
  ./build.sh                         # full build
  ./build.sh train                   # only retrain
  ./build.sh convert model           # re-convert and re-register
  MODEL_NAME=tiny-count-dev ./build.sh model
  PROMPT="count to 3" ./build.sh run
EOF
}

hash_stdin() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 | awk '{print $1}'
    else
        sha256sum | awk '{print $1}'
    fi
}

ensure_llama_cpp() {
    if [[ -d "$LLAMA_CPP_DIR" ]]; then
        [[ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]] \
            || die "$LLAMA_CPP_DIR exists but is not a llama.cpp checkout"
        return
    fi

    command -v git >/dev/null 2>&1 || die "git is required to clone llama.cpp"
    log "cloning llama.cpp into ${LLAMA_CPP_DIR#$ROOT_DIR/}"
    if [[ -n "$LLAMA_CPP_REF" ]]; then
        git clone --depth 1 --branch "$LLAMA_CPP_REF" "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
    else
        git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
    fi
}

ensure_python() {
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        log "creating virtualenv at ${VENV_DIR#$ROOT_DIR/}"
        if command -v uv >/dev/null 2>&1; then
            uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
        else
            command -v "$PYTHON" >/dev/null 2>&1 || die "python not found: $PYTHON"
            "$PYTHON" -m venv "$VENV_DIR"
        fi
    fi
    PY="$VENV_DIR/bin/python"
    [[ -x "$PY" ]] || die "failed to create virtualenv at $VENV_DIR"

    # llama.cpp owns the conversion dependencies; using its requirements keeps
    # the converter and the training script on the same compatible versions.
    local requirements="$ROOT_DIR/requirements.txt"
    local llama_requirements="$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"
    local legacy_requirements="$LLAMA_CPP_DIR/requirements/requirements-convert_legacy_llama.txt"
    [[ -f "$requirements" && -f "$llama_requirements" && -f "$legacy_requirements" ]] \
        || die "llama.cpp requirements not found under $LLAMA_CPP_DIR/requirements"

    local deps_hash stamp
    deps_hash="$(cat "$requirements" "$llama_requirements" "$legacy_requirements" | hash_stdin)"
    stamp="$VENV_DIR/.deps-$deps_hash"

    if [[ -f "$stamp" ]]; then
        return
    fi

    log "installing Python dependencies"
    if command -v uv >/dev/null 2>&1; then
        if ! uv pip install --python "$PY" -r "$requirements"; then
            log "uv install failed; retrying with pip"
            "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
            "$PY" -m pip install --upgrade pip
            "$PY" -m pip install -r "$requirements"
        fi
    else
        "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
        "$PY" -m pip install --upgrade pip
        "$PY" -m pip install -r "$requirements"
    fi

    rm -f "$VENV_DIR"/.deps-*
    touch "$stamp"
}

ensure_ollama_running() {
    command -v ollama >/dev/null 2>&1 || die "ollama is not installed"

    if ollama list >/dev/null 2>&1; then
        return
    fi

    log "starting Ollama server"
    ollama serve >/dev/null 2>&1 &
    OLLAMA_STARTED_PID=$!

    local attempt
    for attempt in $(seq 1 60); do
        if ollama list >/dev/null 2>&1; then
            return
        fi
        if ! kill -0 "$OLLAMA_STARTED_PID" 2>/dev/null; then
            die "ollama serve exited before becoming ready"
        fi
        sleep 0.5
    done
    die "timed out waiting for the Ollama server"
}

stage_setup() {
    ensure_llama_cpp
    ensure_python
    log "environment ready"
}

stage_train() {
    ensure_llama_cpp
    ensure_python
    log "training tiny-count model"
    "$PY" "$ROOT_DIR/train_tiny_count.py"
}

stage_convert() {
    ensure_llama_cpp
    ensure_python
    [[ -f "$HF_MODEL_DIR/config.json" ]] \
        || die "missing HF model at $HF_MODEL_DIR; run ./build.sh train first"

    log "converting ${HF_MODEL_DIR#$ROOT_DIR/} to ${GGUF_PATH#$ROOT_DIR/} ($OUTTYPE)"
    local tmp="${GGUF_PATH}.tmp.gguf"
    rm -f "$tmp"
    PYTHONPATH="$LLAMA_CPP_DIR/gguf-py:$LLAMA_CPP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" "$ROOT_DIR/convert_hf_to_gguf.py" "$HF_MODEL_DIR" \
        --outfile "$tmp" --outtype "$OUTTYPE"
    [[ -f "$tmp" ]] || die "conversion did not produce $tmp"
    mv -f "$tmp" "$GGUF_PATH"
    log "wrote ${GGUF_PATH#$ROOT_DIR/}"
}

stage_model() {
    [[ -f "$GGUF_PATH" ]] || die "missing $GGUF_PATH; run ./build.sh convert first"
    ensure_ollama_running

    # Keep the tracked Modelfile in sync with GGUF_PATH without editing it.
    MODEFILE_TMP="$(mktemp "${TMPDIR:-/tmp}/tiny-count-Modelfile.XXXXXX")"
    {
        printf 'FROM %s\n' "$GGUF_PATH"
        tail -n +2 "$ROOT_DIR/Modelfile"
    } > "$MODEFILE_TMP"

    log "registering Ollama model '$MODEL_NAME'"
    ollama create "$MODEL_NAME" -f "$MODEFILE_TMP"
    rm -f "$MODEFILE_TMP"
    MODEFILE_TMP=""
    log "created model '$MODEL_NAME'"
}

stage_run() {
    ensure_ollama_running
    ollama run "$MODEL_NAME" "$PROMPT"
}

stage_all() {
    stage_setup
    stage_train
    stage_convert
    stage_model
}

if [[ $# -eq 0 ]]; then
    set -- all
fi

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
        setup|train|convert|model|run|all)
            ;;
        *)
            usage >&2
            die "unknown stage: $arg"
            ;;
    esac
done

for stage in "$@"; do
    "stage_$stage"
done
