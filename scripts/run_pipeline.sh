#!/usr/bin/env bash
set -euo pipefail

IMAGES_DIR="${1:-}"
OUT_DIR="${2:-}"
CONFIG="${3:-}"

if [[ -z "${IMAGES_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: $0 <images_dir> <out_dir> [config.yml]" >&2
  exit 1
fi

if [[ -n "${CONFIG}" ]]; then
  ./build/psynth_pipeline --images_dir "${IMAGES_DIR}" --out_dir "${OUT_DIR}" --config "${CONFIG}"
else
  ./build/psynth_pipeline --images_dir "${IMAGES_DIR}" --out_dir "${OUT_DIR}"
fi