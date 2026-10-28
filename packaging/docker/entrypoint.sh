#!/usr/bin/env bash
# Fetch the RAM++ checkpoint into the mounted volume on first start, then run the
# service. The download is checked against the hash pinned in this repository.
set -euo pipefail

case "${USE_VISUAL_TAGS:-}" in
  1 | true | TRUE | True | yes | on)
    if [[ ! -f "$RAM_CHECKPOINT_FILE" ]]; then
      mkdir -p "$(dirname "$RAM_CHECKPOINT_FILE")"
      echo "Downloading the RAM++ checkpoint (3 GB) to $RAM_CHECKPOINT_FILE" >&2
      curl -fL -o "$RAM_CHECKPOINT_FILE.part" \
        "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/84d4aee3a0265c4e0df1f714f0572011d1bf2ec3/ram_plus_swin_large_14m.pth"
      echo "$(cat /app/packaging/ram/checkpoint.sha256)  $RAM_CHECKPOINT_FILE.part" | sha256sum -c -
      mv "$RAM_CHECKPOINT_FILE.part" "$RAM_CHECKPOINT_FILE"
    fi
    ;;
esac

exec "$@"
