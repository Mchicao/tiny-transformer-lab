#!/bin/sh
set -eu

root="$HOME/.local/share/tiny-transformer-lab/colab-cli"
umask 077
export HOME="$root/home"
mkdir -p "$HOME"
exec "$root/.venv/bin/colab" "$@"
