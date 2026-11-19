#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exe=$(command -v trtexec || true)
exe=${exe:-/usr/src/tensorrt/bin/trtexec}
[[ -x "$exe" ]] || { echo 'Installer trtexec correspondant à JetPack'; exit 2; }
mkdir -p engines logs
for name in b1 efficientnet deeplab "$@"; do
  "$exe" --onnx="$name.onnx" --fp16 --memPoolSize=workspace:1024 --saveEngine="engines/$name.plan" --warmUp=1000 --duration=15 2>&1 | tee "logs/$name.log"
done