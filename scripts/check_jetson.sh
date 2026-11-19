#!/usr/bin/env bash
# Read-only check of what's installed on the Jetson for the ensemble test.
# Usage: bash check_jetson.sh
echo "== JetPack / L4T ==";   cat /etc/nv_tegra_release; dpkg -l nvidia-jetpack 2>/dev/null | tail -1
echo "== Power mode ==";      nvpmodel -q 2>/dev/null
echo "== Disk / RAM ==";      df -h ~ | tail -1; free -h | head -2
echo "== git ==";             git --version || echo "MISSING git"
echo "== git-lfs ==";         git lfs version || echo "MISSING git-lfs"
echo "== trtexec ==";         command -v trtexec || ls /usr/src/tensorrt/bin/trtexec 2>/dev/null || echo "MISSING trtexec"
echo "== tegrastats ==";      command -v tegrastats || echo "MISSING tegrastats"
echo "== CUDA ==";            nvcc --version 2>/dev/null | tail -1 || ls -d /usr/local/cuda* 2>/dev/null || echo "MISSING CUDA toolkit"
echo "== Python ==";          which python3; python3 --version
for m in tensorrt torch torchvision numpy PIL; do
  python3 -c "import $m; print('$m', getattr($m,'__version__','ok'))" 2>/dev/null || echo "MISSING python: $m"
done
python3 -c "import torch; print('torch CUDA available:', torch.cuda.is_available())" 2>/dev/null
echo "== Existing repo clone? =="; find ~ -maxdepth 3 -type d -iname "BIRA-foodsegCV" 2>/dev/null
echo "== Internet to GitHub =="; git ls-remote https://github.com/Eric-mmmm/BIRA-foodsegCV.git HEAD >/dev/null && echo "GitHub OK" || echo "NO GitHub access"
