#!/bin/bash

echo "===== Step 0: Entering interactive OSCAR session ====="
echo "Make sure you ran: interact -n 4 -m 30g -t 6:00:00 -q batch -a hding33"
echo "If not, exit and run interact first."
sleep 2

echo "===== Step 1: Load LLVM (required by OpenSpiel) ====="
module load llvm/16.0.2-mq6g5lb

export CC=$(llvm-config --bindir)/clang
export CXX=$(llvm-config --bindir)/clang++

echo "Compiler:"
$CC --version
$CXX --version

echo "===== Step 2: Move into project directory ====="
cd ~/Junqi-DRL || { echo "Junqi-DRL not found!"; exit 1; }

echo "===== Step 3: Remove old venv to avoid conflicts ====="
rm -rf .venv
echo "Old virtual environment removed."

echo "===== Step 4: Install uv (Python environment manager) ====="
pip install --upgrade uv

echo "===== Step 5: Run uv sync to fully rebuild environment ====="
uv sync

if [ ! -d ".venv" ]; then
    echo "ERROR: uv sync failed to create .venv"
    exit 1
fi

echo "===== Step 6: Activate new .venv ====="
source .venv/bin/activate

echo "===== Step 7: Test if environment works ====="
python examples/junqi_standard_example.py

echo "===== SETUP COMPLETE ====="
echo ""
echo "You can now run:"
echo "    source .venv/bin/activate"
echo "    python scripts/train_transformer.py"
echo ""
echo "🎉 Your OSCAR environment is fully configured!"
