#!/usr/bin/env bash

set -euo pipefail
export USER=sb2ea

REPO=/home/sb2ea/loadimbalance
PROJ=/projects/sb2ea

SRC=$REPO/code/third_party

BIN=$PROJ/bin
mkdir -p "$SRC" "$BIN"


conda deactivate 2>/dev/null || true
echo "gcc in use: $(which gcc)"
gcc --version | head -n 1

if [ ! -f "$SRC/stream.c" ]; then
    curl -fsSL https://www.cs.virginia.edu/stream/FTP/Code/stream.c -o "$SRC/stream.c"
fi
echo "stream.c sha256: $(sha256sum "$SRC/stream.c" | cut -d' ' -f1)"

N=40000000


FLAGS="-O3 -march=haswell -mtune=broadwell -fopenmp -DSTREAM_ARRAY_SIZE=$N -DNTIMES=20"

echo "flags: $FLAGS"
gcc $FLAGS "$SRC/stream.c" -o "$BIN/stream_cpu"
echo "built: $BIN/stream_cpu"


if [ -d "$PROJ/envs/gcc12" ]; then
    "$PROJ/envs/gcc12/bin/gcc" $FLAGS -static-libgcc \
        "$SRC/stream.c" -o "$BIN/stream_cpu_portable" \
      && echo "built: $BIN/stream_cpu_portable (el8-safe)"
fi

# stream.c is fetched, not committed
grep -qxF 'code/third_party/' "$REPO/.gitignore" 2>/dev/null \
  || echo 'code/third_party/' >> "$REPO/.gitignore"

echo
echo "Sanity: array size and expected memory"
"$BIN/stream_cpu" 2>&1 | sed -n '1,12p'
