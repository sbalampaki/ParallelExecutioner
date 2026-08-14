#!/bin/bash
# verify_core_objects.sh -- check every kernel_core.o against its OWN
# expected hash.


set -uo pipefail

BIN=/projects/sb2ea/bin
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common"
UPDATE=0
[ "${1:-}" = "--update" ] && UPDATE=1

# label : object path : expected file
ROWS=(
  "broadwell   :$BIN/kernel_core.o     :$EXP/kernel_core.o.sha256.expected"
  "sandybridge :$BIN/snb/kernel_core.o :$EXP/kernel_core.o.sha256.expected.snb"
  "cascadelake :$BIN/el8/kernel_core.o :$EXP/kernel_core.o.sha256.expected.el8"
)

rc=0
printf '%-13s %-10s %s\n' "ISA" "STATUS" "sha256"
for row in "${ROWS[@]}"; do
    IFS=: read -r isa obj exp <<< "$row"
    isa=${isa// /}; obj=${obj// /}; exp=${exp// /}

    if [ ! -f "$obj" ]; then
        printf '%-13s %-10s %s\n' "$isa" "ABSENT" "$obj"
        continue                      # not every cluster has every build
    fi
    got=$(sha256sum "$obj" | cut -d' ' -f1)

    if [ "$UPDATE" = 1 ]; then
        printf '%s  kernel_core.o\n' "$got" > "$exp"
        printf '%-13s %-10s %s\n' "$isa" "UPDATED" "$got"
        continue
    fi
    if [ ! -f "$exp" ]; then
        printf '%-13s %-10s %s\n' "$isa" "NO-EXPECT" "$got"
        echo "    no $exp -- run with --update to record it" >&2
        rc=1; continue
    fi
    want=$(cut -d' ' -f1 < "$exp")
    if [ "$got" = "$want" ]; then
        printf '%-13s %-10s %s\n' "$isa" "OK" "$got"
    else
        printf '%-13s %-10s %s\n' "$isa" "MISMATCH" "$got"
        echo "    expected $want" >&2
        rc=1
    fi
done

b=$([ -f "$BIN/kernel_core.o" ]     && sha256sum "$BIN/kernel_core.o"     | cut -d' ' -f1)
e=$([ -f "$BIN/el8/kernel_core.o" ] && sha256sum "$BIN/el8/kernel_core.o" | cut -d' ' -f1)
s=$([ -f "$BIN/snb/kernel_core.o" ] && sha256sum "$BIN/snb/kernel_core.o" | cut -d' ' -f1)
echo
if [ -n "${b:-}" ] && [ -n "${e:-}" ]; then
    [ "$b" = "$e" ] \
      && echo "OK       el8 == broadwell  (same ISA level, different machine)" \
      || { echo "VIOLATED el8 != broadwell -- addendum 2 s5 asserts equality"; rc=1; }
fi
if [ -n "${b:-}" ] && [ -n "${s:-}" ]; then
    [ "$b" != "$s" ] \
      && echo "OK       snb != broadwell  (no FMA3: different ISA level)" \
      || { echo "VIOLATED snb == broadwell -- the snb build did not take"; rc=1; }
fi

[ $rc -eq 0 ] && echo && echo "all kernel_core.o objects verified"
exit $rc
