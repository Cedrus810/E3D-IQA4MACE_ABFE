#!/usr/bin/env bash
# Step 3 (DESIGN §10): CypD-8 S0, eta 0.0..1.0 step 0.1 x starts rep1..rep4 (S0 last frames), 500 cycles each,
# L 20 x 2 fs, float64 model energy. NW workers (default 1) share one GPU. Resumable: finished chains with the same
# settings (incl. backbone / head paths) are reused (eta_hmc.py chain checks the saved metadata).
# Failed chains and analysis use logs/step3${TAG}_* paths; the analysis says ALL DONE only when
# every chain finished and the analysis ran, otherwise INCOMPLETE (and the script exits 1).
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONDONTWRITEBYTECODE=1
ETAS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
STARTS="1 2 3 4"
TAG=${TAG:-_atoms_v2}
mkdir -p logs
FAILED="logs/step3${TAG}_failed.txt"
ANALYSIS="logs/step3${TAG}_analyze.log"
: > "$FAILED"
jobs=()
for s in $STARTS; do
  for e in $ETAS; do
    jobs+=("$e:$s")
  done
done
NW=${NW:-1}
for w in $(seq 0 $((NW - 1))); do
  (
    for i in "${!jobs[@]}"; do
      (( i % NW == w )) || continue
      e=${jobs[$i]%:*}; s=${jobs[$i]#*:}
      $P -B eta_hmc.py chain CyclophilinD_ligand-8 --eta "$e" --start "$s" --cycles 500 --L 20 --tag "$TAG" \
        >> "logs/step3${TAG}_w$w.log" 2>&1 || { echo "FAILED eta $e start $s" >> "logs/step3${TAG}_w$w.log"
                                        echo "eta $e start $s" >> "$FAILED"; }
    done
  ) &
done
wait
nfail=$(wc -l < "$FAILED")
ana=0
$P -B eta_hmc.py analyze CyclophilinD_ligand-8 --tag "$TAG" --expect-etas "${ETAS// /,}" \
  --expect-starts "${STARTS// /,}" > "$ANALYSIS" 2>&1 || ana=$?
if (( nfail == 0 && ana == 0 )); then
  echo "ALL DONE $(date '+%F %T')" >> "$ANALYSIS"
else
  echo "INCOMPLETE $(date '+%F %T'): $nfail of ${#jobs[@]} chains failed (see $FAILED); analyze exit $ana" \
    >> "$ANALYSIS"
  exit 1
fi
