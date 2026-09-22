#!/bin/bash
# GPU queue for the E3D-IQA decomposition project.
# Run from /home/ruigengji/E3D.  Logs land in logs/.
#
# Each block is independent -- comment out what you do not want.
# Total on one RTX 2080 Ti: ~2 h.  Less on an A6000.
set -u
cd "$(dirname "$0")"
P=/home/ruigengji/miniforge3/envs/openmm_dev_ubio/bin/python
N='warning|visitor|cuequiv|detach|^  return|^  print'
mkdir -p logs
run () { echo "=== $1 ==="; shift; "$@" 2>&1 | grep -viE --line-buffered "$N"; }

# ---------------------------------------------------------------- sanity
# ~1 min.  Must pass before anything else is worth reading.
run "self-checks" $P -u test_decomp.py       | tee logs/decomp.log
run "mace adapter" $P -u test_mace_adapter.py | tee logs/adapter.log

# ------------------------------------------------- WP3: does L_int scale?
# args: config  n_train  steps  n_test
# The open question. hcno_small gave 0.80 kcal/mol but on only 26 held-out
# molecule pairs; these have 9x-15x the chemical diversity.
run "L_int organic"  $P -u test_lint.py organic 232011 15000 2000 | tee logs/lint_organic.log
run "L_int hcno"     $P -u test_lint.py hcno    151200 15000 2000 | tee logs/lint_hcno.log
run "L_int full"     $P -u test_lint.py full    295617 15000 2000 | tee logs/lint_full.log

# --------------------------------------------- how low does L_int actually go
# 4x the steps on the broadest config that passed above. ~50 min.
run "L_int organic long" $P -u test_lint.py organic 232011 60000 4000 | tee logs/lint_organic_long.log

# ------------------------------------- is the 30% / 124% gap sample size?
# S13.2 saw D_ij at 30% with 32 structures, S13.3 at 124% with 8. Same
# backbone, different n -- so the comparison is confounded. Settle it.
for n in 8 32 128 512; do
  run "frozen gauge n=$n" env NSTRUCT=$n $P -u test_frozen_gauge.py | tee logs/frozen_n$n.log
done

# ------------------------- alchemical endpoint identities (seconds, no GPU load)
run "endpoints" $P -u test_endpoints.py | tee logs/endpoints.log

# ------------------------------------- do the four constraints cooperate?
# S13.4 trained on L_int alone, so nothing held the total energy and the
# decomposition was not a potential. This anchors E/F to the backbone's own
# output and asks what L_int costs. ~25 min.
# S13.6: the four constraints fight (3.25x on energy from A to C). Is that
# fundamental, or is a 45,696-parameter head simply too small? ~1.5 h total.
for H in 64 256 512; do
  run "joint hidden=$H" $P -u test_joint.py organic 50000 8000 $H \
    | tee logs/joint_h$H.log
done

echo "=== queue done ==="
echo
echo "what to read:"
echo "  logs/lint_*.log        last 6 lines: test MAE vs predict-zero, in kcal/mol"
echo "  logs/frozen_n*.log     D MAE column vs |D| -- is the 30%/124% gap sample size?"
echo "  logs/joint.log         last 3 lines: what L_int costs the total energy"
