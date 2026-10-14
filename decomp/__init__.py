"""Energy-decomposition head for equivariant MLIPs.

Dev environment: miniforge3/envs/openmm_dev (e3nn 0.4.4, torch 2.12.0,
mace-torch 0.3.16). e3nn 0.4.4's Wigner-table load breaks on torch >= 2.6
(`weights_only` default flipped to True); the line below fixes that. Under
e3nn >= 0.5.1 (openmm_dev_ubio) it is a no-op.

# Keep while the dev env is on e3nn 0.4.4. Requires `decomp`
# to be imported before e3nn to have any effect.
"""

import torch

torch.serialization.add_safe_globals([slice])
