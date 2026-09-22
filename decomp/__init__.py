"""Energy-decomposition head for equivariant MLIPs.

Dev environment: miniforge3/envs/openmm_dev_ubio (e3nn 0.5.1, torch 2.12.1,
mace-torch 0.3.16). e3nn >= 0.5.1 needs no shim; the line below is a no-op there
and only matters if the package is imported under e3nn 0.4.4, whose Wigner-table
load breaks on torch >= 2.6 (`weights_only` default flipped to True).

# ponytail: keep only while e3nn 0.4.4 envs are still in use. Requires `decomp`
# to be imported before e3nn to have any effect.
"""

import torch

torch.serialization.add_safe_globals([slice])
