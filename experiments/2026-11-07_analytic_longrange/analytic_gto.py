"""Isolated analytic l=0/1 candidate for the installed real-space GTO stencil.

Preserves its softened radial kernel and normalizations. Not wired into production.
"""

import copy
import math

import torch
from torch import nn
from graph_longrange.utils import FIELD_CONSTANT


def radial(positions, widths):
    """r_j-r_i, unit vectors, K, K', K''; harmless diagonal excluded by callers."""
    n = len(positions)
    delta = positions[None, :, :] - positions[:, None, :]
    diagonal = torch.eye(n, dtype=positions.dtype, device=positions.device)
    r = torch.sqrt((delta * delta).sum(-1) + diagonal)
    direction = delta / r[..., None]
    rr = r[..., None]
    a = .5 / widths
    f = torch.erf(rr * a)
    fp = 2 * a / math.sqrt(math.pi) * torch.exp(-(rr * a) ** 2)
    fpp = -2 * a * a * rr * fp
    b = rr + 1e-6
    k = f / b
    kp = fp / b - f / (b * b)
    kpp = fpp / b - 2 * fp / (b * b) + 2 * f / (b * b * b)
    return r, direction, k, kp, kpp


def dipoles(source):
    return source[:, [3, 1, 2]] if source.shape[-1] == 4 else source.new_zeros((len(source), 3))


def mask_for(batch):
    return (batch[:, None] == batch[None, :]) & ~torch.eye(len(batch), dtype=torch.bool, device=batch.device)


class AnalyticEnergy(nn.Module):
    def __init__(self, template):
        super().__init__()
        if template.density_max_l not in (0, 1):
            raise ValueError("candidate supports only l=0/1")
        self.density_max_l = template.density_max_l
        self.density_smearing_width = template.density_smearing_width
        self.include_self_interaction = template.include_self_interaction
        self.self_interaction = copy.deepcopy(template.self_interaction)

    def forward(self, source_feats, positions, batch):
        source = source_feats.squeeze(-2) if source_feats.ndim == 3 else source_feats
        q, p = source[:, 0], dipoles(source)
        width = positions.new_tensor([self.density_smearing_width])
        r, direction, k, kp, kpp = radial(positions, width)
        k, kp, kpp = k[..., 0], kp[..., 0], kpp[..., 0]
        pi_n = (p[:, None, :] * direction).sum(-1)
        pj_n = (p[None, :, :] * direction).sum(-1)
        pp = p @ p.T
        pair = (q[:, None] * q[None, :] * k
                + (q[:, None] * pj_n - q[None, :] * pi_n) * kp
                - (kpp - kp / r) * pi_n * pj_n - (kp / r) * pp)
        pair = pair * mask_for(batch).to(pair.dtype) * (FIELD_CONSTANT / (4 * math.pi))
        node = .5 * pair.sum(1)
        energy = node.new_zeros(int(batch.max()) + 1).index_add(0, batch, node)
        if self.include_self_interaction:
            self_fields = self.self_interaction(source)
            self_node = .5 * (source * self_fields).sum(-1)
            energy = energy.index_add(0, batch, self_node)
        return energy


class AnalyticFeatures(nn.Module):
    def __init__(self, template):
        super().__init__()
        if template.density_max_l not in (0, 1) or template.projection_max_l not in (0, 1):
            raise ValueError("candidate supports only l=0/1")
        self.density_max_l = template.density_max_l
        self.projection_max_l = template.projection_max_l
        self.include_self_interaction = template.include_self_interaction
        self.num_radial = template.num_radial
        self.self_interaction = copy.deepcopy(template.self_interaction)
        self.register_buffer("widths", template.total_width_factors.clone())
        self.register_buffer("scalar_scale", template.l0_factors.clone())
        self.register_buffer("vector_scale", template.l1_factors.clone() * template.offset)

    def forward(self, source_feats, node_positions, batch):
        source = source_feats.squeeze(-2) if source_feats.ndim == 3 else source_feats
        q, p = source[:, 0], dipoles(source)
        r, direction, k, kp, kpp = radial(node_positions, self.widths)
        valid = mask_for(batch).to(k.dtype)[..., None]
        pj_n = (p[None, :, :] * direction).sum(-1)
        phi = ((q[None, :, None] * k + pj_n[..., None] * kp) * valid).sum(1)
        scalar = phi * self.scalar_scale
        if self.projection_max_l:
            # d(phi_i)/d(R_i), evaluated directly rather than a one-sided receiver stencil.
            hess_p = ((kpp - kp / r[..., None])[..., None] * direction[:, :, None, :] * pj_n[..., None, None]
                      + (kp / r[..., None])[..., None] * p[None, :, None, :])
            gradient = (-(q[None, :, None, None] * kp[..., None] * direction[:, :, None, :] + hess_p)
                        * valid[..., None]).sum(1)
            vector = gradient * self.vector_scale[None, :, None]
            features = torch.cat([scalar, vector[..., [1, 2, 0]].reshape(len(source), -1)], dim=-1)
        else:
            features = scalar
        self_terms = self.self_interaction(source)
        if self.include_self_interaction:
            features = features + self_terms
        return features, self_terms, None


def replace_in_clone(model):
    """Caller owns this model clone; installed modules and checkpoint files are untouched."""
    from graph_longrange.realspace_electrostatics import RealSpaceFiniteDiffereneEnergy, RealSpaceFiniteDifferenceElectrostaticFeatures
    changes = []
    for parent_path, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, RealSpaceFiniteDiffereneEnergy):
                replacement = AnalyticEnergy(child)
            elif isinstance(child, RealSpaceFiniteDifferenceElectrostaticFeatures):
                replacement = AnalyticFeatures(child)
            else:
                continue
            setattr(parent, name, replacement)
            changes.append({"path": f"{parent_path}.{name}", "old": type(child).__name__,
                            "new": type(replacement).__name__, "original_offset_A": child.offset})
    if not changes:
        raise ValueError("no supported real-space modules found")
    return changes
