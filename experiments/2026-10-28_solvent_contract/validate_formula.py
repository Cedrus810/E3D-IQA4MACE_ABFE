"""Independent partition-function checks of the proposed path attribution.

Two groups / three atoms / five configurations. Exact Boltzmann sums supply
the reference; quadrature computes the atomic TI. No molecular data are fit.
"""

import itertools
import json
from pathlib import Path

import numpy as np

from attribution import assemble_step_atom_matrix, integrate_endpoint_atoms

HERE = Path(__file__).resolve().parent
RT = 2.49433878
BASE = np.array([1.0, -0.5, 2.0, 0.7, 3.0])
MM = np.array([[2.0, -3.0, 1.0, 4.0, -1.0],
               [1.0, 2.0, -4.0, 0.0, 3.0],
               [-3.0, 1.0, 3.0, -2.0, 1.5]])
DELTA0 = np.array([[1.0, -2.0, 3.0, -1.0, 2.0],
                   [-1.0, 3.0, -2.0, 2.0, 0.0],
                   [2.0, -1.0, 0.5, 3.0, -2.0]])
RESPONSE = np.array([[0.0, 1.2, -0.8, 0.5, -0.4],
                     [0.5, -0.2, 1.0, -0.7, 0.2],
                     [-1.5, 0.4, 2.0, -0.3, 1.0]])
GROUPS = ((0, 1), (2,))
NODES, WEIGHTS = np.polynomial.legendre.leggauss(96)
NODES, WEIGHTS = (NODES + 1) / 2, WEIGHTS / 2


def free_energy(u):
    q = -np.asarray(u, float) / RT
    m = q.max(axis=-1)
    return -RT * (m + np.log(np.exp(q - m[..., None]).sum(axis=-1)))


def probabilities(u):
    q = -np.asarray(u, float) / RT
    q -= q.max(axis=-1, keepdims=True)
    w = np.exp(q)
    return w / w.sum(axis=-1, keepdims=True)


def state(active):
    v = BASE + MM[active].sum(axis=0)
    d = np.zeros_like(DELTA0)
    # Remaining atoms respond when OTHER groups become dummy; this is a
    # chosen finite-state example, not a claim about the real backbone.
    for i in np.flatnonzero(active):
        d[i] = DELTA0[i] + (3 - active.sum()) * RESPONSE[i]
    return v, d


def solve(order):
    active = np.ones(3, bool)
    vs, ds, bs, aas, grid_as = [], [], [], [], []
    for j in range(len(order) + 1):
        v, d = state(active)
        vs.append(v)
        ds.append(d)
        p = probabilities(v[None, :] + NODES[:, None] * d.sum(axis=0)[None, :])
        aas.append(WEIGHTS @ (p @ d.T))
        grid = np.linspace(0, 1, 1025)
        means = probabilities(v[None, :] + grid[:, None] * d.sum(axis=0)[None, :]) @ d.T
        grid_as.append(integrate_endpoint_atoms(grid, means))
        if j < len(order):
            group = GROUPS[order[j]]
            dg = MM[list(group)].sum(axis=0)
            pmm = probabilities(v[None, :] - NODES[:, None] * dg[None, :])
            b = np.zeros(3)
            b[list(group)] = -(WEIGHTS @ (pmm @ MM[list(group)].T))
            bs.append(b)
            active[list(group)] = False
    vs, ds, b, a = map(np.asarray, (vs, ds, bs, aas))
    fv, fe = free_energy(vs), free_energy(vs + ds.sum(axis=1))
    m = assemble_step_atom_matrix(b, a)
    ga = np.asarray(grid_as)
    mg = assemble_step_atom_matrix(b, ga)
    exact_a = fe - fv
    np.testing.assert_allclose(a.sum(axis=1), exact_a, atol=1e-11, rtol=0)
    np.testing.assert_allclose(b.sum(axis=1), np.diff(fv), atol=1e-11, rtol=0)
    np.testing.assert_allclose(m["step_total"], np.diff(fe), atol=1e-11, rtol=0)
    np.testing.assert_allclose(m["atom_total"], m["telescoped_atom"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(m["total"], fe[-1] - fe[0], atol=1e-11, rtol=0)
    assert np.array_equal(a[-1], np.zeros(3))
    for j in range(3):
        # Independent definition via exact partition ratios (not TI).
        p0 = probabilities(vs[j])
        exp_a = -RT * np.log(p0 @ np.exp(-ds[j].sum(axis=0) / RT))
        np.testing.assert_allclose(exp_a, exact_a[j], atol=1e-12, rtol=0)
    assert np.max(np.abs(mg["step_total"] - np.diff(fe))) < 2e-6
    return {"B": b, "A": a, "matrix": m, "F_V": fv, "F_E": fe,
            "grid_A": ga, "vs": vs, "ds": ds,
            "max_step_error": float(np.max(np.abs(m["step_total"] - np.diff(fe)))),
            "grid_max_step_error": float(np.max(np.abs(mg["step_total"] - np.diff(fe))))}


def shapley_comparison():
    """Same smooth free-energy surface, diagonal TI vs all 3! orders.

    This example has a fixed additive interaction energy and EXACT state
    partition functions. Even here diagonal TI need not equal permutation
    averaging, because the free energy is nonlinear in the couplings.
    """
    diagonal = -(WEIGHTS @ (probabilities(BASE[None, :] +
                 (1 - NODES[:, None]) * MM.sum(axis=0)[None, :]) @ MM.T))
    orders = []
    for order in itertools.permutations(range(3)):
        active = np.ones(3, bool)
        current = float(free_energy(BASE + MM.sum(axis=0)))
        shares = np.zeros(3)
        for i in order:
            active[i] = False
            nxt = float(free_energy(BASE + MM[active].sum(axis=0)))
            shares[i] = nxt - current
            current = nxt
        orders.append(shares)
    shapley = np.mean(orders, axis=0)
    np.testing.assert_allclose(diagonal.sum(), shapley.sum(), atol=1e-11, rtol=0)
    assert np.max(np.abs(diagonal - shapley)) > 1e-3
    return {"diagonal_atomic_TI": diagonal.tolist(), "permutation_average": shapley.tolist(),
            "max_difference": float(np.max(np.abs(diagonal - shapley))),
            "scope": "finite-state counterexample, not the molecular target"}


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("invalid / missing input was accepted")


def main():
    fwd, rev = solve((0, 1)), solve((1, 0))
    np.testing.assert_allclose(fwd["matrix"]["total"], rev["matrix"]["total"], atol=1e-11, rtol=0)
    assert np.max(np.abs(fwd["matrix"]["atom_total"] - rev["matrix"]["atom_total"])) > 1e-3
    # Atom 2 remains coupled during G1 removal: its endpoint correction
    # changes, giving a NONZERO contribution outside the removed group.
    cross = float(fwd["matrix"]["step_atom"][0, 2])
    assert abs(cross) > 1e-3
    chemistry_group_totals = np.array([fwd["matrix"]["atom_total"][list(g)].sum() for g in GROUPS])
    assert np.max(np.abs(chemistry_group_totals - fwd["matrix"]["step_total"])) > 1e-3
    # Independent per-atom EXP is not additive, despite additive energies.
    p0, d0 = probabilities(fwd["vs"][0]), fwd["ds"][0]
    wrong_atoms = -RT * np.log(np.exp(-d0 / RT) @ p0)
    wrong_exp_error = float(abs(wrong_atoms.sum() - fwd["A"][0].sum()))
    assert wrong_exp_error > 1e-3
    # Wrong ensemble negative control: eta=0 means used for every eta.
    wrong_ensemble_error = float(abs((d0 @ p0).sum() - fwd["A"][0].sum()))
    assert wrong_ensemble_error > 1e-3
    # Common energy origin does not change any partition-function differences.
    np.testing.assert_allclose(free_energy(fwd["vs"] + 1234.0) - free_energy(fwd["vs"]),
                               np.full(3, 1234.0), atol=1e-11, rtol=0)
    for eta in ([0.2, 1.0], [0.0, 0.8], [0.0, 0.8, 0.4, 1.0]):
        expect_value_error(lambda eta=eta: integrate_endpoint_atoms(eta, np.zeros((len(eta), 3))))
    expect_value_error(lambda: integrate_endpoint_atoms([0, 1], [[0, 1], [np.nan, 0]]))
    expect_value_error(lambda: assemble_step_atom_matrix(np.zeros((2, 3)), np.zeros((2, 3))))
    expect_value_error(lambda: assemble_step_atom_matrix(np.zeros((2, 3)), np.full((3, 3), np.nan)))
    result = {"kind": "independent exact finite-state validation; no molecular sampling acceptance",
              "checks_passed": ["atomic_endpoint_eta_TI_vs_exact_partition_ratio",
                                "atomic_MM_TI_vs_exact_partition_difference", "step_atom_matrix_closure",
                                "endpoint_corrections_telescope", "all_dummy_analytic_zero",
                                "same_total_different_group_order_attributions", "cross_group_response_retained",
                                "chemical_grouping_not_confused_with_removal_steps", "wrong_atom_EXP_rejected",
                                "wrong_ensemble_negative_control", "common_energy_offset",
                                "incomplete_eta_nonfinite_and_shape_input_rejected",
                                "diagonal_TI_not_assumed_equal_to_permutation_average"],
              "max_step_error": fwd["max_step_error"], "grid_max_step_error": fwd["grid_max_step_error"],
              "forward_step_atom": fwd["matrix"]["step_atom"].tolist(),
              "forward_step_totals": fwd["matrix"]["step_total"].tolist(),
              "forward_atom_totals": fwd["matrix"]["atom_total"].tolist(),
              "reverse_atom_totals": rev["matrix"]["atom_total"].tolist(),
              "forward_chemical_group_totals": chemistry_group_totals.tolist(),
              "total": fwd["matrix"]["total"], "cross_group_step1_atom2": cross,
              "wrong_independent_EXP_error": wrong_exp_error, "wrong_ensemble_error": wrong_ensemble_error,
              "shapley_counterexample": shapley_comparison()}
    (HERE / "formula_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
