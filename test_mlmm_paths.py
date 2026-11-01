"""CPU regressions for POLAR-shell endpoints and equivalent force optimizations.

Run with python -m unittest test_mlmm_paths. No pretrained weights or MD required.
"""

import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import openmm
import torch
from openmm import unit

from decomp.fires import ShapeFires
from decomp.mlmm import build_mlmm_system, segment, set_path
from decomp.mlmm_hremd import Evaluator, MLMMLadder, analyze, path_states, segments_of
from decomp.openmm_bridge import AnchoredForce


class CoulombKernel(torch.nn.Module):
    def forward(self, source_feats, positions, batch):
        distances = (positions[:, None] - positions[None]).square().sum(-1)
        kernel = (distances + 0.2).rsqrt() * (batch[:, None] == batch[None])
        terms = 0.5 * source_feats[:, 0] * (kernel @ source_feats[:, 0])
        return terms.new_zeros(int(batch.max()) + 1).index_add(0, batch, terms)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64), requires_grad=False)
        self.register_buffer("atomic_numbers", torch.tensor([1, 6, 7, 8]))
        self.register_buffer("r_max", torch.tensor(5.0))
        self.coulomb_energy = CoulombKernel()
        self.calls = 0

    def forward(self, data, **kwargs):
        self.calls += 1
        positions, batch = data["positions"], data["batch"]
        distances = (positions[:, None] - positions[None]).square().sum(-1)
        mask = (batch[:, None] == batch[None]) & ~torch.eye(len(batch), dtype=torch.bool)
        response = ((distances + 0.3).rsqrt() * mask).sum(1)
        sources = (self.weight + 0.04 * response).unsqueeze(-1)
        energy = self.coulomb_energy(source_feats=sources, positions=positions, batch=batch)
        node = 0.2 * response + 0.01 * data["total_charge"][batch].to(response.dtype)
        energy = energy + energy.new_zeros(len(energy)).index_add(0, batch, node)
        return dict(energy=energy, node_feats=torch.stack([response, sources[:, 0]], dim=-1))


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 0.8

    def forward(self, features, positions, edge_index):
        source, target = edge_index
        radius = (positions[target] - positions[source]).norm(dim=-1)
        pairs = self.scale * (features[source] * features[target]).sum(-1) * torch.exp(-radius)
        return features[:, 0], pairs


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.head = Backbone(), Head()


class MLMMPathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.directory.name)
        cls.positions = np.array([[12, 15, 15], [14, 15, 15],
                                  [16, 15, 15], [16.7, 15.5, 15], [15.3, 15.5, 15],
                                  [9, 15, 15], [9.7, 15.5, 15], [8.3, 15.5, 15],
                                  [12, 19, 15], [12.7, 19.5, 15], [11.3, 19.5, 15],
                                  [15, 18, 15], [15.7, 18.5, 15], [14.3, 18.5, 15]], dtype=float)
        cls.numbers = np.array([6, 7] + [8, 1, 1] * 4)
        cls.frag = np.array([0, 0] + [1] * 12)
        system = openmm.System()
        nonbonded = openmm.NonbondedForce()
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(1.0)
        nonbonded.setEwaldErrorTolerance(1e-7)
        for index, number in enumerate(cls.numbers):
            system.addParticle(float(number))
            charge = (0.1 if index == 0 else -0.1) if index < 2 else (-0.834 if number == 8 else 0.417)
            nonbonded.addParticle(charge, 0.3 if number != 1 else 0.1, 0.2 if number != 1 else 0)
        for oxygen in range(2, len(cls.positions), 3):
            for left, right in ((oxygen, oxygen + 1), (oxygen, oxygen + 2), (oxygen + 1, oxygen + 2)):
                nonbonded.addException(left, right, 0, 1, 0)
        system.addForce(nonbonded)
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
        source = cls.root / "system.xml"
        source.write_text(openmm.XmlSerializer.serialize(system))
        cls.leg = cls.root / "leg.npz"
        np.savez(cls.leg, positions=cls.positions, numbers=cls.numbers, box=np.full(3, 30.0),
                 resids=np.array([0, 0] + [value for value in range(1, 5) for _ in range(3)]),
                 resnames=np.array(["MOL", "MOL"] + ["HOH"] * 12), source=str(source))
        cls.platform = openmm.Platform.getPlatformByName("Reference")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_diagonal_force_and_fragment_batches(self):
        for electrostatics in (False, True):
            for drop_environment in (False, True):
                for cap in (None, 0.5):
                    model = Model()
                    force = AnchoredForce(model, self.numbers, self.frag, es=electrostatics,
                                           cap=cap, drop_env=drop_environment, device="cpu", ligand_charge=1)
                    for coordinate in (0.0, 0.4, 1.0, 1.5, 2.0):
                        force.set_lambda(coordinate)
                        force.fast_forces = False
                        force.batch_fragments = False
                        reference = force.energy_and_forces(self.positions, need_dudl=True)
                        derivatives = force.last_dudl.copy()
                        for batched in (False, True):
                            force.fast_forces = True
                            force.batch_fragments = batched
                            actual = force.energy_and_forces(self.positions)
                            np.testing.assert_allclose(actual[0], reference[0], atol=2e-11, rtol=0)
                            np.testing.assert_allclose(actual[1], reference[1], atol=2e-11, rtol=0)
                            force.energy_and_forces(self.positions, need_dudl=True)
                            np.testing.assert_allclose(force.last_dudl, derivatives, atol=2e-11, rtol=0)

    def test_endpoint_avoids_unneeded_backbones(self):
        model = Model()
        force = AnchoredForce(model, self.numbers, self.frag, cap=0.5, device="cpu")
        force.set_lambda(0)
        force.energy_and_forces(self.positions)
        self.assertEqual(model.backbone.calls, 1)
        model.backbone.calls = 0
        force.set_lambda(2)
        force.batch_fragments = True
        force.energy_and_forces(self.positions)
        self.assertEqual(model.backbone.calls, 1)

    def test_active_cap_matches_scalar_propagation(self):
        model = Model()
        model.head.scale = 20.0
        force = AnchoredForce(model, self.numbers, self.frag, es=True, cap=0.5,
                               drop_env=True, device="cpu")
        force.set_lambda(0.6)
        data = force._graph(self.positions)
        parts = force._U(data, force.lam)[2]
        self.assertGreater(float(parts["excess"].detach()), 1.0)
        force.fast_forces = False
        reference = force.energy_and_forces(self.positions, need_dudl=True)
        force.fast_forces, force.batch_fragments = True, True
        actual = force.energy_and_forces(self.positions)
        np.testing.assert_allclose(actual[0], reference[0], atol=2e-11, rtol=0)
        np.testing.assert_allclose(actual[1], reference[1], atol=2e-11, rtol=0)

    def test_fp32_takes_fast_paths(self):
        # The production model is FP32; fast propagation and batching are not dtype-gated.
        force = AnchoredForce(Model().float(), self.numbers, self.frag, es=True, cap=0.5,
                               drop_env=True, device="cpu")
        force.fast_forces, force.batch_fragments = True, True
        force.set_lambda(0.5)
        with patch.object(force, "_U", wraps=force._U) as energy_call:
            force.energy_and_forces(self.positions)
        self.assertEqual(energy_call.call_args.args[1].ndim, 0)
        force.set_lambda(2)
        force.energy_and_forces(self.positions)

    def test_shape_restraint_gradient_and_identity(self):
        water = np.arange(2, len(self.positions)).reshape(-1, 3)[:, :1]
        fires = ShapeFires([0, 1], water, 2)
        identity = fires.select_inner(self.positions, np.full(3, 30.0)).copy()
        positions = self.positions.copy()
        positions[8:11] += [0, -2.137, 0]
        for fraction in (0.0, 0.4, 1.0):
            fires.set_shape(fraction)
            energy, forces = fires.energy_and_forces(positions, np.full(3, 30.0))
            self.assertTrue(np.isfinite(energy))
            for atom, axis in ((0, 0), (1, 1), (8, 1)):
                plus, minus = positions.copy(), positions.copy()
                plus[atom, axis] += 1e-5
                minus[atom, axis] -= 1e-5
                difference = -(fires.energy_and_forces(plus, np.full(3, 30.0))[0]
                               - fires.energy_and_forces(minus, np.full(3, 30.0))[0]) / 2e-5
                self.assertAlmostEqual(forces[atom, axis], difference, delta=5e-5)
            np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8, rtol=0)
            if fraction == 1.0:
                np.testing.assert_array_equal(forces[1], 0.0)
        np.testing.assert_array_equal(identity, fires.inner.numpy())
        np.testing.assert_array_equal(identity, fires.point.inner.numpy())

    def test_all_state_energies_and_derivatives(self):
        for mode in ({}, dict(shell_mm=True), dict(polar_shell=True)):
            model = Model()
            system, force, info = build_mlmm_system(str(self.leg), model, n_inner=2, device="cpu",
                                                   batch_fragments=True, **mode)
            limit = 4 if mode.get("shell_mm") else 5
            coordinates = np.arange(0.0, limit + 0.1, 0.5)
            evaluator = Evaluator(system, force, info, coordinates, self.platform)
            energies, parts = evaluator.all_states(self.positions)
            for index, coordinate in enumerate(coordinates):
                set_path(evaluator.ctx, force, coordinate, info["lig"], **mode)
                evaluator.ctx.setPositions(self.positions * unit.angstrom)
                energy = evaluator.ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                self.assertAlmostEqual(energy, energies[index], delta=2e-7)
            for coordinate in (0.5, 1.5, 2.5, 3.5, 4.5):
                if coordinate >= limit:
                    continue
                values, atoms = evaluator.derivatives(self.positions, coordinate, parts)
                name = segments_of(coordinate, **mode)[0]
                endpoints = []
                increment = 1e-4
                for shifted in (coordinate - increment, coordinate + increment):
                    set_path(evaluator.ctx, force, shifted, info["lig"], **mode)
                    endpoints.append(evaluator.ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                        unit.kilojoule_per_mole))
                self.assertAlmostEqual(values[name], (endpoints[1] - endpoints[0]) / (2 * increment), delta=3e-4)
                if atoms:
                    self.assertAlmostEqual(sum(value.sum() for value in atoms.values()), values[name], delta=2e-5)
            second, _ = evaluator.all_states(self.positions)
            np.testing.assert_allclose(second, energies, atol=2e-7, rtol=0)

    def test_path_validation(self):
        self.assertEqual(segment(3.5, polar_shell=True), (0.0, 0.5, 2.0, 0.0))
        self.assertEqual(segment(5, polar_shell=True), (0.0, 0.0, 2.0, 0.0))
        self.assertEqual(segments_of(4, polar_shell=True), ["W3", "shape"])
        self.assertIn(5, path_states(polar_shell=True))
        with self.assertRaises(ValueError):
            segment(1, shell_mm=True, polar_shell=True)
        with self.assertRaises(ValueError):
            segment(1, ref=True, polar_shell=True)

    def test_checkpoint_rejects_model_change_before_restoring(self):
        directory = self.root / "checkpoint"
        directory.mkdir(exist_ok=True)
        np.savez(directory / "state.npz", qs=[0.0, 1.0], shell_mm=True)
        ladder = MLMMLadder.__new__(MLMMLadder)
        ladder.qs = np.array([0.0, 1.0])
        ladder.shell_mm, ladder.polar_shell, ladder.ref = False, True, False
        with self.assertRaisesRegex(RuntimeError, "different Hamiltonian"):
            ladder.load(str(directory))
        with self.assertRaisesRegex(RuntimeError, "explicit init_q"):
            ladder.init_from(str(directory))
        np.savez(directory / "state.npz", qs=[0.0, 1.0], polar_shell=True)
        ladder.polar_shell = False
        with self.assertRaisesRegex(RuntimeError, "explicit init_q"):
            ladder.init_from(str(directory))

    def test_shape_work_is_global_in_closure(self):
        coordinates = np.arange(6, dtype=float)
        records = dict(u=[np.zeros((6, 6))], dudq=[[]], atoms=[[]])
        coefficients = dict(W1=1.0, W2=2.0, cut=3.0, W3=4.0, shape=5.0)
        for coordinate in coordinates:
            names = segments_of(coordinate, polar_shell=True)
            records["dudq"][0].append({name: coefficients[name] for name in names})
            atoms = {}
            for name in names:
                if name in ("W1", "W3"):
                    atoms[name] = np.array([coefficients[name]])
                elif name == "W2":
                    atoms.update(W2_ml=np.array([2.0]), W2_coul=np.array([0.0]), W2_lj=np.array([0.0]))
            records["atoms"][0].append(atoms)
        fake = types.ModuleType("decomp.fe")
        fake.mbar_free_energy = lambda *args, **kwargs: dict(dG=0.0, dG_err=0.0, overlap=np.eye(6))
        fake.report_overlap = lambda value: dict(min=0.0)
        with patch.dict("sys.modules", {"decomp.fe": fake}):
            result = analyze(coordinates, records, 2.5, polar_shell=True)
        self.assertEqual(result["dG_shape"], 5.0)
        self.assertEqual(result["closure"], 0.0)
        self.assertTrue(result["path_complete"])
        self.assertIn("not the bulk-MM", result["reference_note"])
        with patch.dict("sys.modules", {"decomp.fe": fake}):
            partial = analyze(np.array([0.0, 1.0]), dict(u=[], dudq=[], atoms=[]), 2.5, polar_shell=True)
        self.assertFalse(partial["path_complete"])
        self.assertEqual(partial["endpoint_reference"], "partial POLAR-shell path")


if __name__ == "__main__":
    unittest.main()
