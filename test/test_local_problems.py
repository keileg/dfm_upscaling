import numpy as np
import unittest
import porepy as pp

from test import test_utils

import sys

sys.path.insert(0, "../")

import local_problems as lp
import interaction_region as ia_reg
from local_grid_bucket import LocalGridBucketSet
from utils import create_grids
from fv_dfm import FVDFM, Tpfa_DFM, Mpfa_DFM


class TestLocalProblems(unittest.TestCase):
    def test_partition_unit_tpfa_no_micro_fractures(self):
        g = create_grids.cart_2d()
        internal_faces = [1, 4, 7, 11, 12, 13, 14]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM())

    def test_partition_unit_mpfa_no_micro_fractures(self):
        g = create_grids.cart_2d()
        internal_nodes = [4, 7]
        self._check_partition_unity(g, internal_nodes, Mpfa_DFM())

    def test_partition_unity_tpfa_single_local_micro_fracture(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.8, 1.2], [1.3, 1.6]])
        edges = np.array([[0], [1]])
        internal_faces = [4]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM(), pts, edges)

    def test_partition_unity_mpfa_single_local_micro_fracture(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.6, 1.4], [0.6, 1.1]])
        edges = np.array([[0], [1]])
        internal_nodes = [4]
        self._check_partition_unity(g, internal_nodes, Mpfa_DFM(), pts, edges)

    def test_partition_unity_tpfa_single_boundary_intersecting_local_micro_fracture(
        self,
    ):
        g = create_grids.cart_2d()
        pts = np.array([[0.7, 0.9], [1.1, 1.4]])
        edges = np.array([[0], [1]])
        internal_faces = [4]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM(), pts, edges)

    def test_partition_unity_mpfa_single_boundary_intersecting_local_micro_fracture(
        self,
    ):
        g = create_grids.cart_2d()
        pts = np.array([[0.7, 0.9], [0.475, 1.4]])
        edges = np.array([[0], [1]])
        internal_nodes = [4]
        self._check_partition_unity(g, internal_nodes, Mpfa_DFM(), pts, edges)

    def test_partition_unity_tpfa_multiple_local_micro_fractures(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.8, 1.2, 0.8, 1.1], [1.3, 1.6, 1.6, 1.8]])
        edges = np.array([[0, 2], [1, 3]])
        internal_faces = [4]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM(), pts, edges)

    def test_partition_unity_mpfa_multiple_local_micro_fractures(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.6, 1.2, 1.2, 0.6], [0.6, 0.8, 1.2, 1.4]])
        edges = np.array([[0, 2], [1, 3]])
        internal_nodes = [4]
        self._check_partition_unity(g, internal_nodes, Mpfa_DFM(), pts, edges)

    def test_partition_unity_tpfa_multiple_intersecting_local_micro_fractures(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.8, 1.2, 1.1, 1.1], [1.3, 1.6, 1.4, 1.8]])
        edges = np.array([[0, 2], [1, 3]])
        internal_faces = [4]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM(), pts, edges)

    def test_partition_unity_mpfa_multiple_intersecting_local_micro_fractures(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.6, 1.2, 0.6, 1.2], [0.6, 0.8, 1.2, 0.6]])
        edges = np.array([[0, 2], [1, 3]])
        internal_nodes = [4]
        self._check_partition_unity(g, internal_nodes, Mpfa_DFM(), pts, edges)

    def test_partition_unity_tpfa_multiple_boundary_intersecting_local_micro_fracture(
        self,
    ):
        g = create_grids.cart_2d()
        pts = np.array([[0.7, 0.9, 0.815, 0.95], [1.1, 1.4, 1.35, 1]])
        edges = np.array([[0, 2], [1, 3]])
        internal_faces = [4]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM(), pts, edges)

    def test_partition_unity_mpfa_multiple_boundary_intersecting_local_micro_fracture(
        self,
    ):
        g = create_grids.cart_2d()
        pts = np.array([[0.7, 0.9, 0.6, 1.6], [0.475, 1.4, 0.8, 0.8]])
        edges = np.array([[0, 2], [1, 3]])
        internal_nodes = [4]
        self._check_partition_unity(g, internal_nodes, Mpfa_DFM(), pts, edges)

    def test_partition_unity_tpfa_single_local_micro_fracture_full_cut(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.6, 1.4], [1.2, 1.8]])
        edges = np.array([[0], [1]])
        internal_faces = [4]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM(), pts, edges)

    def test_partition_unity_mpfa_single_local_micro_fracture_full_cut(self):
        g = create_grids.cart_2d()
        pts = np.array([[0.48, 0.9], [0.69, 0.5]])
        edges = np.array([[0], [1]])
        internal_nodes = [4]
        self._check_partition_unity(g, internal_nodes, Mpfa_DFM(), pts, edges)

    def test_partition_unity_tpfa_multiple_boundary_intersecting_local_micro_fracture_at_coarse_interface(
        self,
    ):
        g = create_grids.cart_2d()
        pts = np.array([[0.8, 1.2, 0.8, 1.2], [1.2, 1.8, 1.4, 1.6]])
        edges = np.array([[0, 2], [1, 3]])
        internal_faces = [4]
        self._check_partition_unity(g, internal_faces, Tpfa_DFM(), pts, edges)

    def test_transmissibility_tpfa_no_micro_fractures(self):
        g = create_grids.cart_2d()
        macro_bc = pp.BoundaryCondition(g)
        discr = Tpfa_DFM()
        internal_faces = [1, 4, 7, 11, 12, 13, 14]
        for reg in discr._interaction_regions(g):
            # consider only the internal faces
            if np.all(np.isin(reg.reg_ind, internal_faces)):
                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, cc_bc_values = lp.cell_basis_functions(
                    reg, local_gb, discr, {"bc": macro_bc}
                )

                _, _, trm = lp.compute_transmissibilies(
                    reg,
                    local_gb,
                    basis_functions,
                    cc_assembler,
                    cc_bc_values,
                    g,
                    discr,
                    {"bc": macro_bc},
                )

                self.assertTrue(np.allclose(np.abs(trm), 1))
                self.assertTrue(np.allclose(np.sum(trm), 0))

    def test_transmissibility_mpfa_no_micro_fractures(self):
        g = create_grids.cart_2d()
        discr = Mpfa_DFM()
        internal_nodes = [4, 7]
        macro_bc = pp.BoundaryCondition(g)
        for reg in discr._interaction_regions(g):
            # consider only the internal nodes
            if np.all(np.isin(reg.reg_ind, internal_nodes)):
                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, cc_bc_values = lp.cell_basis_functions(
                    reg, local_gb, discr, {"bc": macro_bc}
                )

                _, _, trm = lp.compute_transmissibilies(
                    reg,
                    local_gb,
                    basis_functions,
                    cc_assembler,
                    cc_bc_values,
                    g,
                    discr,
                    {"bc": macro_bc},
                )

                # @Eirik not sure the value here
                # EK: Me neither.
                # self.assertTrue(np.allclose(np.abs(trm), 1))
                self.assertTrue(np.allclose(np.sum(trm), 0))

    def _check_partition_unity(self, g, where, discr, pts=None, edges=None):
        macro_bc = pp.BoundaryCondition(g)
        for reg in discr._interaction_regions(g):
            # consider only the internal faces
            if np.all(np.isin(reg.reg_ind, where)):
                if pts is not None and edges is not None:
                    reg.add_fractures(points=pts, edges=edges)

                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, _ = lp.cell_basis_functions(
                    reg, local_gb, discr, {"bc": macro_bc}
                )

                # the assembler is the same for both
                assembler = next(iter(cc_assembler.values()))

                basis_sum = np.sum(
                    np.array([b for b in basis_functions.values()]), axis=0
                )
                # Check if the basis functions form a partition of unity.
                for g, _ in assembler.gb:
                    dof = assembler.dof_ind(g, discr.cell_variable)
                    self.assertTrue(np.allclose(basis_sum[dof], 1))

                # Check that the mortar fluxes sum to zero for local problems.
                for e, _ in assembler.gb.edges():
                    dof = assembler.dof_ind(e, discr.mortar_variable)
                    self.assertTrue(np.allclose(basis_sum[dof], 0))


if __name__ == "__main__":
    unittest.main()
