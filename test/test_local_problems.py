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
        discr = Tpfa_DFM()
        internal_faces = [1, 4, 7, 11, 12, 13, 14]
        for reg in discr._interaction_regions(g):
            # consider only the internal faces
            if np.all(np.isin(reg.reg_ind, internal_faces)):
                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, _ = lp.cell_basis_functions(
                    reg, local_gb, discr
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

    def test_partition_unit_mpfa_no_micro_fractures(self):
        g = create_grids.cart_2d()
        discr = Mpfa_DFM()
        internal_nodes = [4, 7]
        for reg in discr._interaction_regions(g):
            # consider only the internal nodes
            if np.all(np.isin(reg.reg_ind, internal_nodes)):
                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, _ = lp.cell_basis_functions(
                    reg, local_gb, discr
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

    def test_transmissibility_tpfa_no_micro_fractures(self):
        g = create_grids.cart_2d()
        discr = Tpfa_DFM()
        internal_faces = [1, 4, 7, 11, 12, 13, 14]
        for reg in discr._interaction_regions(g):
            # consider only the internal faces
            if np.all(np.isin(reg.reg_ind, internal_faces)):
                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, cc_bc_values = lp.cell_basis_functions(
                    reg, local_gb, discr
                )

                _, _, trm = lp.compute_transmissibilies(
                    reg, local_gb, basis_functions, cc_assembler, cc_bc_values, g, discr
                )

                self.assertTrue(np.allclose(np.abs(trm), 1))
                self.assertTrue(np.allclose(np.sum(trm), 0))

    def test_transmissibility_mpfa_no_micro_fractures(self):
        g = create_grids.cart_2d()
        discr = Mpfa_DFM()
        internal_nodes = [4, 7]
        for reg in discr._interaction_regions(g):
            # consider only the internal nodes
            if np.all(np.isin(reg.reg_ind, internal_nodes)):
                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, cc_bc_values = lp.cell_basis_functions(
                    reg, local_gb, discr
                )

                _, _, trm = lp.compute_transmissibilies(
                    reg, local_gb, basis_functions, cc_assembler, cc_bc_values, g, discr
                )

                # @Eirik not sure the value here
                # EK: Me neither.
                # self.assertTrue(np.allclose(np.abs(trm), 1))
                self.assertTrue(np.allclose(np.sum(trm), 0))

    def test_partition_unity_tpfa_local_micro_fractures(self):
        g = create_grids.cart_2d()
        p = np.array([[0.8, 1.2, 0.9, 1.2], [1.3, 1.6, 1.7, 1.4]])
        e = np.array([[0, 2], [1, 3]])
        e = np.array([[0], [1]])

        discr = Tpfa_DFM()
        internal_faces = [4]

        for reg in discr._interaction_regions(g):
            # consider only the internal faces
            if np.all(np.isin(reg.reg_ind, internal_faces)):
                reg.add_fractures(points=p, edges=e)

                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, _ = lp.cell_basis_functions(
                    reg, local_gb, discr
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

    def test_partition_unity_mpfa_local_micro_fractures(self):
        g = create_grids.cart_2d()
        p = np.array([[0.6, 1.4, 0.9, 1.2], [0.6, 1.1, 1.7, 1.4]])
        e = np.array([[0, 2], [1, 3]])
        e = np.array([[0], [1]])

        discr = Mpfa_DFM()
        internal_nodes = [4]

        for reg in discr._interaction_regions(g):
            # consider only the internal faces
            if np.all(np.isin(reg.reg_ind, internal_nodes)):
                reg.add_fractures(points=p, edges=e)

                local_gb = LocalGridBucketSet(g.dim, reg)
                local_gb.construct_local_buckets()

                basis_functions, cc_assembler, _ = lp.cell_basis_functions(
                    reg, local_gb, discr
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
