#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:03:51 2020

@author: eke001
"""

import numpy as np
import unittest

from dfm_upscaling.utils import create_grids
from dfm_upscaling import interaction_region as ia_reg
from dfm_upscaling.test import test_utils


class TestRegions(unittest.TestCase):
    def test_tpfa_cart_grid_2d_interior_face(self):

        g = create_grids.cart_2d()

        interior_face = 4
        cells = [2, 3]

        fn = g.face_nodes.indices.reshape((2, -1), order="f")
        fn_loc = fn[:, interior_face]

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0]],
                [cells[0], fn_loc[1]],
                [cells[1], fn_loc[0]],
                [cells[1], fn_loc[1]],
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node", "cell"))

    def test_tpfa_cart_grid_2d_boundary_face(self):

        g = create_grids.cart_2d()

        boundary_face = 3
        cells = [2]

        reg = ia_reg.extract_tpfa_regions(g, faces=[boundary_face])[0]

        fn = g.face_nodes.indices.reshape((2, -1), order="f")

        fn_loc = fn[:, boundary_face]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0]],
                [cells[0], fn_loc[1]],
                [fn_loc[0], fn_loc[1]],  # boundary surface
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node"))

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if np.all(np.isin(surf, fn_loc)):
                # This is a boundary surface
                self.assertTrue(bound_info == True)
                self.assertTrue(node_types == ("node", "node"))
            else:
                self.assertTrue(bound_info == False)
                self.assertTrue(node_types == ("cell", "node"))

    def test_tpfa_cart_grid_3d_interior_face(self):

        g = create_grids.cart_3d()

        interior_face = 4
        cells = [2, 3]

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        fn = g.face_nodes.indices.reshape((4, -1), order="f")

        fn_loc = fn[:, interior_face]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0], fn_loc[1]],
                [cells[0], fn_loc[1], fn_loc[2]],
                [cells[0], fn_loc[2], fn_loc[3]],
                [cells[0], fn_loc[3], fn_loc[0]],
                [cells[1], fn_loc[0], fn_loc[1]],
                [cells[1], fn_loc[1], fn_loc[2]],
                [cells[1], fn_loc[2], fn_loc[3]],
                [cells[1], fn_loc[3], fn_loc[0]],
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        known_edges = np.array(
            [
                [cells[0], fn_loc[0], cells[1]],
                [cells[0], fn_loc[1], cells[1]],
                [cells[0], fn_loc[2], cells[1]],
                [cells[0], fn_loc[3], cells[1]],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(known_edges.T, reg.edges.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node", "cell"))

    def test_tpfa_cart_grid_3d_boundary_face(self):

        g = create_grids.cart_3d()

        interior_face = 3
        cells = [2]

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        fn = g.face_nodes.indices.reshape((4, -1), order="f")

        fn_loc = fn[:, interior_face]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0], fn_loc[1]],
                [cells[0], fn_loc[1], fn_loc[2]],
                [cells[0], fn_loc[2], fn_loc[3]],
                [cells[0], fn_loc[3], fn_loc[0]],
                [fn_loc[0], fn_loc[1], fn_loc[2]],
                [fn_loc[0], fn_loc[2], fn_loc[3]],
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        known_edges = np.array(
            [
                [cells[0], fn_loc[0]],
                [cells[0], fn_loc[1]],
                [cells[0], fn_loc[2]],
                [cells[0], fn_loc[3]],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(known_edges.T, reg.edges.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node"))

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if np.all(np.isin(surf, fn_loc)):
                # This is a boundary surface
                self.assertTrue(bound_info == True)
                self.assertTrue(node_types == ("node", "node", "node"))
            else:
                self.assertTrue(bound_info == False)
                self.assertTrue(node_types == ("cell", "node", "node"))

    def test_mpfa_cart_grid_2d_interior_node(self):
        g = create_grids.cart_2d()

        interior_node = 4
        # cells = [0, 1, 2, 3]
        # faces = [1, 4, 11, 12]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_edges = np.array([[0, 11, 2], [0, 1, 1], [1, 12, 3], [2, 4, 3]])

        known_surfaces = np.array(
            [[0, 1], [1, 1], [0, 11], [2, 11], [2, 4], [3, 4], [3, 12], [1, 12]]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))
        self.assertTrue(test_utils.compare_arrays(known_edges.T, reg.edges.T))

    def test_mpfa_cart_grid_2d_boundary_node(self):
        g = create_grids.cart_2d()

        interior_node = 3
        cells = [0, 2]
        faces = [0, 3, 11]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_edges = np.array([[0, 0], [0, 11, 2], [2, 3]])

        known_surfaces = np.array([[0, 0], [0, 11], [2, 11], [2, 3], [0, 3], [3, 3]])

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))
        self.assertTrue(
            test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)
        )

        for i in range(reg.edges.size):
            if len(reg.edges[i]) == 2:
                self.assertTrue(reg.edge_node_type[i] == ("cell", "face"))
            if len(reg.edges[i]) == 3:
                self.assertTrue(reg.edge_node_type[i] == ("cell", "face", "cell"))

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if "node" in node_types:
                # This is a boundary surface
                self.assertTrue(bound_info == True)
                self.assertTrue(node_types == ("face", "node"))
                self.assertTrue(surf[-1] == interior_node)
                self.assertTrue(np.any(surf[0] == faces))
            else:
                self.assertTrue(bound_info == False)
                self.assertTrue(node_types == ("cell", "face"))

    def test_mpfa_cart_grid_3d_internal_node(self):

        g = create_grids.cart_3d()
        interior_node = 13
        # cells = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # faces = [1, 4, 7, 10, 14, 15, 20, 21, 28, 29, 30, 31]
        # other_nodes = [4, 10, 12, 14, 16, 22]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_surfaces = np.asarray(
            [
                [0, 1, 4],
                [0, 1, 10],
                [1, 1, 4],
                [1, 1, 10],
                [0, 14, 4],
                [0, 14, 12],
                [2, 14, 4],
                [2, 14, 12],
                [2, 4, 4],
                [2, 4, 16],
                [3, 4, 4],
                [3, 4, 16],
                [1, 15, 4],
                [1, 15, 14],
                [3, 15, 4],
                [3, 15, 14],  # Done with all lower faces
                [4, 7, 10],
                [4, 7, 22],
                [5, 7, 10],
                [5, 7, 22],
                [4, 20, 12],
                [4, 20, 22],
                [6, 20, 12],
                [6, 20, 22],
                [5, 21, 14],
                [5, 21, 22],
                [7, 21, 14],
                [7, 21, 22],
                [6, 10, 16],
                [6, 10, 22],
                [7, 10, 16],
                [7, 10, 22],  # Done with all upper faces
                [0, 28, 10],
                [0, 28, 12],
                [4, 28, 10],
                [4, 28, 12],
                [1, 29, 10],
                [1, 29, 14],
                [5, 29, 10],
                [5, 29, 14],
                [2, 30, 12],
                [2, 30, 16],
                [6, 30, 12],
                [6, 30, 16],
                [3, 31, 14],
                [3, 31, 16],
                [7, 31, 14],
                [7, 31, 16],
            ]
        )

        known_edges = np.array(
            [
                [0, 1, 1],
                [0, 14, 2],
                [2, 4, 3],
                [1, 15, 3],
                [4, 7, 5],
                [4, 20, 6],
                [5, 21, 7],
                [6, 10, 7],
                [0, 28, 4],
                [1, 29, 5],
                [2, 30, 6],
                [3, 31, 7],
            ]
        )

        self.assertTrue(
            test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T, sort=False)
        )
        self.assertTrue(
            test_utils.compare_arrays(known_edges.T, reg.edges.T, sort=False)
        )

    def test_mpfa_cart_grid_3d_boundary_node(self):
        g = create_grids.cart_3d()
        interior_node = 3
        # cells = [0, 2]
        # faces = [0, 3, 14, 24, 26]
        # other_nodes = [0, 4, 6, 12]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_edges = np.array([[0, 0], [0, 24], [0, 14, 2], [2, 3], [2, 26]])

        known_surfaces = np.array(
            [
                [0, 0, 0],
                [0, 24, 0],
                [0, 24, 4],
                [0, 14, 4],
                [2, 26, 4],
                [2, 14, 4],
                [0, 0, 12],
                [0, 14, 12],
                [2, 14, 12],
                [2, 3, 12],
                [2, 3, 6],
                [2, 26, 6],  # after this, boundary surfaces
                [0, 0, 3],
                [0, 12, 3],
                [3, 6, 3],
                [3, 12, 3],
                [24, 0, 3],
                [24, 4, 3],
                [26, 4, 3],
                [26, 6, 3],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))
        self.assertTrue(
            test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)
        )

    def test_regions_different_grids(self):
        # Run through all prepared simple grids, check that the creation does not break
        for g in create_grids.create_grids():
            _ = ia_reg.extract_tpfa_regions(g)
            _ = ia_reg.extract_mpfa_regions(g)


class TestFindEdges(unittest.TestCase):
    def test_cart_grid_3d_internal(self):

        g = create_grids.cart_3d()
        node = 13
        faces = g.face_nodes.tocsr()[13].indices

        ia_reg._find_edges(g, node, faces)


if __name__ == "__main__":
    TestRegions().test_regions_different_grids()
    unittest.main()
