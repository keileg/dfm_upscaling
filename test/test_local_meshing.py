#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:37:26 2020

@author: eke001
"""

import numpy as np
import porepy as pp
import unittest

from dfm_upscaling import interaction_region as ia_reg
from dfm_upscaling.utils import create_grids


class TestMeshing(unittest.TestCase):
    def test_2d_internal_tpfa(self):
        g = create_grids.cart_2d()
        interior_face = 4

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        reg.mesh()

    def test_2d_boundary_tpfa(self):
        g = create_grids.cart_2d()
        boundary_face = 3

        reg = ia_reg.extract_tpfa_regions(g, faces=[boundary_face])[0]

        reg.mesh()

    def test_2d_internal_mpfa(self):
        g = create_grids.cart_2d()
        interior_node = 4

        reg = ia_reg.extract_mpfa_regions(g, nodes=[interior_node])[0]

        reg.mesh()

    def test_2d_boundary_mpfa(self):
        g = create_grids.cart_2d()
        boundary_node = 3

        reg = ia_reg.extract_mpfa_regions(g, nodes=[boundary_node])[0]

        reg.mesh()

    def test_3d_internal_tpfa(self):
        g = create_grids.cart_3d()
        interior_face = 4

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        reg.mesh()

    def test_3d_internal_mpfa(self):
        g = create_grids.cart_3d()
        interior_node = 13

        reg = ia_reg.extract_mpfa_regions(g, nodes=[interior_node])[0]

        reg.mesh()

    ### Tests with fractures

    def test_2d_internal_tpfa_with_fracture(self):
        g = create_grids.cart_2d()
        interior_face = 4

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]
        p = np.array([[0.7, 1.3], [1.5, 1.5]])
        edges = np.array([[0], [1]])

        reg.add_fractures(points=p, edges=edges)
        reg.mesh()

    def test_2d_boundary_tpfa_with_fracture(self):
        g = create_grids.cart_2d()
        face = 3

        reg = ia_reg.extract_tpfa_regions(g, faces=[face])[0]
        p = np.array([[0.0, 1.3], [1.1, 1.5]])
        edges = np.array([[0], [1]])

        reg.add_fractures(points=p, edges=edges)
        reg.mesh()

    def test_2d_internal_mpfa_with_fracture(self):
        g = create_grids.cart_2d()

        reg = ia_reg.extract_mpfa_regions(g, nodes=[4])[0]
        p = np.array([[0.7, 1.3], [1.0, 1.5]])
        edges = np.array([[0], [1]])

        reg.add_fractures(points=p, edges=edges)
        reg.mesh()

    def test_2d_boundary_mpfa_with_fracture(self):
        g = create_grids.cart_2d()

        reg = ia_reg.extract_mpfa_regions(g, nodes=[3])[0]
        p = np.array([[0.0, 1.3], [1.1, 1.5]])
        edges = np.array([[0], [1]])

        reg.add_fractures(points=p, edges=edges)
        reg.mesh()

    def test_3d_internal_tpfa_with_fracture(self):
        g = create_grids.cart_3d()
        interior_face = 4

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        frac = pp.Fracture(
            np.array([[0.7, 1.3, 1.3, 0.7], [1.5, 1.5, 1.5, 1.5], [0.2, 0.2, 0.8, 0.8]])
        )
        reg.add_fractures(fractures=[frac])
        reg.mesh()

    def test_3d_boundary_tpfa_with_fracture(self):
        g = create_grids.cart_3d()
        interior_face = 3

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        frac = pp.Fracture(
            np.array([[0.0, 0.3, 0.3, 0.0], [1.5, 1.5, 1.5, 1.5], [0.2, 0.2, 0.8, 0.8]])
        )
        reg.add_fractures(fractures=[frac])
        reg.mesh()

    def test_3d_internal_mpfa_with_fracture(self):
        g = create_grids.cart_3d()

        reg = ia_reg.extract_mpfa_regions(g, nodes=[13])[0]

        frac = pp.Fracture(
            np.array([[0.7, 1.3, 1.3, 0.7], [1.1, 1.1, 1.1, 1.1], [0.2, 0.2, 0.8, 0.8]])
        )
        reg.add_fractures(fractures=[frac])
        reg.mesh()

    def test_3d_boundary_mpfa_with_fracture(self):
        g = create_grids.cart_3d()

        reg = ia_reg.extract_mpfa_regions(g, nodes=[4])[0]

        frac = pp.Fracture(
            np.array([[0.0, 0.3, 0.3, 0.0], [1.1, 1.1, 1.1, 1.1], [0.2, 0.2, 0.8, 0.8]])
        )
        reg.add_fractures(fractures=[frac])
        reg.mesh()


if __name__ == "__main__":
    unittest.main()
