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
    
if __name__ == '__main__':
    TestMeshing().test_2d_internal_tpfa()