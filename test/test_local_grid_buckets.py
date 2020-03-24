"""
Test creation of local grid buckets. Mainly check that the relevant functions do not
break.
"""


import numpy as np
import unittest

from dfm_upscaling.utils import create_grids
from dfm_upscaling import interaction_region as ia_reg
from dfm_upscaling.local_grid_bucket import LocalGridBucketSet
from dfm_upscaling.test import test_utils



class TestGridBucket(unittest.TestCase):
    
    def test_tpfa_internal_domain_2d(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[4])[0]
        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_internal_domain_2d(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[4])[0]
        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()

    def test_tpfa_boundary_domain_2d(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[3])[0]
        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_boundary_domain_2d(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[3])[0]
        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()
        
    def test_tpfa_internal_domain_3d(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[4])[0]
        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_internal_domain_3d(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[13])[0]
        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()        
        
    def test_tpfa_boundary_domain_3d(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[3])[0]
        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_boundary_domain_3d(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[3])[0]
        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()        
        
if __name__ == '__main__':
    unittest.main()
