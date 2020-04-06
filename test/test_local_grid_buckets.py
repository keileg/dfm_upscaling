"""
Test creation of local grid buckets. Mainly check that the relevant functions do not
break.
"""
import numpy as np
import unittest
import porepy as pp

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
        
    def test_tpfa_internal_domain_2d_with_fractures(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[4])[0]
        
        # Two crossing fractures. One internal to the domain, one crosses the boundary
        p = np.array([[0.7, 1.3, 1.1, 1.5],
                      [1, 1, 0.9, 1.5]])
        e = np.array([[0, 2], [1, 3]])
        
        reg.add_fractures(points=p, edges=e)
        
        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_internal_domain_2d_with_fractures(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[4])[0]
        
        # Two crossing fractures. One internal to the domain, one crosses the boundary
        p = np.array([[0.7, 1.3, 1.1, 1.5],
                      [1.2, 1.2, 0.9, 1.7]])
        e = np.array([[0, 2], [1, 3]])
        
        reg.add_fractures(points=p, edges=e)
        
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
        
    def test_tpfa_boundary_domain_2d_with_fractures(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[3])[0]
        
        # Two crossing fractures. One internal to the domain, one crosses the boundary
        p = np.array([[-0.7, 1.3, 0.1, 0.5],
                      [1.2, 1.2, 0.9, 1.7]])
        e = np.array([[0, 2], [1, 3]])
        reg.add_fractures(points=p, edges=e)
        
        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_boundary_domain_2d_with_fractures(self):
        g = create_grids.cart_2d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[3])[0]
        # Two crossing fractures. One internal to the domain, one crosses the boundary
        p = np.array([[-0.7, 1.3, 0.1, 0.5],
                      [1.2, 1.2, 0.9, 1.7]])
        e = np.array([[0, 2], [1, 3]])
        reg.add_fractures(points=p, edges=e)
        
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

    def test_tpfa_internal_domain_3d_with_fractures(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[4])[0]
        
        f_1 = pp.Fracture(
            np.array([[0.7, 1.4, 1.4, 0.7], [0.5, 0.5, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]])
        )
        f_2 = pp.Fracture(
            np.array([[1.1, 1.1, 1.1, 1.1], [0.7, 1.4, 1.4, 0.7], [0.2, 0.2, 0.8, 0.8]])
        )
        reg.add_fractures(fractures=[f_1, f_2])
        
        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_internal_domain_3d_with_fractures(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[13])[0]
        
        f_1 = pp.Fracture(
            np.array([[0.7, 1.4, 1.4, 0.7], [0.4, 0.4, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]])
        )
        f_2 = pp.Fracture(
            np.array([[1.1, 1.1, 1.1, 1.1], [0.3, 1.4, 1.4, 0.3], [0.1, 0.1, 0.9, 0.9]])
        )
        reg.add_fractures(fractures=[f_1, f_2]) 
        
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

    def test_tpfa_boundary_domain_3d_with_fractures(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[3])[0]
        
        f_1 = pp.Fracture(
            np.array([[-0.7, 1.4, 1.4, -0.7], [0.4, 0.4, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]])
        )
        f_2 = pp.Fracture(
            np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 1.4, 1.4, 0.3], [0.1, 0.1, 0.9, 0.9]])
        )
        reg.add_fractures(fractures=[f_1, f_2])
        
        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()
        
    def test_mpfa_boundary_domain_3d_with_fractures(self):
        g = create_grids.cart_3d()
        reg = ia_reg.extract_mpfa_regions(g, nodes=[3])[0]

        f_1 = pp.Fracture(
            np.array([[-0.7, 1.4, 1.4, -0.7], [0.4, 0.4, 1.4, 1.4], [0.6, 0.6, 0.6, 0.6]])
        )
        f_2 = pp.Fracture(
            np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 1.4, 1.4, 0.3], [0.1, 0.1, 0.9, 0.9]])
        )
        reg.add_fractures(fractures=[f_1, f_2])
        
        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()        

        
if __name__ == '__main__':
    TestGridBucket().test_mpfa_internal_domain_3d_with_fractures()
    unittest.main()
