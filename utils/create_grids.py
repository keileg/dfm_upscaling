#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:52:38 2020

@author: eke001
"""
import numpy as np
import porepy as pp


def create_grids():
    g_list = [cart_2d(),
              cart_3d(),
              triang_grid(),
              tet_grid()]
    
    for g in g_list:
        yield g

def cart_2d(nx=None):
    if nx is None:
        nx = np.array([2, 3])
        
    g = pp.CartGrid(nx)
    g.compute_geometry()
    return g

def cart_3d(nx=None):
    if nx is None:
        nx = np.array([2, 2, 2])
    
    g = pp.CartGrid(nx)
    g.compute_geometry()
    return g

def triang_grid(nx=None):
    if nx is None:
        nx = np.array([2, 2])
    
    g = pp.StructuredTriangleGrid(nx)
    g.compute_geometry()
    return g

def tet_grid(nx=None):
    if nx is None:
        nx = np.array([2, 2, 2])
    
    g = pp.StructuredTetrahedralGrid(nx)
    g.compute_geometry()
    return g
