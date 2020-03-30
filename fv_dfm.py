#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 08:39:20 2020

@author: eke001
"""

import numpy as np
import porepy as pp

from dfm_upscaling import interaction_region as ia_reg
from dfm_upscaling import local_problems
from dfm_upscaling.local_grid_bucket import LocalGridBucketSet


class FVDFM(pp.FVElliptic):
    
    def __init__(self, keyword="flow"):
        super(FVDFM, self).__init__(keyword)
        self.cell_variable = "pressure"
        
        self.mortar_variable = "mortar_flux"
        
    def set_parameters_cell_basis(self, gb):
        """
        Assign parameters. Very simple for now, this must be improved.
    
        Args:
            gb (TYPE): DESCRIPTION.
    
        Returns:
            None.
    
        """
    
        Nd = gb.dim_max()
    
        keyword = "flow"
    
        # First initialize data
        for g, d in gb:
    
            param = {}
    
            if g.dim == Nd:
                domain_boundary = np.logical_and(
                    g.tags["domain_boundary_faces"],
                    np.logical_not(g.tags["fracture_faces"]),
                )
    
                boundary_faces = np.where(domain_boundary)[0]
                bc_type = boundary_faces.size * ["dir"]
    
                bc = pp.BoundaryCondition(g, boundary_faces, bc_type)
                param["bc"] = bc
    
            pp.initialize_default_data(g, d, keyword, param)
    
        for e, d in gb.edges():
            mg = d["mortar_grid"]
    
            g1, g2 = gb.nodes_of_edge(e)
    
            param = {}
    
            if g1.from_fracture:
                param["normal_diffusivity"] = 1e1
    
            pp.initialize_data(mg, d, keyword, param)
    

    def set_variables_discretizations(self, gb):
        """
        Assign variables, and set discretizations. 
        
        NOTE: keywords and variable names are hardcoded here. This should be centralized.
    
        Args:
            gb (TYPE): DESCRIPTION.
    
        Returns:
            None.
    
        """
        mpfa = pp.Mpfa(self.keyword)
    
        robin = pp.RobinCoupling(self.keyword, mpfa, mpfa)
        continuity = pp.FluxPressureContinuity(self.keyword, mpfa, mpfa)
    
    
        diffusion_term_flow = "flow"
    
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {self.cell_variable: {"cells": 1, "faces": 0}}
            d[pp.DISCRETIZATION] = {self.cell_variable: {diffusion_term_flow: mpfa}}
        # Loop over the edges in the GridBucket, define primary variables and discretizations
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {self.mortar_variable: {"cells": 1}}
    
            # The type of lower-dimensional discretization depends on whether this is a
            # (part of a) fracture, or a transition between two line or surface grids.
            if g1.from_fracture:
                edge_discretization = robin
            else:
                edge_discretization = continuity
    
            d[pp.COUPLING_DISCRETIZATION] = {
                "coupling_flux": {
                    g1: (self.cell_variable, diffusion_term_flow),
                    g2: (self.cell_variable, diffusion_term_flow),
                    e: (self.mortar_variable, edge_discretization),
                }
            }
            
    def discretize(self, g, d):
        
        param = d[pp.PARAMETERS][self.keyword]
        
        ia_type = param['interaction_region_type']
        
        # This for-loop could be parallelized
        for reg in self._interaction_regions(g, ia_type):
            
            gb_set = LocalGridBucketSet(g.dim, reg)
            
            # First basis functions for local problems
            basis = local_problems.cell_basis_functions(reg, gb_set, self)
            
            # Call method to transfer basis functions to transmissibilties over coarse
            # edges
        
        
        
    def _interaction_regions(self, g, method):
        
        
        if method == 'tpfa':
            for fi in range(g.num_faces):
                yield ia_reg.extract_tpfa_regions(g, fi)[0]
                
        elif method == 'mpfa':
            for ni in range(g.num_nodes):
                yield ia_reg.extract_mpfa_regions(g, ni)[0]
                
        else: 
            raise ValueError(f"unknown interaction region type {method}")
        
        
