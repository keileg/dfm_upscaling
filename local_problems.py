#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 06:56:22 2020

@author: eke001
"""

import numpy as np
import porepy as pp

from scipy.sparse.linalg import spsolve






def transfer_bc(g_prev, v_prev, g_new, data_new):
    """
    Transfer the solution from one grid to boundary condition values on a grid of
    dimension one more

    Args:
        g_prev (TYPE): DESCRIPTION.
        v_prev (TYPE): DESCRIPTION.
        g_new (TYPE): DESCRIPTION.
        data_new (TYPE): DESCRIPTION.

    Returns:
        cell_ind (TYPE): DESCRIPTION.

    """
    # If the grids are not one dimension appart, there will be no match
    if g_prev.dim != (g_new.dim - 1):
        return []

    cc = g_prev.cell_centers
    fc = g_new.face_centers

    nc = g_prev.num_cells

    # Find coinciding cell centers and face centers.
    # Ideally, the comparison should have been made by g.global_point_ind, as this would
    # have made the calculations insensitive to geometric tolerances. However, since
    # the grids come from different grid buckets, that have undergone different
    # node splittings due to fractures, such a comparison is not feasible.
    # The computation of cell centers should be fairly robust, though.
    
    # IMPLEMENTATION NOTE: we can speed this up by calculating the distance from fc
    # to the plane with cc, and disregard all points with some distance
    # Find mapping to unique centers
    _, _, mapping = pp.utils.setmembership.unique_columns_tol(
        np.hstack((cc, fc)), tol=1e-13
    )

    num_occ = np.bincount(mapping)
    # If this brakes, the geometric tolerance is too loose
    assert num_occ.max() <= 2
    # Find coordinates repeated twice
    matches = np.where(num_occ == 2)[0]

    cell_ind = []

    bc_values = data_new[pp.PARAMETERS]["flow"]["bc_values"]

    # Loop over all matching pairs
    for ind in np.unique(matches):
        hit = np.where(mapping == ind)[0]
        # Split faces in g_new will have coinciding cell centers. Disregard these.
        if hit[0] >= nc:
            continue
        # One of the matches should be on the previous grid
        assert hit[1] >= nc

        # Assign boundary condition
        bc_values[hit[1] - nc] = v_prev[hit[0]]
        # Store the hit
        cell_ind.append(hit[0])

    return cell_ind


def cell_basis_functions(reg, local_gb, discr):
    """ 
    Calculate basis function related to coarse cells for an interaction region
    

    """

    # Identify cells in the interaction region with nodes in the fine-scale grid
    coarse_cell_ind, coarse_cell_cc =  reg.coarse_cell_centers()

    # Data structure to store Assembler of each grid bucket
    assembler_map = {}

    # Get a list of the local grid bucktes. This will be first 1d, then 2d (if dim == 3)
    # and then the real local gb
    bucket_list = local_gb.bucket_list()


    # Loop over all grid buckets: first type of gb (line, surface, 3d)
    for gb_set in bucket_list:
        # Then all gbs of this type
        for gb in gb_set:
            # Set parameters and discretization. 
            # The parameter definition should become much more general at some point
            # Also, it is now assumed that we use the same variable and parameter
            # keywords everywhere. This should be fixed
            discr.set_parameters_cell_basis(gb)
            discr.set_variables_discretizations(gb)

            # Create an Assembler and discretized the specified problem. The parameters
            # and type (not value) of boundary condition will be the same throughout the
            # calculations, thus discretization can be done once.
            assembler = pp.Assembler(gb)
            assembler.discretize()

            # Associate the assembler with this gb
            assembler_map[gb] = assembler

    # Then the basis function calculation. 
    # The structure of the calculations is: Loop over grid buckets with increasing
    # maximum dimension. Use values from buckets one dimension less as boundary
    # conditions. Solve for all buckets of this dimension, move on to higher dimension.
    
    basis_functions = {}
    
    # There is one basis function per coarse degree of freedom
    for coarse_ind, coarse_cc in zip(coarse_cell_ind, coarse_cell_cc.T):

        # Initialize the problem with a unit value in the fine-scale node corresponding
        # to the coarse cell center. Create a PointGrid to be compatible with the
        # data structures assumed below
        v = np.array([1])
        g = pp.PointGrid(coarse_cc.reshape((-1, 1)))
        g.compute_geometry()

        # The previous values, to be used as boundary conditions for the 1d problems.
        # Note that the value in the other coarse cell centers will be 0 by default.
        prev_values = [(g, v)]

        # Loop over all types of buckets
        # gb_set will first consist of 1d and 0d grids, then 2d-0d and finally
        # 3d-0d (if reg.dim == 3)
        for gb_set in bucket_list:
            # Data structure to store computed values, that will be used as boundary 
            # condition for the next gb_set
            new_prev_val = []
            # Loop over all buckets within this set.
            # if gb.dim_max() == 1, there will be one gb for each edge in the
            # interaction region, etc.
            for gb in gb_set:
                # Loop over the set of pressure values from the previous dimension
                # Find matches between previous cells and current faces, and assign
                # boundary conditions
                for g_prev, values in prev_values:
                    # Keep track of which cells in g_prev has been used to define bcs in
                    # this gb
                    found = np.zeros(g_prev.num_cells, dtype=np.bool)
                    
                    # Loop over all grids in gb, 
                    for g, d in gb:
                        # This will update the boundary condition values
                        cells_found = transfer_bc(g_prev, values, g, d)
                        found[cells_found] = True
                    
                    # Verify that either all values in the previous grid has been used
                    # as boundary conditions, or none have (the latter may happen in
                    # 3d problems).
                    # It is not 100% clear that this assertion is critical, but it seems
                    # likely, so we will need to debug if this is ever broken
                    assert np.all(found) or np.all(np.logical_not(found))

                # Get assembler                
                assembler = assembler_map[gb]
                # This will use the updated values for the boundary conditions
                A, b = assembler.assemble_matrix_rhs()
                
                # Solve and distribute
                x = spsolve(A, b)
                assembler.distribute_variable(x)

                for g, d in gb:
                    # Reset the boundary conditions in preparation for the next basis
                    # function
                    d[pp.PARAMETERS]["flow"]["bc_values"][:] = 0
                    # Store the pressure values to be used as new boundary conditions
                    # for the problem with one dimension more
                    new_prev_val.append((g, d[pp.STATE][discr.cell_variable]))

            # We are done with all buckets of this dimension. Redefine the current
            # values to previous values, and move on to the next set of buckets.
            prev_values = new_prev_val
            
        # Done with all calculations for this basis function. Store it.
        basis_functions[coarse_ind] = x
        # Move on to the next basis function
            
    # All done
    return basis_functions
    
