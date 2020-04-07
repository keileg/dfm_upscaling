#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 06:56:22 2020

@author: eke001
"""

import numpy as np
import porepy as pp
import meshio

from scipy.sparse.linalg import spsolve
from porepy.grids.constants import GmshConstants
from porepy.grids.gmsh import mesh_2_grid


def transfer_bc(g_prev, v_prev, g_new, bc_values, dim):
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

    # Find coinciding cell centers and face centers.
    cell_ind = []

    for pair in match_points_on_surface(cc, fc, dim, g_prev.dim):
        # Assign boundary condition
        bc_values[pair[1]] = v_prev[pair[0]]
        # Store the hit
        cell_ind.append(pair[0])

    return cell_ind


def match_points_on_surface(sp, p, spatial_dim, dim_of_sp, tol=1e-10):
    """
    Match points in two point sets by coordinates. One of the point sets is assumed to
    reside on a surface, or a line.

    Args:
        sp (np.ndarray, 3 x n): Point set on a surface.
        p (np.ndarray, 3 x n): Point set, to be compared with the surface points.
        spatial_dim (int): Spatial dimension of the point cloud. Should be 2 or 3.
        dim_of_sp (int): Dimension of the geometric object of sp.

        tol (double, optional): Geometric tolerance in point comparison. Defaults to
            1e-10.

    Returns:
        pairs (list of tuples): Each list member contains a 2-tuple, that identifies a
            match, by column indices, between a point in sp and p.

    """

    # Ideally, the comparison should have been made by g.global_point_ind, as this would
    # have made the calculations insensitive to geometric tolerances. However, since
    # the grids come from different grid buckets, that have undergone different
    # node splittings due to fractures, such a comparison is not feasible.
    # The computation of coordinates should be fairly robust, though.

    # To speed up the calculations, we first identify points that lay in the same plane
    # or on the same line as the surface points

    num_surf_pts = sp.shape[1]

    # Center point on the surface, and vector to all points in p
    cp = sp.mean(axis=1).reshape((-1, 1))
    vec_cp_p = p - cp

    # If there are few points in p, or too few points in sp to define a normal direction
    # we will simply compare all points.
    if p.shape[1] < 2 or sp.shape[1] < spatial_dim:
        in_plane = np.arange(p.shape[1])

    # If the surface is of co-dimension 2 (will be a line in 3d), we project the points
    # to the line
    elif dim_of_sp < spatial_dim - 1:
        # Center point on the line
        ind = np.argmax(np.sum(np.abs(sp - cp), axis=1))

        # Normalized vector along the line
        vec_on_line = (
            sp[:spatial_dim, ind].reshape((-1, 1)) - cp[:spatial_dim]
        ).reshape((-1, 1))
        vec_on_line /= np.linalg.norm(vec_on_line)

        # The norm of the vector from cp to the point cloud
        nrm = np.linalg.norm(vec_cp_p, axis=0)
        # Special treatment if a point in the cloud is almost on cp
        not_on_cp = np.where(nrm > tol)[0]
        vec_cp_p[:, not_on_cp] /= nrm[not_on_cp]

        # Points on the line - although we call it a plane to be consistent with below
        in_plane = np.where(
            np.logical_or(
                np.logical_not(not_on_cp),
                np.abs(np.sum(vec_on_line * vec_cp_p, axis=0)) < tol,
            )
        )[0]

    # Here we will find the normal vector, and find points in the plane by a dot product
    else:
        # Normal vector in 2d, the construction is somewhat elaborate
        if spatial_dim == 2:
            ind = np.argmax(np.sum(np.abs(sp - cp), axis=1))

            v = sp[:spatial_dim, ind].reshape((-1, 1)) - cp[:spatial_dim]
            n = np.array([v[1], -v[0]])

        # In 3d, we can use a pp function to get the normal vector
        else:
            n = pp.map_geometry.compute_normal(sp).reshape((-1, 1))

        # Index of points in the plane
        in_plane = np.where(np.abs(np.sum(vec_cp_p[:spatial_dim] * n, axis=0)) < tol)[0]

    # Restrict point cloud to the plane
    p_in_plane = p[:, in_plane]

    # Bookeeping
    np_in_plane = in_plane.size
    if np_in_plane == 1:
        p_in_plane = p_in_plane.reshape((-1, 1))

    # Intersection of coordinates
    _, _, mapping = pp.utils.setmembership.unique_columns_tol(
        np.hstack((sp, p_in_plane)), tol=tol
    )

    num_occ = np.bincount(mapping)
    # If this brakes, the geometric tolerance is too loose
    assert num_occ.max() <= 2
    # Find coordinates repeated twice
    matches = np.where(num_occ == 2)[0]

    pairs = []

    # Loop over all matching pairs
    for ind in np.unique(matches):
        hit = np.where(mapping == ind)[0]
        # Split faces in g_new will have coinciding cell centers. Disregard these.
        if hit[0] >= num_surf_pts:
            continue
        # One of the matches should be on the previous grid
        # If we get an error here, it is likely because two fractures cross at an
        # auxiliary surface (at least this is the likely cause at the time of writing)
        assert hit[1] >= num_surf_pts

        pairs.append([hit[0], in_plane[hit[1] - num_surf_pts]])

    return pairs


def cell_basis_functions(reg, local_gb, discr):
    """
    Calculate basis function related to coarse cells for an interaction region


    """

    # Identify cells in the interaction region with nodes in the fine-scale grid
    coarse_cell_ind, coarse_cell_cc = reg.coarse_cell_centers()

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
            discr.set_variables_discretizations_cell_basis(gb)

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
    coarse_gb = {}
    coarse_assembler = {}

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
                        bc_values = d[pp.PARAMETERS][discr.keyword]["bc_values"]
                        cells_found = transfer_bc(g_prev, values, g, bc_values, reg.dim)
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

        coarse_gb[coarse_ind] = gb
        coarse_assembler[coarse_ind] = assembler
        # Move on to the next basis function

    # All done
    # Check if the basis functions form a partition of unity.
    # NOTE: This makes the tacit assumption that the ordering of the grids is the same
    # in all assemblers. This is probably true, but it should be the first item to
    # check if we get an error message here
    basis_sum = np.sum(np.array([b for b in basis_functions.values()]), axis=0)
    for g, _ in assembler.gb:
        dof = assembler.dof_ind(g, discr.cell_variable)
        assert np.allclose(basis_sum[dof], 1)

    # Check that the mortar fluxes sum to zero for local problems.
    for e, _ in assembler.gb.edges():
        dof = assembler.dof_ind(e, discr.mortar_variable)
        assert np.allclose(basis_sum[dof], 0)

    return basis_functions, coarse_assembler


def compute_transmissibilies(
    reg, local_gb, basis_functions, cc_assembler, coarse_grid, discr
):
    # Read the mesh file for the micro problem
    mesh = meshio.read(local_gb.file_name + ".msh")

    # Invert the meshio field_data so that phys_names maps from the tags that gmsh
    # assigns to XXX, to the physical names
    phys_names = {v[0]: k for k, v in mesh.field_data.items()}

    # Mesh points
    pts = mesh.points

    gmsh_constants = GmshConstants()

    # Data structures for storing cell and face index for the local problems, together
    # with the corresponding transmissibilities.
    coarse_cell_ind, coarse_face_ind, trm = [], [], []

    # The macro transmissibilities can be recovered from the fluxes over those micro
    # faces that coincide with a macro face.
    # To identify these micro faces, we recover the micro surface grids. These were
    # present as internal constraints in the local grid bucket.
    if coarse_grid.dim == 2:
        # Create all 2d grids that correspond to an auxiliary surface
        g_surf, _ = mesh_2_grid.create_1d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            line_tag=gmsh_constants.PHYSICAL_NAME_AUXILIARY,
        )
    else:
        # EK: 3d domains have not been tested.
        # Create all 2d grids that correspond to a domain boundary
        g_surf = mesh_2_grid.create_2d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            network=local_gb.network,
            is_embedded=True,
            surface_tag=gmsh_constants.PHYSICAL_NAME_AUXILIARY,
        )

    # Loop over all created surface grids,
    for gi, gs in enumerate(g_surf):

        # Cell and node coordinates.
        # EK: Not sure if we need all of this, or more
        gs.compute_geometry()
        cc = gs.cell_centers
        nc = gs.nodes

        # There should be a single face among the constraint nodes (both in 2d and 3d)
        if reg.name == "mpfa":
            face_node = np.where([t == "face" for t in reg.constraint_node_type[gi]])[0]
            assert face_node.size == 1
            # Macro face index
            cfi = reg.constraints[gi][face_node[0]]
        else:  # tpfa
            # For tpfa, the face is identified by the region number
            cfi = reg.reg_ind

        # Macro normal vector of the face
        coarse_normal = coarse_grid.face_normals[:, cfi]

        # Loop over all basis functions constructed by the local problem.
        for cci, basis in basis_functions.items():

            # Get the assembler used for the construction of the basis function for
            # this coarse cell. Distribute the solution.
            cc_assembler[cci].distribute_variable(basis)

            # Grid bucket of this local problem.
            # This will be the full Nd grid bucket.
            gb = cc_assembler[cci].gb

            # Reconstruct fluxes in the grid bucket
            pp.fvutils.compute_darcy_flux(
                gb, p_name=discr.cell_variable, lam_name=discr.mortar_variable
            )

            # Loop over all grids in the grid_bucket.
            for loc_g, d in gb:

                # Flux field for this problem
                full_flux = d[pp.PARAMETERS][discr.keyword]["darcy_flux"]

                if loc_g.dim == gs.dim + 1:
                    # find faces in the matrix grid on the surface
                    grid_map = match_points_on_surface(
                        cc, loc_g.face_centers, coarse_grid.dim, gs.dim
                    )
                elif loc_g.dim == gs.dim:
                    # find fractures that intersect with the surface
                    # If we get an error message from this call, about more than one
                    # hit in the second argument (p), chances are that a fracture is
                    # intersecting at the auxiliary surface

                    # EK: Something goes wrong in this case (fracture crosses the
                    # surface), though not necessarily in this function call. No idea
                    # what is wrong.
                    grid_map = match_points_on_surface(
                        nc, loc_g.face_centers, coarse_grid.dim, gs.dim
                    )
                else:
                    # EK: Not sure what to do here
                    # We may drop this alternative for a while, but it will in effect
                    # correspond to basis functions that are not a partition of unity.
                    raise NotImplementedError("Have not gotten this far")

                # If we found any matches, loop over all micro faces, sum the fluxes,
                # possibly with an adjustment of the flux direction.
                if len(grid_map) > 0:
                    surface_flux = []
                    for fi in grid_map:
                        loc_flux = full_flux[fi[1]]
                        # If the micro and macro normal vectors point in different
                        # directions, we should switch the flux
                        fine_normal = loc_g.face_normals[:, fi[1]]
                        sgn = np.sign(fine_normal.dot(coarse_normal))
                        surface_flux.append(loc_flux * sgn)

                    # Store the macro cell and face index, together with the
                    # transmissibility
                    coarse_cell_ind.append(cci)
                    coarse_face_ind.append(cfi)
                    trm.append(np.asarray(surface_flux).sum())
                    if loc_g.dim == gs.dim:
                        # EK: This is roughly where I'm looking for errors right now
                        print(surface_flux)
                        #   pdb.set_trace()
                        debug = []

    # The macro transmissibilities should sum to zero to preserve no flow for constant
    # pressure
    trm_sum = np.bincount(coarse_face_ind, weights=trm)

    assert np.allclose(trm_sum, 0)

    return coarse_cell_ind, coarse_face_ind, trm
