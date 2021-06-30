#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 06:56:22 2020

@author: eke001
"""

import numpy as np
import porepy as pp
from collections import namedtuple
from typing import List, Tuple, Dict, Union, Optional


import scipy.sparse as sps
import scipy.sparse.linalg as spla
from porepy.fracs import msh_2_grid
from porepy.fracs.gmsh_interface import Tags, PhysicalNames
from porepy.fracs import simplex

from .interaction_region import InteractionRegion
from .local_grid_bucket import LocalGridBucketSet


def transfer_bc(
    g_prev,
    v_prev,
    g_new,
    bc_values,
    dim,
    cell_ind: Optional[List[int]] = None,
    face_ind: Optional[List[int]] = None,
):
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
        return [], []

    cc = g_prev.cell_centers
    fc = g_new.face_centers

    # Find coinciding cell centers and face centers.
    if cell_ind is None or face_ind is None:
        cell_ind = []
        face_ind = []
        for pair in _match_points_on_surface(cc, fc, dim, g_prev.dim, g_prev.nodes):
            cell_ind.append(pair[0])
            face_ind.append(pair[1])

    for ci, fi in zip(cell_ind, face_ind):
        # Assign boundary condition
        bc_values[fi] = v_prev[ci]
        # Store the hit

    return cell_ind, face_ind


def _match_points_on_surface(
    sp, p, spatial_dim, dim_of_sp, supporting_points, tol=1e-8
):
    """
    Match points in two point sets by coordinates. One of the point sets is assumed to
    reside on a surface, or a line.

    Args:
        sp (np.ndarray, 3 x n): Point set on a surface.
        p (np.ndarray, 3 x n): Point set, to be compared with the surface points.
        spatial_dim (int): Spatial dimension of the point cloud. Should be 2 or 3.
        dim_of_sp (int): Dimension of the geometric object of sp.
        supporting_points (np.ndarray, 3 x n): Additional points on the surface.
            Will not look for duplicates of these, but they are used to define a normal
            vector for the surface.
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

    if sp.shape[1] < spatial_dim:
        all_surface_points = np.hstack((sp, supporting_points))
    else:
        # Sufficient number of points to construct the surface vector
        all_surface_points = sp

    # Center point on the surface, and vector to all points in p
    cp = sp.mean(axis=1).reshape((-1, 1))
    vec_cp_p = p - cp

    if p.shape[1] < 2 or all_surface_points.shape[1] < spatial_dim:
        # If there are few points in p we will simply compare all points.
        in_plane = np.arange(p.shape[1])

    elif dim_of_sp < spatial_dim - 1:
        # If the surface is of co-dimension 2 (will be a line in 3d), we project the points
        # to the line
        # Point furthest away from the center point
        ind = np.argmax(np.sum(np.abs(all_surface_points - cp), axis=0))

        # Normalized vector along the line
        vec_on_line = (
            all_surface_points[:spatial_dim, ind].reshape((-1, 1)) - cp[:spatial_dim]
        ).reshape((-1, 1))
        vec_on_line /= np.linalg.norm(vec_on_line)

        # The norm of the vector from cp to the point cloud
        nrm = np.linalg.norm(vec_cp_p, axis=0)
        # Special treatment if a point in the cloud is almost on cp
        not_on_cp = np.where(nrm > tol)[0]
        vec_cp_p[:, not_on_cp] /= nrm[not_on_cp]

        cross = np.array(
            [
                vec_on_line[1] * vec_cp_p[2] - vec_on_line[2] * vec_cp_p[1],
                vec_on_line[2] * vec_cp_p[0] - vec_on_line[0] * vec_cp_p[2],
                vec_on_line[0] * vec_cp_p[1] - vec_on_line[1] * vec_cp_p[0],
            ]
        )

        # Points on the line - although we call it a plane to be consistent with below
        in_plane = np.where(np.sum(np.abs(cross), axis=0) < tol)[0]

    else:
        # Here we will find the normal vector, and find points in the plane by a dot product
        if spatial_dim == 2:
            # Normal vector in 2d, the construction is somewhat elaborate
            ind = np.argmax(np.sum(np.abs(all_surface_points - cp), axis=0))

            v = (
                all_surface_points[:spatial_dim, ind].reshape((-1, 1))
                - cp[:spatial_dim]
            )
            assert np.linalg.norm(v) > tol
            n = np.array([v[1], -v[0]])

        else:
            # In 3d, we can use a pp function to get the normal vector
            n = pp.map_geometry.compute_normal(all_surface_points, tol=tol).reshape(
                (-1, 1)
            )

        # Index of points in the plane
        assert np.all(np.isfinite(n))
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
    # Find coordinates repeated twice or more
    matches = np.where(num_occ >= 2)[0]

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
        assert np.all(hit[1:] >= num_surf_pts)

        # Store all the possible pairs that are matching at the interface
        for h in hit[1:]:
            pairs.append([hit[0], in_plane[h - num_surf_pts]])

    return pairs


def cell_basis_functions(
    reg: InteractionRegion,
    local_gb: LocalGridBucketSet,
    discr: pp.FVElliptic,
    macro_data: Dict,
):
    """
    Calculate basis function related to coarse cells for an interaction region.



    """
    # Identify cells in the interaction region with nodes in the fine-scale grid
    coarse_cell_ind, coarse_cell_cc = reg.coarse_cell_centers()

    # Data structure to store Assembler of each grid bucket
    assembler_map = {}

    # Get a list of the local grid bucktes. This will be first 1d, then 2d (if dim == 3)
    # and then the real local gb
    bucket_list = local_gb.bucket_list()

    ilu_map = {}
    ilu_threshold = 10000

    # Loop over all grid buckets: first type of gb (line, surface, 3d)
    for gb_set in bucket_list:
        # Then all gbs of this type
        for gb in gb_set:
            # Set parameters and discretization.
            # The parameter definition should become much more general at some point
            # Also, it is now assumed that we use the same variable and parameter
            # keywords everywhere. This should be fixed
            discr.set_parameters_cell_basis(gb, macro_data)
            discr.set_variables_discretizations_cell_basis(gb)

            # Create an Assembler and discretized the specified problem. The parameters
            # and type (not value) of boundary condition will be the same throughout the
            # calculations, thus discretization can be done once.
            assembler = pp.Assembler(gb)
            assembler.discretize()

            A, _ = assembler.assemble_matrix_rhs(only_matrix=True)
            if A.shape[0] > ilu_threshold:
                ilu = spla.spilu(A)
                print("ILU done")
                Mx = lambda v: ilu.solve(v)
                M = spla.LinearOperator(A.shape, Mx)
                ilu_map[gb] = M

            # Associate the assembler with this gb
            assembler_map[gb] = (assembler, A)

    # Then the basis function calculation.
    # The structure of the calculations is: Loop over grid buckets with increasing
    # maximum dimension. Use values from buckets one dimension less as boundary
    # conditions. Solve for all buckets of this dimension, move on to higher dimension.

    basis_functions = {}
    coarse_gb = {}
    coarse_assembler = {}
    coarse_bc_values = {}

    # Store the sum of basis functions for all subproblems. Useful to ensure partition
    # of unity for the boundary problems in addition to the main check for the basis
    # functions.
    debug_pou_map = {}
    debug_bc_val_map = {}
    for gb_set in bucket_list:
        for gb in gb_set:
            for g, _ in gb:
                debug_pou_map[g] = np.zeros(g.num_cells)
                debug_bc_val_map[g] = np.zeros(g.num_faces)
            for e, d in gb.edges():
                mg = d["mortar_grid"]
                debug_pou_map[e] = np.zeros(mg.num_cells)

    # Make lists of all 1d and 2d grids. Useful for debugging.
    g1 = [g for g in debug_pou_map if isinstance(g, pp.Grid) and g.dim == 1]
    g2 = [g for g in debug_pou_map if isinstance(g, pp.Grid) and g.dim == 2]

    cell_face_relations: Dict[Tuple[pp.Grid, pp.Grid], Tuple[List, List]] = {}

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
        # Also keep track of the maximum dimension of the GridBucket from which the
        # previous values were computed. This is needed to handle a special case related
        # to high-dimensional couplings below.
        max_dim_prev_values = [0]

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

                # If this coarse grid bucket is not involved with the local coarse cell
                # the solution to the local problem will be all zeros. No need to compute
                # anything, just go on.
                if coarse_ind not in gb_set[gb]:
                    continue

                # Loop over the set of pressure values from the previous dimension
                # Find matches between previous cells and current faces, and assign
                # boundary conditions

                # Flag for whether the right hand side has non-zero elements
                trivial_solution = True

                for prev_dim, (g_prev, values) in zip(max_dim_prev_values, prev_values):
                    # Keep track of which cells in g_prev has been used to define bcs in
                    # this gb.
                    # This is a reasonable approach if the cell center in the previous
                    # grid is associated with the point where the boundary
                    found = np.zeros(g_prev.num_cells, dtype=np.bool)

                    # Loop over all grids in gb,
                    for g, d in gb:
                        # This will update the boundary condition values
                        bc_values = d[pp.PARAMETERS][discr.keyword]["bc_values"]

                        cf = cell_face_relations.get((g_prev, g), [None, None])

                        if cf[0] is None:
                            cells_found, faces_found = transfer_bc(
                                g_prev, values, g, bc_values, reg.dim
                            )
                            cell_face_relations[(g_prev, g)] = (
                                cells_found,
                                faces_found,
                            )
                        elif len(cf[0]) == 0:
                            # We have tried this, there is nothing to do
                            cells_found, cells_found = cf
                        else:
                            cells_found, faces_found = transfer_bc(
                                g_prev, values, g, bc_values, reg.dim, cf[0], cf[1]
                            )

                        found[cells_found] = True

                        if len(faces_found) > 0 and prev_dim < gb.dim_max() - 1:
                            # Special case, where a face in the current dimension hit
                            # a cell of a grid bucket two dimensions lower. This will be a
                            # line fracture in a 2d domain intersecting with the cell
                            # center in the macro grid. In this case, the boundary condition
                            # for the new grid must be changed to Dirichlet, and the
                            # right value assigned. The change of type of boundary condition
                            # further requires rediscretization and reassembly of the local
                            # linear system.
                            loc_discr = d[pp.DISCRETIZATION][discr.cell_variable][
                                discr.cell_discr
                            ]
                            breakpoint()
                            # Rediscretize local problem, unless this is a 1d auxiliary line
                            # where a continuity condition has been imposed (this effectively
                            # will have no boundary condition). In the latter case, it would not
                            # hurt to rediscretize, but it is not necessary.
                            if not isinstance(
                                loc_discr,
                                pp.numerics.fv.fv_elliptic.EllipticDiscretizationZeroPermeability,
                            ):
                                loc_discr.discretize(g, d)

                                bc = d[pp.PARAMETERS][discr.keyword]["bc"]
                                bc.is_dir[faces_found] = True
                                bc.is_neu[faces_found] = False
                                # breakpoint()
                                # Reassemble on this gb, and update the assembler map
                                assembler, _ = assembler_map[gb]
                                A_new, _ = assembler.assemble_matrix_rhs(
                                    only_matrix=True
                                )
                                assembler_map[gb] = (assembler, A_new)

                    # Verify that either all values in the previous grid has been used
                    # as boundary conditions, or none have (the latter may happen in
                    # 3d problems).
                    # It is not 100% clear that this assertion is critical, but it seems
                    # likely, so we will need to debug if this is ever broken
                    assert np.all(found) or np.all(np.logical_not(found))

                    if np.any(found):
                        # This gb has had boundary conditions transferred from a
                        # lower-dimensional problem. The solution will not be trivial
                        trivial_solution = False

                for g, d in gb:
                    debug_bc_val_map[g] += d[pp.PARAMETERS][discr.keyword]["bc_values"]

                # Get assembler
                assembler, A = assembler_map[gb]

                if trivial_solution:
                    # The solution is known to be zeros
                    x = np.zeros(assembler.num_dof())
                else:
                    # This will use the updated values for the boundary conditions
                    _, b = assembler.assemble_matrix_rhs(only_rhs=True)
                    # Solve and distribute
                    if gb in ilu_map:
                        x, info = spla.gmres(
                            A=A, b=b, M=ilu_map[gb], restart=500, maxiter=20, tol=1e-8
                        )
                        if info > 0:
                            raise ValueError(
                                "Gmres failed in basis function computation"
                            )
                    else:
                        x = sps.linalg.spsolve(A, b)

                    if not np.all(np.isfinite(x)):
                        breakpoint()

                assembler.distribute_variable(x)

                for g, d in gb:
                    debug_pou_map[g] += d[pp.STATE][discr.cell_variable]

                for e, d in gb.edges():
                    # In some special cases, there are edges with no variables.
                    # Safeguard against this.
                    if "state" in d.keys():
                        debug_pou_map[e] += d[pp.STATE][discr.mortar_variable]

                # Avoid this operation for the highest dimensional gb - that will be
                # reset after we have stored the values (below)
                if gb.dim_max() < local_gb.dim:
                    for g, d in gb:
                        # Reset the boundary conditions in preparation for the next
                        # basis function
                        d[pp.PARAMETERS]["flow"]["bc_values"][:] = 0

                        # Store the pressure values to be used as new boundary
                        # conditions for the problem with one dimension more
                        new_prev_val.append((g, d[pp.STATE][discr.cell_variable]))
                        # Also store the dimension of the previosu value
                        max_dim_prev_values.append(gb.dim_max())

            # We are done with all buckets of this dimension. Redefine the current
            # values to previous values, and move on to the next set of buckets.
            prev_values += new_prev_val

        # Done with all calculations for this basis function. Store it.
        basis_functions[coarse_ind] = x

        coarse_gb[coarse_ind] = gb

        coarse_assembler[coarse_ind] = assembler
        coarse_bc_values[coarse_ind] = {}
        for g, d in gb:
            coarse_bc_values[coarse_ind][g] = d[pp.PARAMETERS]["flow"][
                "bc_values"
            ].copy()
            # Now that we have saved the boundary values, we reset them to avoid
            # disturbing the computation for other basis functions.
            d[pp.PARAMETERS]["flow"]["bc_values"][:] = 0

        # Move on to the next basis function

    # All done
    # Check if the basis functions form a partition of unity, but only for internal
    # faces, or for purely Neumann boundaries.
    check_basis = True
    for bi in reg.macro_boundary_faces():
        if macro_data["bc"].is_dir[bi]:
            check_basis = False

    if False:
        for g, d in gb:
            bc_values = debug_bc_val_map[g]
            bc = d["parameters"]["flow"]["bc"]
            d["parameters"]["flow"]["bc_values"] = debug_bc_val_map[g]
            discr_loc = d["discretization"]["pressure"]["pressure_discr"]
            mat, rhs = discr_loc.assemble_matrix_rhs(g, d)
            x = sps.linalg.spsolve(mat, rhs)
            print(x)

    # NOTE: This makes the tacit assumption that the ordering of the grids is the same
    # in all assemblers. This is probably true, but it should be the first item to
    # check if we get an error message here
    if check_basis:
        basis_sum = np.sum(np.array([b for b in basis_functions.values()]), axis=0)
        for g, _ in assembler.gb:
            dof = assembler._dof_manager.dof_ind(g, discr.cell_variable)
            assert np.allclose(basis_sum[dof], 1, atol=1e-2)

        # Check that the mortar fluxes sum to zero for local problems.
        for e, _ in assembler.gb.edges():
            dof = assembler._dof_manager.dof_ind(e, discr.mortar_variable)
            assert np.allclose(basis_sum[dof], 0, atol=1e-4)

    return (
        basis_functions,
        coarse_assembler,
        coarse_bc_values,
        assembler_map,
        ilu_map,
        cell_face_relations,
    )


def discretize_pressure_trace_macro_bound(
    macro_g, local_gb, discr, cc_assembler, basis_functions
):
    row, col, val = [], [], []

    gb = local_gb.gb
    # For now only consider the highest dimensional grids. We could also include cells
    # on fractures touching the micro boundary, but that would require a more ellaborate
    # area weighting (including the aperture of the micro fracture)
    micro_g = gb.grids_of_dimension(gb.dim_max())[0]

    if hasattr(micro_g, "macro_face_ind") and micro_g.macro_face_ind.size > 0:

        d = gb.node_props(micro_g)

        for coarse_cell, assembler in cc_assembler.items():
            basis = basis_functions[coarse_cell]
            assembler.distribute_variable(basis)

            micro_matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][discr.keyword]

            # Map from micro cells to the boundary conditions
            micro_bound_pressure_map = micro_matrix_dictionary[
                discr.bound_pressure_cell_matrix_key
            ]

            # the cell variable carries the micro basis functions, both if this is a
            # cell or a boundary face discretization
            p = d[pp.STATE][discr.cell_variable]
            micro_boundary_pressure = micro_bound_pressure_map * p

            row += list(micro_g.macro_face_ind)
            col += micro_g.macro_face_ind.size * [coarse_cell]
            # Use an area-weighted pressure reconstruction
            scaled_vals = (
                micro_boundary_pressure[micro_g.face_on_macro_bound]
                * micro_g.face_areas[micro_g.face_on_macro_bound]
                / macro_g.face_areas[micro_g.macro_face_ind]
            )
            val += list(scaled_vals)

    return col, row, val


def compute_transmissibilies(
    reg,
    local_gb,
    basis_functions,
    cc_assembler,
    cc_bc_values,
    coarse_grid,
    discr,
    macro_data,
    cell_face_relations,
    sanity_check=False,
):
    pts, cells, cell_info, phys_names = local_gb.gmsh_data

    # Data structures for storing cell and face index for the local problems, together
    # with the corresponding transmissibilities.
    coarse_cell_ind, coarse_face_ind, trm = [], [], []

    # The macro transmissibilities can be recovered from the fluxes over those micro
    # faces that coincide with a macro face.
    # To identify these micro faces, we recover the micro surface grids. These were
    # present as internal constraints in the local grid bucket. Moreover, to discretize
    # boundary conditions on the macro boundary, we also recover grids on the micro
    # domain boundary; we will dump all grids not on the macro domain boundary shortly
    if coarse_grid.dim == 2:
        # Create all 2d grids that correspond to an auxiliary surface
        constraint_surfaces, _ = msh_2_grid.create_1d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            line_tag=PhysicalNames.AUXILIARY_LINE.value,
        )
        micro_domain_boundary, _ = msh_2_grid.create_1d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            line_tag=PhysicalNames.DOMAIN_BOUNDARY_LINE.value,
        )
        if reg.is_tip:
            # At tips of macro fractures, we should also process transmissibilities
            # for the macro fracture (which was temporarily degraded to a micro fracture
            # in the meshing, see interaction_region).
            # First recover all fracture grids - this will have some overhead, but it
            # should not be too bad.
            fracture_domain_boundary, _ = msh_2_grid.create_1d_grids(
                pts,
                cells,
                phys_names,
                cell_info,
                line_tag=PhysicalNames.FRACTURE.value,
            )
            # We know by construction in interaction region that the macro fracture
            # was put first in the list of micro fractures. Pick out all grid with
            # that frac_num - that should take care of split fractures as well.
            macro_fracture_boundary = [
                g for g in fracture_domain_boundary if g.frac_num == 0
            ]

    else:
        # EK: 3d domains have not been tested.
        # Create all 2d grids that correspond to a domain boundary
        constraint_surfaces = msh_2_grid.create_2d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            is_embedded=True,
            surface_tag=PhysicalNames.AUXILIARY_PLANE.value,
        )
        micro_domain_boundary = msh_2_grid.create_2d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            is_embedded=True,
            surface_tag=PhysicalNames.DOMAIN_BOUNDARY_SURFACE.value,
        )
        if reg.is_tip:
            fracture_domain_boundary = msh_2_grid.create_2d_grids(
                pts,
                cells,
                phys_names,
                cell_info,
                is_embedded=True,
                surface_tag=PhysicalNames.FRACTURE.value,
            )
            macro_fracture_boundary = [
                g
                for g in fracture_domain_boundary
                if g.frac_num < reg.num_macro_frac_faces
            ]

    # In the main loop over micro surface grids below, we need access to a limited set
    # of information. This includes the coarse face index, which must be obtained in
    # different ways for constraint and boundary grids. Use a dedicated structure for
    # storage of the necessary information.
    Surface = namedtuple(
        "Surface", ["coarse_face_index", "cell_centers", "nodes", "dim"]
    )
    surfaces = []

    # Loop over constraint surfaces, create a Surface for each one
    for gi, gs in enumerate(constraint_surfaces):
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

        surfaces.append(Surface(cfi, cc, nc, gs.dim))

    # Loop over micro domain boundary grids, create a Surface
    for gi, gs in enumerate(micro_domain_boundary):
        # Only consider surfaces on the boundary of the macro domain
        # The index of the domain boundary is mapped to the numbering of IAreg surfaces
        # before checking if it is on the macro boundary
        gi_in_reg = reg.domain_edges_2_reg_surface[gi]

        if not reg.surface_is_boundary[gi_in_reg]:
            continue
        gs.compute_geometry()
        cc = gs.cell_centers
        nc = gs.nodes

        if reg.name == "mpfa":
            ind_face = reg.surface_node_type[gi_in_reg].index("face")
            cfi = reg.surfaces[gi_in_reg, ind_face]
        else:
            cfi = reg.reg_ind

        surfaces.append(Surface(cfi, cc, nc, gs.dim))

    if reg.is_tip:
        # Also add surfaces for the fracture boundary.
        for gs in macro_fracture_boundary:
            gs.compute_geometry()
            for fi in np.where(reg.surface_is_macro_fracture)[0]:
                cfi = reg.surfaces[fi][reg.surface_node_type[fi].index("face")]
                surfaces.append(Surface(cfi, gs.cell_centers, gs.nodes, gs.dim))

    # The definition of mortar fluxes (positive from primary to secondary neighbor)
    # may not correspond to the sign conventions for the local grids, thus a
    # correction will be needed. This is costly to compute several times (once for
    # each surface), so precompute the map.
    sign_map: Dict[pp.Grid, np.ndarray] = {}
    for cci in basis_functions:
        gb = cc_assembler[cci].gb
        for loc_g, _ in gb:
            if np.any(loc_g.tags["fracture_faces"]):
                # to point from the higher to the lower dimensional problem
                _, indices = np.unique(loc_g.cell_faces.indices, return_index=True)
                sign_map[loc_g] = loc_g.cell_faces.data[indices]

    # Loop over all created surface grids,
    for gs in surfaces:

        # Cell and node coordinates.
        cc = gs.cell_centers
        nc = gs.nodes
        cfi = gs.coarse_face_index

        macro_sgn = macro_div[:, cfi].sum()

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

            # Set back the boundary conditions used in the computation
            for g, d in gb:
                d[pp.PARAMETERS]["flow"]["bc_values"] = cc_bc_values[cci][g]

            # Reconstruct fluxes in the grid bucket
            pp.fvutils.compute_darcy_flux(
                gb, p_name=discr.cell_variable, lam_name=discr.mortar_variable
            )
            # Loop over all grids in the grid_bucket.
            for loc_g, d in gb:
                # Flux field for this problem of this grid
                grid_flux = d[pp.PARAMETERS][discr.keyword]["darcy_flux"]

                # Flux field for this problem due to the mortar variables
                edge_flux = np.zeros(grid_flux.size)
                if np.any(loc_g.tags["fracture_faces"]):
                    for e, d_e in gb.edges_of_node(loc_g):
                        mg = d_e["mortar_grid"]
                        # Consider only the higher dimensional case
                        if mg.dim == loc_g.dim:
                            continue

                        # Get hold of projection operator
                        if e[0].dim == e[1].dim:
                            # For grids of equal dimension, care is needed to get the
                            # right projection matrix.
                            if e[0] == loc_g:
                                proj = mg.mortar_to_secondary_int()
                            elif e[1] == loc_g:
                                proj = mg.mortar_to_primary_int()
                            else:
                                raise ValueError(
                                    "cannot match mortar projection with number of faces"
                                )
                        else:
                            # The primary grid is of higher dimension
                            proj = mg.mortar_to_primary_int()

                        edge_flux += sign_map[loc_g] * (
                            proj * d_e[pp.STATE]["mortar_flux"]
                        )

                # Construct the full flux
                full_flux = grid_flux + edge_flux

                # Identify micro faces that form part of the macro fracture.
                if coarse_grid.tags["fracture_faces"][cfi]:
                    # If this is a macro fracture, we already have the mapping between
                    # micro and macro faces.
                    # Note that using the alternative matching (the below 'else') will
                    # fail at macro fracture tips, where micro faces at both sides of the
                    # macro face will give a hit.
                    if not hasattr(loc_g, "face_on_macro_bound"):
                        # If no faces on the macro boundary, we can continue.
                        continue

                    hit = loc_g.macro_face_ind == cfi
                    micro_faces: np.ndarray = loc_g.face_on_macro_bound[hit]

                    fracture_face = True
                    micro_div = pp.fvutils.scalar_divergence(loc_g)
                else:
                    fracture_face = False
                    if loc_g.dim == gs.dim + 1:
                        # find faces in the matrix grid on the surface
                        grid_map = _match_points_on_surface(
                            cc, loc_g.face_centers, coarse_grid.dim, gs.dim, nc
                        )
                    elif loc_g.dim == gs.dim:
                        # find fractures that intersect with the surface
                        # If we get an error message from this call, about more than one
                        # hit in the second argument (p), chances are that a fracture is
                        # intersecting at the auxiliary surface

                        grid_map = _match_points_on_surface(
                            nc, loc_g.face_centers, coarse_grid.dim, gs.dim, cc
                        )
                    else:
                        grid_map = []

                    micro_faces = np.array([i[1] for i in grid_map])

                # If we found any matches, loop over all micro faces, sum the fluxes,
                # possibly with an adjustment of the flux direction.
                surface_flux = []
                for fi in micro_faces:
                    loc_flux = full_flux[fi]

                    # If the micro and macro normal vectors point in different
                    # directions, we should switch the flux
                    # As with the identification of correct micro faces, fracture faces
                    # need special treatment.
                    if fracture_face:
                        sgn = micro_div[:, fi].sum()
                    else:
                        fine_normal = loc_g.face_normals[:, fi]
                        sgn = np.sign(fine_normal.dot(coarse_normal))
                    surface_flux.append(loc_flux * sgn)

                # Store the macro cell and face index, together with the
                # transmissibility
                coarse_cell_ind.append(cci)
                coarse_face_ind.append(cfi)
                trm.append(np.asarray(surface_flux).sum())

    check_trm = sanity_check
    for bi in reg.macro_boundary_faces():
        # The macro transmissibilities should sum to zero to preserve no flow for constant
        # constant pressure.  Check if the basis functions form a partition of unity,
        # but only for internal faces, or for purely Neumann boundaries.
        if macro_data["bc"].is_dir[bi]:
            check_trm = False

    #    check_trm = False
    # NOTE: This makes the tacit assumption that the ordering of the grids is the same
    # in all assemblers. This is probably true, but it should be the first item to
    # check if we get an error message here
    # If the region only has macro boundary faces (think corners of the macro grid)
    # no transmissibilies have been computed, there is no need to run the check.
    if check_trm and len(coarse_face_ind) > 0:
        trm_sum = np.bincount(coarse_face_ind, weights=trm)
        trm_scale = np.amax(np.bincount(coarse_face_ind, weights=0.5 * np.abs(trm)))
        trm_scale = trm_scale if trm_scale else 1
        hit = np.abs(trm_sum) > 1e-6  # tpfa in 3d gave some problems, this fixed them
        assert np.allclose(trm_sum[hit] / trm_scale, 0, atol=1e-4)

    return coarse_cell_ind, coarse_face_ind, trm


def discretize_boundary_conditions(
    reg,
    local_gb,
    discr,
    macro_data,
    coarse_g,
    assembler_map,
    ilu_map,
    cell_face_relations,
):
    """
    Discretization of boundary conditions, consistent with the construction of basis
    functions for internal cells.

    The implementation is in part very similar to that for the basis function
    computation, and should be refactored at some point.

    Args:
        reg (TYPE): DESCRIPTION.
        local_gb (TYPE): DESCRIPTION.
        discr (TYPE): DESCRIPTION.
        macro_data (TYPE): DESCRIPTION.
        coarse_g (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    # Boundary conditions are discretized by a set of local problems, initialized by
    # unit values at the relevant boundary (similar to the cell center initiation for
    # basis function computation). We will therefore need an assembler for each local
    # grid bucket, and initialize the
    # Find the index of region boundary surfaces that are also on a macro boundary
    reg_bound_surface: np.ndarray = np.where(reg.surface_is_boundary)[0]
    if reg_bound_surface.size == 0:
        # Nothing to do here
        return ([], [], []), ([], [], [])

    # IMPLEMENTATION NOTE: We can reuse the discretization from the basis function
    # computation, thereby saving quite some time.

    # Get a list of the local grid bucktes. This will be first 1d, then 2d (if dim == 3)
    # and then the real local gb
    bucket_list = local_gb.bucket_list()

    # The macro face may be split into several region faces (mpfa in 3d).
    # Get hold of these, and get a mapping from the region faces to the macro faces
    if reg.name == "mpfa":

        all_faces = []
        # By construction of the boundary of the mpfa region, the surface will have two
        # (2d) or three (3d) nodes, with one being defined by a macro face center.
        # Use this macro face index to identify region surfaces that are part of the
        # same macro surface.
        for si in reg_bound_surface:
            face_of_si = reg.surfaces[si][reg.surface_node_type[si].index("face")]
            all_faces.append(face_of_si)

        macro_bound_faces, reg_to_macro_bound_ind = np.unique(
            all_faces, return_inverse=True
        )
    else:
        # Tpfa; the region index gives the macro face
        macro_bound_faces = np.array([reg.reg_ind])
        # The mapping between region and macro surfaces is simple
        reg_to_macro_bound_ind = np.array([0])

    # The first item gives the macro boundary face.
    # Second gives the region faces. Third gives coordinates where region edges end on
    # this surface.
    surface_edge_pairs: List[Tuple[int, np.ndarray, List[np.ndarray]]] = []

    # Loop over all macro boundary faces, find its associated region surfaces, and
    # region edge coordinates on the face.
    # TODO: The edge coordinate is used to identify micro points on the surface. For
    # 3d grids, we will likely need more coordinate information; it will be necessary
    # to identify all points on the line between the edge and the face center (this
    # line will be the boundary of a 2d surface, on which we need to set boundary
    # conditions).
    for ind, bfi in enumerate(macro_bound_faces):
        # Region surfaces on this macro surface. Essentially invert
        # reg_to_macro_bound_ind, this could probably have been done in a simpler way.
        loc_surfaces = reg_bound_surface[np.where(reg_to_macro_bound_ind == ind)[0]]

        loc_edges = []

        # Loop over all edges, find those that end on this macro face.
        # The check is different for tpfa and mpfa type regions.
        for edge, edge_node in zip(reg.edges, reg.edge_node_type):
            if reg.name == "mpfa":
                if edge[edge_node.index("face")] == bfi:
                    loc_edges.append(
                        coarse_g.face_centers[:, edge[edge_node.index("face")]]
                    )
            else:  # tpfa
                if len(edge_node) == 2:
                    loc_edges.append(coarse_g.nodes[:, edge[edge_node.index("node")]])

        surface_edge_pairs.append((bfi, loc_surfaces, loc_edges))

    # Boundary condition object for the macro problem. We will use this to set the
    # right boundary condition also for the micro problem.
    macro_bc = macro_data["bc"]

    # Get the positive direction of the macro faces. This will be needed to compare the
    # signs of the macro and micro faces.
    _, macro_fi, _ = sps.find(pp.fvutils.scalar_divergence(coarse_g))

    boundary_basis_functions = {}
    boundary_assemblers = {}
    boundary_bc_values = {}

    # Loop over all macro faces, provide discretization of boundary condition
    for macro_face, surf, edge in surface_edge_pairs:

        # Number of nodes of the macro fracture - needed to get the right scaling of
        # Neumann boundary conditions.
        num_nodes_of_macro_face = coarse_g.face_nodes[:, macro_face].data.size

        # For Neumann faces, the flux through the macro face must be distributed over the
        # micro faces. If no microscale fractures touch the macro face, the micro face areas
        # (of what will then be only the highest-dimensional micro grid) should sum to the
        # macro face area. However, if micro fractures are present, these must be included,
        # and the macro flux distributed over what will be a larger area. Sum the micro
        # face areas (accounting for specific volumes ['aperture'] of lower-dimensional grids
        # for use in the distribution below.
        # NOTE: The mismatch between macro and summed micro area is a consequence of the
        # lower-dimensional representation of the grids.
        # NOTE: Use separate macro areas for each gb (essentially each dimension in the
        # cascade of local problems), to spread a unit flux over the right area.
        macro_area: Dict[pp.GridBucket, float] = {}

        if macro_bc.is_neu[macro_face]:
            for gb_set in bucket_list:
                for gb in gb_set:
                    macro_area[gb] = 0
                    for g, d in gb:
                        if hasattr(g, "macro_face_ind"):
                            # Find those micro faces that form this macro (boundary) face
                            hit = g.macro_face_ind == macro_face
                            if not np.any(hit):
                                continue

                            # Get indices of micro faces on macro boundary
                            micro_bound_face = g.face_on_macro_bound[hit]
                            data = d[pp.PARAMETERS][discr.keyword]
                            # Specific volume should be given as a number per object.
                            # Extension to one value per face (or cell) is simple, but
                            # we do not bother with that.
                            specific_volume: Union[float, int] = data.get(
                                "specific_volume", 1
                            )

                            macro_area[gb] += (
                                g.face_areas[micro_bound_face].sum() * specific_volume
                            )

        # Data structure to store the value for the grid bucket set of a lower dimension
        prev_values = []

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

                # Flag for whether the right hand side has non-zero elements
                trivial_solution = True

                for g_prev, values in prev_values:
                    # Keep track of which cells in g_prev has been used to define bcs in
                    # this gb
                    found = np.zeros(g_prev.num_cells, dtype=np.bool)

                    # Loop over all grids in gb,
                    for g, d in gb:
                        # This will update the boundary condition values
                        bc_values = d[pp.PARAMETERS][discr.keyword]["bc_values"]
                        cf = cell_face_relations.get((g_prev, g), [None, None])

                        if cf[0] is None:
                            cells_found, faces_found = transfer_bc(
                                g_prev, values, g, bc_values, reg.dim
                            )
                            cell_face_relations[(g_prev, g)] = (
                                cells_found,
                                faces_found,
                            )
                        elif len(cf[0]) == 0:
                            # We have tried this, there is nothing to do
                            cells_found, cells_found = cf
                        else:
                            cells_found, faces_found = transfer_bc(
                                g_prev, values, g, bc_values, reg.dim, cf[0], cf[1]
                            )

                    # Verify that either all values in the previous grid has been used
                    # as boundary conditions, or none have (the latter may happen in
                    # 3d problems).
                    # It is not 100% clear that this assertion is critical, but it seems
                    # likely, so we will need to debug if this is ever broken
                    assert np.all(found) or np.all(np.logical_not(found))
                    if np.any(found):
                        trivial_solution = False

                for g, d in gb:
                    if hasattr(g, "macro_face_ind"):
                        trivial_solution = False
                        data = d[pp.PARAMETERS][discr.keyword]
                        bc_values = data["bc_values"]

                        # Find those micro faces that form this macro (boundary) face
                        hit = g.macro_face_ind == macro_face
                        if not np.any(hit):
                            continue

                        # Get indices of micro faces on macro boundary
                        micro_bound_face = g.face_on_macro_bound[hit]

                        if macro_bc.is_dir[macro_face]:
                            # For Dirichlet conditions, simply set a unit pressure
                            bc_values[micro_bound_face] = 1
                        else:
                            # Distribute the Neumann flux between micro faces according
                            # to their areas.
                            # No sign corrections here: Neumann conditions for the (micro) fv
                            # discretizations are treated as positive for flux out of the domain,
                            # independent of the direction of the normal vector. Thus the macro
                            # boundary condition are computed with the same convention (which also
                            # is the sign convention for mortar fluxes).
                            # NOTE: The boundary transmissibility at the Neumann face still
                            # must be adjusted to account for possible sign changes due to the
                            # divergence of that face. This is taken care of in fv_dfm.py.

                            # The face areas are scaled with the specific volume of the grid,
                            # as was done when computing the macro area above.
                            face_areas = g.face_areas[micro_bound_face] * data.get(
                                "specific_volume", 1
                            )
                            # NOTE: The scaling with specific volume is only used to
                            # distribute the boundary condition among the micro faces of
                            # various dimensions. The fluxes retain the PorePy convention of
                            # being volume fluxes (no implicit aperture scaling etc), thus
                            # there is no need for similar compensation in other places.
                            bc_values[micro_bound_face] = (
                                1
                                * face_areas
                                / (macro_area[gb] * num_nodes_of_macro_face)
                            )

                # Get assembler
                assembler, A = assembler_map[gb]
                if trivial_solution:
                    # The solution is known to be zeros
                    x = np.zeros(assembler.num_dof())
                else:
                    # This will use the updated values for the boundary conditions
                    _, b = assembler.assemble_matrix_rhs(only_rhs=True)
                    # Solve and distribute
                    if gb in ilu_map:
                        x, _ = spla.gmres(
                            A=A, b=b, M=ilu_map[gb], restart=500, maxiter=20
                        )
                    else:
                        x = sps.linalg.spsolve(A, b)

                assembler.distribute_variable(x)

                # Postpone this operation for the highest dimensional gb until the
                # boundary values have been stored for use in transmissibility
                if gb.dim_max() < local_gb.dim:
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

        # We have come to the end of the discretization for this boundary face.
        # Store basis functions, assembler and boundary condition.
        boundary_basis_functions[macro_face] = x

        boundary_assemblers[macro_face] = assembler
        boundary_bc_values[macro_face] = {}
        for g, d in gb:
            bc_values = d[pp.PARAMETERS]["flow"]["bc_values"]
            # Store the boundayr conditions
            boundary_bc_values[macro_face][g] = bc_values.copy()
            # Now that the values have been stored, the boundary conditions are deleted
            # to prepare for the next boundary face.
            # We could also have put this as an else after the above assginment of
            # new_prev_val (it would have been equivalent, but the present placement
            # feels more logical).
            bc_values[:] = 0

    # Use the basis functions to compute transmissibilities for the boundary
    # discretizaiton
    col_ind, row_ind, trm = compute_transmissibilies(
        reg,
        local_gb,
        boundary_basis_functions,
        boundary_assemblers,
        boundary_bc_values,
        coarse_g,
        discr,
        macro_data,
        cell_face_relations,
        # The transmissibilities need to sum to zero for boundary discretizaitons, so
        # we skip the sanity check
        sanity_check=False,
    )

    trace_discr = discretize_pressure_trace_macro_bound(
        coarse_g, local_gb, discr, boundary_assemblers, boundary_basis_functions
    )

    return (col_ind, row_ind, trm), trace_discr
