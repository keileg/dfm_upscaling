#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:19:40 2020

@author: eke001
"""

import numpy as np
import porepy as pp
import meshio
import networkx as nx
from typing import Dict, List, Tuple

from porepy.utils.setmembership import unique_columns_tol
from porepy.fracs import msh_2_grid
from porepy.fracs.gmsh_interface import Tags, PhysicalNames
from porepy.fracs import simplex

from dfm_upscaling import interaction_region as ia_reg


class LocalGridBucketSet:
    def __init__(self, dim, reg):

        self.dim = dim
        self.reg = reg
        self.tol = 1e-6

    def bucket_list(self) -> List[Dict[pp.GridBucket, List[int]]]:
        """Get all grid buckets associated with (the boundary of) an interaction region.

        The data is first sorted as a list, with first item being all 0d GridBuckets,
        next item all 1d etc. Each list item is a dictionary with the GridBuckets as keys,
        and the macro cell indices to which they are connected as values.

        """
        if self.dim == 2:
            return [self.line_gb, {self.gb: self.reg.macro_cell_inds()}]
        else:
            return [
                self.line_gb,
                self.surface_gb,
                {self.gb: self.reg.macro_cell_inds()},
            ]

    def construct_local_buckets(self, data=None):
        if data is None:
            data = {}
        if self.dim == 2:
            self._construct_buckets_2d(data)
        elif self.dim == 3:
            self._construct_buckets_3d(data)

        self._tag_faces_macro_boundary(self.gb)

    def _construct_buckets_2d(self, data):

        mesh_args = data.get("mesh_args", None)

        gb, network = self.reg.mesh(mesh_args)
        self.network = network
        self.gb = gb

        for g, _ in gb:
            if g.dim < self.dim:
                g.from_fracture = True

        # We need to define point tags, which are assumed to exist by
        # self._recover_line_gb()
        edges = network._decomposition["edges"]

        # Each point should be classified as either boundary, fracture or fracture and
        # boundary, according to which edges share the point

        # Initialize by a neutral tag
        point_tags = Tags.NEUTRAL.value * np.ones(
            network._decomposition["points"].shape[1], dtype=np.int
        )

        # Find the points of boundary and fracture edges
        boundary_points = np.union1d(
            edges[0, edges[2] == Tags.DOMAIN_BOUNDARY_LINE.value],
            edges[1, edges[2] == Tags.DOMAIN_BOUNDARY_LINE.value],
        )
        fracture_points = np.union1d(
            edges[0, edges[2] == Tags.FRACTURE.value],
            edges[1, edges[2] == Tags.FRACTURE.value],
        )

        # Split into fracture, boundary or both
        fracture_boundary_points = np.intersect1d(boundary_points, fracture_points)
        only_fracture_points = np.setdiff1d(fracture_points, boundary_points)
        only_boundary_points = np.setdiff1d(boundary_points, fracture_points)

        # Tag accordingly
        point_tags[fracture_boundary_points] = Tags.FRACTURE_BOUNDARY_LINE.value
        point_tags[only_fracture_points] = Tags.FRACTURE.value
        point_tags[only_boundary_points] = Tags.DOMAIN_BOUNDARY_POINT.value

        # Store information
        network._decomposition["point_tags"] = point_tags

        # for 2d problems, the physical (gmsh) tags can also be used to identify
        # individual interaction regions (this follows form how the gmsh .geo file is
        # set up).
        network._decomposition["edges"] = network._decomposition["edges"][
            [0, 1, 2, 3, 3]
        ]

        # Read mesh data and store it
        self.gmsh_data = simplex._read_gmsh_file(self.reg.file_name + ".msh")
        self._recover_line_gb(network)

    def _construct_buckets_3d(self, data):
        mesh_args = data.get("mesh_args", None)

        gb, network = self.reg.mesh(mesh_args)

        for g, _ in gb:
            if g.dim < self.dim:
                g.from_fracture = True

        self.gb = gb
        self.network = network
        network._decomposition = network.decomposition
        decomp = network._decomposition

        edges = decomp["edges"]
        edge_tags = decomp["edge_tags"]

        def edge_indices(subset, edges):
            """Helper function to find a subset of edges in the full edge set

            Parameters:
                subset (np.array, 2 x n): Edges, defined by their point indices
                edges (np.array, 2 x n): Full set of edges.

            Raises:
                ValueError if a column in subset cannot be found in edge

            Returns:
                ind np.array, size subset.shape[1]: Column indices, so that
                    subset[:, i] == edges[:, ind[i]], possibly with the rows switched.

            """
            ns = subset.shape[1]

            ind = []

            for ei in range(ns):
                upper = np.intersect1d(
                    np.where(subset[0, ei] == edges[0])[0],
                    np.where(subset[1, ei] == edges[1])[0],
                )
                lower = np.intersect1d(
                    np.where(subset[1, ei] == edges[0])[0],
                    np.where(subset[0, ei] == edges[1])[0],
                )

                hits = np.hstack((upper, lower))
                if hits.size != 1:
                    raise ValueError("Could not match subset edge with a single edge")
                ind.append(hits[0])

            return np.array(ind)

        # To find edges on interaction region edges, we will look for edges on the
        # network boundary that coincide with edges of the interaction region.

        # Data structure for storing start and endpoints of interaction region edges
        ia_edge_start, ia_edge_end = [], []

        # Loop over edgen in region, store coordinates
        for edge, node_type in zip(self.reg.edges, self.reg.edge_node_type):
            coords = self.reg.coords(edge, node_type)

            # The first n-1 points are start points, the rest are end points
            for i in range(coords.shape[1] - 1):
                ia_edge_start.append(coords[:, i])
                ia_edge_end.append(coords[:, i + 1])

        def ia_edge_for_segment(e):
            """
            For an edge, find the interaction region edge which e is part of, that is,
            e should coincide with the whole, or part of, the ia edge, and no part of e
            extends outside the region.

            Args:
                e (np.array, size 2): Index of start and endpoint of the segment.

            Returns:
                int: Index an interaction region edge which the edge e coincides with.
                    The index runs over the start and endpoints, given in ia_edge_start
                    and _end. If no match is found, -1 is returned.

            """
            # Loop over all edges in the interaction region, look for a match
            for ei, (start, end) in enumerate(zip(ia_edge_start, ia_edge_end)):
                # Vectors form the start of the ia edge to the points of this edge
                v0 = decomp["points"][:, e[0]] - start
                v1 = decomp["points"][:, e[1]] - start

                v_edge = end - start

                # If any of the points are not on the ia edge, it is not a match
                # TODO: Sensitivity to the tolerance here is unknown.
                if (
                    np.linalg.norm(np.cross(v0, v_edge)) > self.tol
                    or np.linalg.norm(np.cross(v1, v_edge)) > self.tol
                ):
                    continue

                # Both points are on the line. Now check if it is on the line segment
                dot_0 = v_edge.dot(v0)
                dot_1 = v_edge.dot(v1)
                v_edge_nrm = np.linalg.norm(end - start)

                if dot_0 < 0 or dot_1 < 0:
                    # We are on the wrong side of start
                    continue
                elif (
                    dot_0 > v_edge_nrm ** 2 + self.tol
                    or dot_1 > v_edge_nrm ** 2 + self.tol
                ):
                    # We are on the other side of end
                    continue

                # It is a match!
                return ei

            # The segment is on no interaction region edge
            return -1

        # We will need two sets of edge numberings. First the numbering assigned to the
        # lines during export to gmsh, which will be translated into g.frac_num during
        # import of gmsh meshes. This seems to be linear with the ordering of the edges,
        # but we assign it manually, and set -1 to lines not on the boundary, to uncover
        # bugs more easily
        physical_line_counter = -np.ones(edges.shape[1], dtype=np.int)
        # The second numbering assigns a common index to all lines that form an edge
        # in the interaction region, and leaves -1s for all other edges
        ia_reg_edge_numbering = -np.ones(edges.shape[1], dtype=np.int)

        # Loop over polygons in the network
        for pi, poly in enumerate(decomp["polygons"]):
            # Only consider the network boundary
            if not network.tags["boundary"][pi]:
                continue

            # Get the edge indices for this polygon
            poly_edge_ind = edge_indices(poly, edges)

            # Loop over
            for i, ei in enumerate(poly_edge_ind):
                # If we have treated this edge before, we can continue
                if ia_reg_edge_numbering[ei] >= 0:
                    continue
                # Find the index of the ia segment which this ei coincides
                ia_edge_ind = ia_edge_for_segment(poly[:, i])

                if ia_edge_ind < 0:
                    # We found nothing, this edge is not on an interaction region edge
                    continue
                else:
                    # Store the number part of the physical name of this edge
                    physical_line_counter[ei] = ei
                    # Assign the ia edge number to this network edge. This will ensure
                    # that all edges that are on the same ia_edge also have the same
                    # index
                    ia_reg_edge_numbering[ei] = ia_edge_ind

        # Sanity check
        assert np.all(ia_reg_edge_numbering[edge_tags == 1]) >= 0

        network._decomposition["edges"] = np.vstack(
            (edges, edge_tags, physical_line_counter, ia_reg_edge_numbering)
        )

        # Read mesh data and store it
        self.gmsh_data = simplex._read_gmsh_file(self.reg.file_name + ".msh")

        self._recover_line_gb(network)
        self._recover_surface_gb(network)

    def _recover_line_gb(self, network):
        """We will use the following keys / items in network._decomposition:

        points: Coordinates of all point involved in the description of the network.
        edges: Connection between lines.
            First two rows are connections between decomp.points
            Third row is the type of edge this is, referring to GmshConstant tags.
            Fourth row is the number part of the physical name that gmsh has
                assigned to this grid.
            Fifth row gives a line index, so that segments that are (or were prior
                 to splitting) part of the same line have the same index
        domain_boundary_points: Index, referring to decomp., of points that are
            part of the domain boundary definition.
        fracture_boundary_points: Index, referring to decomp., of points that are
            part of a fracture, and on the domain boundary. Will be formed by the
            intersection of a fracture and a line that has tag DOMAIN_BOUNDARY_TAG

        """
        decomp = network._decomposition
        # Recover the full description of the gmsh mesh

        pts, cells, cell_info, phys_names = self.gmsh_data

        # Create all 1d grids that correspond to a domain boundary
        g_1d = msh_2_grid.create_1d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            line_tag=PhysicalNames.DOMAIN_BOUNDARY_LINE.value,
            return_fracture_tips=False,
        )

        # Create 0d grid that corresponds to a domain boundary point.
        # Some of these are in cell centers of the coarse grid, while others are on
        # faces. Only some of the grids are included in the local grid buckets -
        # specifically the cell centers grids may not be needed - but it is easier to
        # create all, and then dump those not needed.
        g_0d_domain_boundary = msh_2_grid.create_0d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            target_tag_stem=PhysicalNames.DOMAIN_BOUNDARY_POINT.value,
        )
        # Create a mapping from the domain boundary points to the 0d grids
        # The keys are the indexes in the decomposition of the network.
        domain_point_2_g = {}
        for g in g_0d_domain_boundary:
            domain_point_2_g[
                #                decomp["domain_boundary_points"][g._physical_name_index]
                g._physical_name_index
            ] = g

        # Similarly, construct 0d grids for the intersection between a fracture and the
        # edge of a domain
        g_0d_frac_bound = msh_2_grid.create_0d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            target_tag_stem=PhysicalNames.FRACTURE_BOUNDARY_POINT.value,
        )
        fracture_boundary_points = np.where(
            decomp["point_tags"] == Tags.FRACTURE_BOUNDARY_LINE.value
        )[0]

        # Assign the 0d grids an attribute g.from_fracture, depending on whether it
        # coincides with a fracture or is an auxiliary point.
        # Also add a tag that identifies the grids as auxiliary on not - this will be
        # used to assign discretizations for the subproblems.
        for g in g_0d_frac_bound:
            g.from_fracture = True
            g.is_auxiliary = False

        for g in g_0d_domain_boundary:
            g.from_fracture = False
            g.is_auxiliary = True

        for g in g_1d:
            g.compute_geometry()
            g.from_fracture = False
            g.is_auxiliary = False

        # A map from fracture points on the domain boundary to the 0d grids.
        # The keys are the indexes in the decomposition of the network.
        frac_bound_point_2_g = {}
        for g in g_0d_frac_bound:
            frac_bound_point_2_g[g._physical_name_index] = g

        # Get the points that form the boundary of the interaction region
        boundary_point_coord, boundary_point_ind = self._network_boundary_points(
            network
        )

        # Find all edges that are marked as a domain_boundary
        bound_edge = decomp["edges"][2] == Tags.DOMAIN_BOUNDARY_LINE.value
        # Find the index of edges that are associated with the domain boundary. Each
        # part of the boundary may consist of a single line, or several edges that are
        # split either by fractures, or by other auxiliary points.
        bound_edge_ind = np.unique(decomp["edges"][-1][bound_edge])

        # Mapping from the frac_num, which **should** (TODO!) be equivalent to the
        # edge numbering in decomp[edges][3], thus bound_edge_ind, to 1d grids on the
        # domain boundary.
        bound_edge_ind_2_g = {g.frac_num: g for g in g_1d}

        # Data structure for storage of 1d grids
        buckets_1d: Dict[pp.GridBucket, List[int]] = {}

        # Loop over the edges in the interaction region
        for ia_edge, node_type in zip(self.reg.edges, self.reg.edge_node_type):

            # Recover coordinates of the edge points
            ia_edge_coord = self.reg.coords(ia_edge, node_type)

            # Match points in the region with points in the network
            # It may be possible to recover this information from the network
            # decomposition, but comparing coordinates should be safe, since we should
            # know the distance between the points (they cannot be arbitrary close,
            # contrary to points defined by gmsh)

            # Find which of the domain boundary points form the interaction region edge
            domain_pt_ia_edge = self._match_points(ia_edge_coord, boundary_point_coord)

            # We know how many vertexes there should be on an ia_edge
            if node_type[-1] == "cell":
                num_ia_edge_vertexes = 3
                is_boundary_edge = False
            else:
                # this is a boundary edge
                num_ia_edge_vertexes = 2
                is_boundary_edge = True

            # All ia_edge vertexes should be found among the boundary points
            if len(domain_pt_ia_edge) != num_ia_edge_vertexes:
                raise ValueError(
                    """Could not match domain boundary points with edges in
                                 interaction region"""
                )

            # Express the ia_reg edge in terms of indices in the decomposition of the
            # network
            ia_edge_by_network_decomposition = np.vstack(
                (
                    boundary_point_ind[domain_pt_ia_edge][:-1],
                    boundary_point_ind[domain_pt_ia_edge][1:],
                )
            )

            # Data structure for the 1d grids that together form an edge
            edge_grids_1d = []

            # The 0d grids that connects the parts of this interaction region edge.
            # These will be located on the middle point of the edge
            # We differ between this and points introduced by the fracture, as the two
            # types of point grids will be assigned different conditions in the flow
            # problem
            if is_boundary_edge:
                edge_grids_0d = []
            else:
                # To get the right grid, first map the midpoint on the ia_edge to the
                # boundary point, and then access the dictionary of all boundary point
                # grids
                pt_ind = boundary_point_ind[domain_pt_ia_edge[1]]
                # The midpoint can be either among the domain or the fracture boundary points
                if pt_ind in domain_point_2_g:
                    edge_grids_0d = [domain_point_2_g[pt_ind]]
                elif pt_ind in frac_bound_point_2_g:
                    edge_grids_0d = [frac_bound_point_2_g[pt_ind]]
                else:
                    raise KeyError("Point on 1d region edge not found among domain or fracture boundary points")

            #                breakpoint()

            # Data structure for storing point grids along the edge that corresponds to
            # fracture grids
            fracture_point_grids_0d = []

            # Loop over the parts of this edge. This will be one straight line that
            # forms a part of the domain boundary
            # There is one ia_edge less than there are ia edge vertexes
            for partial_edge in range(num_ia_edge_vertexes - 1):
                # Find the nodes of this edge, in terms of the network decomposition
                # indices
                loc_reg_edge = ia_edge_by_network_decomposition[:, partial_edge]

                # Keep track of whether we have found the 1d grid. Not doing so will be
                # an error
                found = False

                # Loop over all network edges on the boundary, grouped so that edges
                # that belong to the same ia edge are considered jointly.
                # The network edge should have start and endpoints corresponding to one
                # of the loc_edge_edge, and possible other points in between.
                for bi in bound_edge_ind:

                    if bi < 0:
                        # This is a line that is on the domain boundary, but not on an
                        # edge of the interaction region
                        continue

                    # Network edges that have this edge number
                    loc_edges = decomp["edges"][-1] == bi

                    # edge_components contains the points in the decomposition of the
                    # domain that together form an edge on the domain boundary
                    edge_components = decomp["edges"][:2, loc_edges]
                    # Sort the points, to form a line from start to end.
                    sorted_edge_components, _ = pp.utils.sort_points.sort_point_pairs(
                        edge_components, is_circular=False
                    )

                    # The end points of this boundary edge, in the decomposition, are
                    # sorted_edge[0, 0] and sorted_edge[1, -1]. These should both be
                    # points on the domain boundary
                    assert (
                        sorted_edge_components[0, 0] in decomp["domain_boundary_points"]
                    )
                    if not is_boundary_edge:
                        assert (
                            sorted_edge_components[1, -1]
                            in decomp["domain_boundary_points"]
                        )

                    # Check if the local interaction region edge have indices coinciding
                    # with this edge on the domain boundary
                    if (
                        loc_reg_edge[0] == sorted_edge_components[0, 0]
                        and loc_reg_edge[1] == sorted_edge_components[1, -1]
                    ) or (
                        loc_reg_edge[1] == sorted_edge_components[0, 0]
                        and loc_reg_edge[0] == sorted_edge_components[1, -1]
                    ):
                        # If yes, register this
                        found = True

                        # Add all grids to the set of 1d grids along the region edge
                        for ind in np.where(loc_edges)[0]:
                            g_loc = bound_edge_ind_2_g[decomp["edges"][-2, ind]]
                            # Add the grid if it is not added before - this is necessary
                            # because of minor differences in edge tagging between 2d
                            # and 3d domains
                            if g_loc not in edge_grids_1d:
                                edge_grids_1d.append(g_loc)

                        # Add all 0d grids to the set of 0d grids along the region edge
                        # This will be fracture points, should inherit properties from
                        # the fracture.
                        for pi in sorted_edge_components[0, 1:]:
                            if pi in frac_bound_point_2_g.keys():
                                fracture_point_grids_0d.append(frac_bound_point_2_g[pi])

                        # No need to look for a partner of this ia_edge
                        break

                if not found:
                    raise ValueError(
                        """Could not match an edge in the interaction
                                     region with a 1d boundary grid"""
                    )
                # End of loop over this part of the ia_edge

            # We have now found all parts of the interaction region edge.
            # Create a grid list
            # TODO: We probably need to differ between the two types of 0d grids
            grid_list = [edge_grids_1d, edge_grids_0d + fracture_point_grids_0d]
            #            if self.reg.reg_ind == 2:
            #                breakpoint()
            # Create a grid bucket for this edge
            gb_edge = pp.meshing.grid_list_to_grid_bucket(grid_list)

            # The macro cells of this grid bucket corresponds to the items in
            # ia_edge that are macro cells.
            cell_in_node_type = [
                i for i in range(len(node_type)) if node_type[i] == "cell"
            ]
            cell_ind = ia_edge[cell_in_node_type]

            # Store the bucket - macro cell ind combinations for this ia_edge
            buckets_1d[gb_edge] = cell_ind

        # Store all edge buckets for this region
        for gb in buckets_1d:
            self._tag_faces_macro_boundary(gb)

        self.line_gb = buckets_1d

    def _recover_surface_gb(self, network):

        # Network decomposition
        decomp = network._decomposition

        # Recover the full description of the gmsh mesh
        pts, cells, cell_info, phys_names = self.gmsh_data

        # We need to recover four types of grids:
        #  1) 2d grids on the domain surfaces
        #  2) 1d grids formed by the intersection of domain surfaces and fractures
        #  3) 1d grids formed at the junction between 2d domain surface grids
        #  4) 0d grids formed at the intersection between 1d fracture line grids

        # Create all 2d grids that correspond to a domain boundary
        g_2d_all = msh_2_grid.create_2d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            is_embedded=True,
            surface_tag=PhysicalNames.DOMAIN_BOUNDARY_SURFACE.value,
        )

        # By construction of the meshing in FractureNetwork3d, boundary surfaces are
        # indexed after auxiliary and fracture surfaces. To map from surfaces in the
        # fracture network, to the boundary surfaces as defined by the interaction region,
        # the offset caused by auxiliary and fracture surfaces must be found.
        index_offset = min([g.frac_num for g in g_2d_all])

        # Create maps between surface grids and their representation in the list of
        # region surfaces. This is a bit technical, since in the presence of macro fractures,
        # some of the region surfaces might have been dropeed from the region boundary
        # description before mesh construction (see InteractionRegion._mesh_3d()).
        g_2_surface_ind = {}
        # Also make a map between the index a surface would have when adjusting for
        # region constraints and microscale fractures (but not for macro fracture faces)
        # to the index in the region surface list (accounting for the macro fractures).
        # EK: it may be possible to make do without this construct, but that was beyond me.
        surface_index_adjustment_map = {}

        # Loop over all 2d grids, make a map
        # Adjustment compensates for surfaces being dropped because they were macro fracture faces
        adjustment = 0
        for g in g_2d_all:
            # Index without macro fracture faces
            tmp_index = g.frac_num - index_offset

            # Check if we have passed a new macro fracture face. If so, ramp up the adjustment
            # term
            if tmp_index + adjustment in self.reg.ind_surf_on_macro_frac:
                adjustment += 1
            # Add to the maps
            g_2_surface_ind[g] = tmp_index + adjustment
            surface_index_adjustment_map[tmp_index] = tmp_index + adjustment

        # Gather all surfaces that are on the boundary of the interaction region,
        # but not on the bonudary of the macro domain.
        g_2d = [
            g for g in g_2d_all if not self.reg.surface_is_boundary[g_2_surface_ind[g]]
        ]

        # Map form the frac_num (which by construction in pp.msh_2_grid will correspond
        # to the number part of the gmsh physical name of the surface polygon) to the
        # corresponding physical grid.
        # Note that we need to be careful when accessing this information below, since
        # the frac_num also will count any fracture surface in the network.
        g_2d_map = {g.frac_num: g for g in g_2d}

        # Also make a map between the 2d boundary grids and the macro cells to which
        # the grids belong. We will need this at the very end of this function, when
        # identifying surface GridBuckets with macro cell indices.
        g_2d_2_macro_cell_ind = {}
        for g in g_2d:
            surf = self.reg.surfaces[g_2_surface_ind[g]]
            node_type = self.reg.surface_node_type[g_2_surface_ind[g]]
            macro_cells = surf[
                [i for i in range(len(node_type)) if node_type[i] == "cell"]
            ]
            g_2d_2_macro_cell_ind[g] = list(macro_cells)

        # 1d grids formed on the intersection of fracture surfaces with the domain
        # boundary
        g_1d = msh_2_grid.create_1d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            line_tag=PhysicalNames.FRACTURE_BOUNDARY_LINE.value,
            return_fracture_tips=False,
        )

        # Map from the frac_num, which corresponds to the column index of this line in
        # network.decomposition['edges'], to the corresponding grids.
        g_1d_map = {g.frac_num: g for g in g_1d}

        # Next, get the 1d grids that are on the bounday between two domain surfaces.
        # These are tagged as being on the domain boundary, however, we should only pick
        # up those parts that are not on an edge of an interaction region.

        # Find index of boundray edges that are part of an interaction region edge.
        ia_edge = np.where(decomp["edges"][4] >= 0)[0]

        # Find 1d grids along domain boundaries that are not interaction region edges.
        # The latter is excluded by the constraints keyword.
        g_1d_auxiliary = msh_2_grid.create_1d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            line_tag=PhysicalNames.DOMAIN_BOUNDARY_LINE.value,
            constraints=ia_edge,
            return_fracture_tips=False,
        )

        # On macro boundaries, the local meshing will generate surface grids that are
        # completely on the macro boundary. Remove these.
        for si in np.where(self.reg.surface_is_boundary)[0]:
            # Nodes on the ia surface boundary
            s_pts = self.reg.coords(
                self.reg.surfaces[si], self.reg.surface_node_type[si]
            )
            # Build a list of 1d grids that are not in this surface.
            g_tmp = []
            for g in g_1d_auxiliary:
                g_pts = g.nodes
                # Find points on the (triangular) surface. If at least one point is
                # not on the surface, the grid will be kept.
                assert (
                    s_pts.shape[1] == 3
                )  # If this breaks, use distances.point_polygon
                on_surf = self._points_in_triangle(s_pts, g_pts)
                if on_surf.size < g_pts.shape[1]:
                    g_tmp.append(g)

            # Update list of 1d grids
            g_1d_auxiliary = g_tmp

        # Points that are tagged as both on a fracture and on the domain boundary
        fracture_boundary_points = np.where(
            decomp["point_tags"] == Tags.FRACTURE_BOUNDARY_POINT.value
        )[0]

        # Create grids for physical points on the boundary surfaces. This may be both
        # on domain edges, in the meeting of surface and fracture polygons, and by the
        # meeting of fracture surfaces within a boundary surface.
        g_0d_boundary = msh_2_grid.create_0d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            target_tag_stem=PhysicalNames.FRACTURE_BOUNDARY_POINT.value,
        )

        # 0d grids for points where a fracture is cut by an auxiliary surface.
        g_0d_constraint = msh_2_grid.create_0d_grids(
            pts,
            cells,
            phys_names,
            cell_info,
            target_tag_stem=PhysicalNames.FRACTURE_CONSTRAINT_INTERSECTION_POINT.value,
        )
        g_0d = g_0d_boundary + g_0d_constraint

        # Assign the 1d and 0d grids an attribute g.from_fracture, depending on
        # wether they coincide with a fracture or are auxiliary.
        # Also add a tag that identifies the grids as auxiliary on not - this will be
        # used to assign discretizations for the subproblems.
        for g in g_1d:
            g.from_fracture = True
            g.is_auxiliary = False

        for g in g_1d_auxiliary:
            g.from_fracture = False
            g.is_auxiliary = True

        for g in g_0d:
            g.from_fracture = True
            g.is_auxiliary = False

        # Map from the fracture boundary points, in the network decomposition index, to
        # the corresponding 0d grids
        g_0d_boundary_map, g_0d_constraint_map = {}, {}
        for g in g_0d_boundary:
            ind = np.where(fracture_boundary_points == g._physical_name_index)[0][0]
            g_0d_boundary_map[fracture_boundary_points[ind]] = g

        for g in g_0d_constraint:
            g_0d_constraint_map[g.global_point_ind[0]] = g

        # We now have all the grids needed. The next step is to group them into surfaces
        # that are divided by the interaction region edges. Specifically, 2d surface
        # grids will be joined if they are divided by an auxiliary 1d grid.

        # Data structures for storing pairs of 2d surface and 1d auxiliary grids
        pairs = []
        # bookkeeping
        num_2d_grids = len(g_2d)
        num_1d_grids = len(g_1d_auxiliary)

        # For all surface grids, find all auxiliary grids with which is share
        # nodes (a surface grid face should coincide with a 1d cell).
        # This code is borrowed from pp.meshing.grid_list_to_grid_bucket()
        # It is critical that the operation is carried out before splitting of the
        # nodes, or else the local-to-global node numbering is not applicable.

        # Only look for matches if there are any auxiliary line grids
        if num_1d_grids > 0:
            pairs += self._match_cells_faces(g_2d, g_1d_auxiliary, 0, num_2d_grids)

            # Also
            if len(g_0d_boundary) > 0:
                pairs += self._match_cells_faces(
                    g_1d_auxiliary,
                    g_0d_boundary,
                    num_2d_grids,
                    num_2d_grids + num_1d_grids,
                )

        # To find the isolated components, make the pairs into a graph.
        graph = nx.Graph()
        if len(pairs) > 0:
            for couple in pairs:
                graph.add_edge(couple[0], couple[1])
        else:
            # If no pairs were found, we add the 2d surfaces separately
            for i in range(len(g_2d)):
                graph.add_node(i)

        # Clusters refers to 2d surfaces, together wiht 1d auxiliary nodes that should
        # be solved together. Index up to num_2d_grids points to 2d grids, accessed via
        # g_2d, while higher index points to elements in g_1d_auxiliary
        clusters = []

        # Find the isolated subgraphs. Each of these will correspond to a set of
        # surfaces and auxiliary grids, that should be merged in order to construct the
        # local problems.
        for component in nx.connected_components(graph):
            # Extract subgraph of this cluster
            sg = graph.subgraph(component)
            # Make a list of edges of this subgraph
            clusters.append(list(sg.nodes))

        # Create mappings from the surface grids to their embedded 1d and 0d grids
        g_2d_2_frac_g_map = {}
        g_2d_2_0d_g_map = {}

        # Loop over the surface grids, find their embedded lower-dimensional grids
        for si in np.where(network.tags["boundary"])[0]:
            # Check if the region is on the macro boundary, compensating first for
            # micro fractures and constraints (index_offset), next macro fracture faces.
            if self.reg.surface_is_boundary[
                surface_index_adjustment_map[si - index_offset]
            ]:
                continue

            g_surf = g_2d_map[si]

            # 1d fracture grids are available from the network decomposition
            g_1d_loc = []
            for gi in decomp["line_in_frac"][si]:
                g_1d_loc.append(g_1d_map[gi])
            # Register information
            g_2d_2_frac_g_map[g_surf] = g_1d_loc

            # Sanity check
            g_surf_vertexes = decomp["points"][:, decomp["polygons"][si][0]]
            for g in g_1d_loc:
                dist, _, _ = pp.distances.points_polygon(g.nodes, g_surf_vertexes)
                if dist.max() > 1e-12:
                    raise ValueError("1d grid is not in 2d surface")

            # Find all nodes, in the network decomposition, of the local fracture grids
            loc_network_nodes = decomp["edges"][
                :2, [g.frac_num for g in g_1d_loc]
            ].ravel()

            # Intersection nodes are referred to by the local fracture grids more than
            # once
            intersection_nodes = np.where(np.bincount(loc_network_nodes) > 1)[0]
            # Register information
            loc_0d_grids = [g_0d_boundary_map[i] for i in intersection_nodes]
            for ni in loc_network_nodes:
                if ni in g_0d_constraint_map:
                    loc_0d_grids.append(g_0d_constraint_map[ni])

            g_2d_2_0d_g_map[g_surf] = loc_0d_grids

        # Finally, we can collect the surface grid buckets. There will be one for each
        # cluster, identified above.
        # Data structure
        surface_buckets: Dict[pp.GridBucket, List[int]] = {}
        # Loop over clusters

        for c in clusters:

            g2, g1, g0 = [], [], []

            # Loop over cluster members
            for grid_ind in c:
                # This is either a surface grid, in which case we need to register the
                # grid itself, and its embedded 1d and 0d fracture grids
                if grid_ind < num_2d_grids:
                    g_surf = g_2d[grid_ind]
                    g2 += [g_surf]
                    g1 += list(g_2d_2_frac_g_map[g_surf])
                    g0_this = list(g_2d_2_0d_g_map[g_surf])

                    for g in g0_this:
                        # 0d grids may reside on the boundary between 2d surfaces
                        # (think an auxiliray line between two surfaces, which is intersected
                        # by a fracture line). These point grids should only be added once to
                        # the list of grids
                        if g not in g0:
                            g0 += [g]

                elif grid_ind < num_2d_grids + num_1d_grids:
                    # Here we need to adjust the grid index, to account for the
                    # numbering used in defining the pairs above.
                    # Also check that the 1d grid has not been added before (if it was a
                    # fracture line as well - this is maybe not possible, but better safe than
                    # sorry).
                    target_g = g_1d_auxiliary[grid_ind - num_2d_grids]
                    if not target_g in g1:
                        g1 += [target_g]
                else:
                    # Check that the 0d grid has not been added before. This is a real
                    # possibility, maybe because the tagging of 0d points does not fully
                    # reflect all possibilities for this kind of grids. Sigh.
                    target_g = g_0d_boundary[grid_ind - num_2d_grids - num_1d_grids]
                    if not target_g in g0:
                        g0 += [target_g]

            # Make list, make bucket.
            grid_list = [g2, g1, g0]
            gb_loc = pp.meshing.grid_list_to_grid_bucket(grid_list)

            # The tagging of boundary faces in gb_loc is not to be trusted, since the function
            # which takes care of this (pp.fracs.meshing._tag_faces()) is mainly made for a
            # single grid in the highest dimension. Fixing these issues seems complex (I tried)
            # so instead we reimpose boundary information here.
            # The idea is to find all faces on the boundary of each surface grid, make a
            # unified list of such faces, and count them. Faces that only occur once are on
            # the domain boundary (relative to the surface bucket). Correct the boundary
            # information, and tag those lower-dimensional faces that share a node with the
            # True boundary surfaces as boundary.

            # Data structure for boundary information
            fn_all = np.zeros((2, 0), dtype=int)
            # Number of boundary faces registered for each surface grid.
            num_bnd_face = []

            # Loop over surface grids
            for g in gb_loc.grids_of_dimension(2):
                bnd_face = g.get_all_boundary_faces()
                fn_loc = (g.face_nodes[:, bnd_face].indices).reshape(
                    (2, bnd_face.size), order="F"
                )
                # Append boundary faces, in terms of global node indices
                fn_all = np.hstack((fn_all, g.global_point_ind[fn_loc]))
                # Store number of faces on this boundary
                num_bnd_face.append(bnd_face.size)

            # Index for first element of fn_all for each surface grid
            bnd_face_start = np.hstack((0, np.cumsum(num_bnd_face)))

            # Uniquify the face-nodes. The global boundary is those that only occur
            # once.
            _, _, all_2_unique = pp.utils.setmembership.unique_columns_tol(fn_all)
            bnd = np.where(np.bincount(all_2_unique) == 1)[0]
            # Boolean of boundary faces
            true_bnd = np.zeros(all_2_unique.size, dtype=bool)
            true_bnd[np.in1d(all_2_unique, bnd)] = True

            # Data structue for storing nodes on the boundary
            bnd_nodes = np.array([], dtype=int)

            # Loop again over the surface grids, tag boundary faces, and get global index
            # of nodes on the boundary
            for gi, g in enumerate(gb_loc.grids_of_dimension(2)):
                start = bnd_face_start[gi]
                end = bnd_face_start[gi + 1]

                # Prepare to reset domain boundary information
                bnd_tags = np.zeros(g.num_faces, dtype=bool)
                # Indices of those local boundary faces on the true global boundary
                loc_bnd = g.get_all_boundary_faces()[true_bnd[start:end]]
                bnd_tags[loc_bnd] = True
                # Update tags
                g.tags["domain_boundary_faces"] = bnd_tags

                # also take note of the global indices of boundary nodes
                bnd_nodes = np.hstack(
                    (bnd_nodes, g.global_point_ind[g.face_nodes[:, loc_bnd].indices])
                )

            # Uniquify global boundary nodes.
            unique_bnd_nodes = np.unique(bnd_nodes)
            # Find nodes on 1d grids that are on the global boundary. This ammounts to
            # fracture lines extending all the way to the boundary of the interaction
            # region. Tag the nodes as domain_boundary.
            for g in gb_loc.grids_of_dimension(1):
                # 1d faces have a single node
                fn = g.face_nodes.indices
                domain_boundary = np.in1d(g.global_point_ind[fn], unique_bnd_nodes)
                g.tags["domain_boundary_faces"][domain_boundary] = True

            # Fetch the macro cell ind for all 2d grids of this surface bucket.
            macro_cells = []
            for g in g2:
                macro_cells += g_2d_2_macro_cell_ind[g]

            # Store the bucket and its associated macro cell indices
            # Uniquify the macro cells
            surface_buckets[gb_loc] = list(set(macro_cells))

        # Tag faces that are on the boundary of the macro domain
        for gb in surface_buckets:
            self._tag_faces_macro_boundary(gb)

        self.surface_gb = surface_buckets

    def _match_cells_faces(
        self,
        g_high: List[pp.Grid],
        g_low: List[pp.Grid],
        offset_high: int,
        offset_low: int,
    ) -> List[Tuple[int, int]]:
        # First make a merged cell-node map for all lower-dimensional grids
        cn: List[np.ndarray] = []
        # Number of cells per grid. Will be used to define offsets
        # for cell-node relations for each grid, hence initialize with
        # zero.
        num_cn = [0]
        for lg in g_low:
            # Local cell-node relation
            if lg.dim > 0:
                cn_loc = lg.cell_nodes().indices.reshape(
                    (lg.dim + 1, lg.num_cells), order="F"
                )
                cn.append(np.sort(lg.global_point_ind[cn_loc], axis=0))
            else:
                # for 0d-grids, this is much simpler
                cn.append(lg.global_point_ind)
            #
            num_cn.append(lg.num_cells)

        # Stack all cell-nodes, and define offset array
        # enforce a 2d array, also if the grids are 0d
        cn_all = np.atleast_2d(np.hstack([c for c in cn]))
        cell_node_offsets = np.cumsum(num_cn)

        pairs: List[Tuple[int, int]] = []

        # Loop over surface grids, look for matches between the 2d face-nodes
        # and the 1d cell-nodes of axiliary grids
        for hi, hg in enumerate(g_high):
            # First connect the 2d grid to itself
            pairs.append((hi, hi))

            # Next, connection between hg and lower-dimensional grids.
            # We have to specify the number of nodes per face to generate a
            # matrix of the nodes of each face.
            nodes_per_face = hg.dim
            fn_loc = hg.face_nodes.indices.reshape(
                (nodes_per_face, hg.num_faces), order="F"
            )
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            # Find intersection between cell-node and face-nodes.
            # Node need to sort along 0-axis, we know we've done that above.
            is_mem, cell_2_face = pp.utils.setmembership.ismember_rows(
                cn_all, fn, sort=False
            )
            # Special treatment if not all cells were found: cell_2_face then only
            # contains those cells found; to make them conincide with the indices
            # of is_mem (that is, as the faces are stored in cn_all), we expand the
            # cell_2_face array
            if is_mem.size != cell_2_face.size:
                # If something goes wrong here, we will likely get an index of -1
                # when initializing the sparse matrix below - that should be a
                # clear indicator.
                tmp = -np.ones(is_mem.size, dtype=np.int)
                tmp[is_mem] = cell_2_face
                cell_2_face = tmp

            # Then loop over all low-dim grids, look for matches with this high-dim grid
            for li, lg in enumerate(g_low):
                ind = slice(cell_node_offsets[li], cell_node_offsets[li + 1])
                loc_mem = is_mem[ind]
                # Register the grid pair is there is a match between the grids
                if np.sum(loc_mem) > 0:
                    pairs.append((hi + offset_high, li + offset_low))
        return pairs

    def _match_points(self, p1, p2):
        """Find occurences of coordinates in the second array within the first array.

        Args:
            p1 (np.array): Set of coordinates.
            p2 (np.array): Set of coordinates.

        Raises:
            ValueError: If either p1 or p2 contains duplicate coordinates within an
                accuracy of self.tol.

        Returns:
            indices (list): Index of items in p2 that are also in p1.

        """

        # Check that each of the input points are unique
        for p in (p1, p2):
            p_unique, _, _ = unique_columns_tol(p, self.tol)
            if p_unique.shape[1] != p.shape[1]:
                raise ValueError("Original point sets should be unique")

        # Find a mapping to a common unique set
        _, _, new_2_all = unique_columns_tol(np.hstack((p1, p2)), self.tol)

        # indices in the mapping that occur more than once.
        # To be precise, there should be exactly two occurences (since both p1 and p2
        # are known to be unique).
        doubles = np.where(np.bincount(new_2_all) > 1)[0]

        # Data structure to store the target indices
        indices = []

        for ind in doubles:
            # Find the occurences of this double index
            hit = np.where(new_2_all == ind)[0]
            # The target index is known to be the second one (since we know there is one
            # occurence each in p1 and p2).
            # Adjust for the size of p1.
            indices.append(hit[1] - p1.shape[1])

        return indices

    def _tag_faces_macro_boundary(self, gb: pp.GridBucket) -> None:
        """
        In the main GridBucket, tag faces that coincide with the boundary of the macro
        domain.

        Returns:
            None.

        """
        reg = self.reg

        macro_g: pp.Grid = reg.g
        macro_cf = macro_g.cell_faces.tocsr()

        # Data structures to store faces on macro boundaries in each grid. We keep track
        # of both the faces on the micro boundaries, and the corresponding macro face.
        micro_map: Dict[pp.Grid, np.ndarray] = {}
        macro_map: Dict[pp.Grid, np.ndarray] = {}
        for g, _ in gb:
            micro_map[g] = np.array([], dtype=np.int)
            macro_map[g] = np.array([], dtype=np.int)

        # Loop over all surfaces in the region.
        for surf, node_type, is_bound in zip(
            reg.surfaces, reg.surface_node_type, reg.surface_is_boundary
        ):

            # If this is not a boundary surface, we can continue
            if not is_bound:
                continue

            # Get coordinates of the surface points. There will be reg.dim points
            pts = reg.coords(surf, node_type)

            # Loop over all grids in the main gb, and look for faces on the surface.
            for g, _ in gb:
                # Grids of co-dimension > 1 will not be assigned a bc on the macro
                # boundary.
                if self.reg.dim == 2 and g.dim < reg.dim - 1:
                    continue

                fc = g.face_centers
                cc = g.cell_centers

                # Find face centers on the region surface
                if reg.dim == 2:
                    dist, _ = pp.distances.points_segments(fc, pts[:, 0], pts[:, 1])
                    on_bound = np.where(dist < self.tol)[0]
                else:  # reg.dim == 3
                    assert pts.shape[1] == 3
                    on_bound = self._points_in_triangle(pts, fc)

                if on_bound.size > 0:
                    # On macro domain boundaries, the micro face indices in on_bound are
                    # what we are looking for. However, if the surface corresponds to a
                    # macro fracture (thus there are split macro faces laying on the same
                    # surface), we need to filter away half the micro faces.
                    # Do this by finding cell centers of micro and macro cells, create
                    # vectors from cell to face centers, and pick micro faces with vectors
                    # that point in the same direction as the macro vector.

                    # Create micro vectors
                    micro_cell_ind = g.cell_faces.tocsr()[on_bound].indices
                    micro_vec = fc[:, on_bound] - cc[:, micro_cell_ind]
                    if on_bound.size == 1:
                        micro_vec = micro_vec.reshape((-1, 1))

                    # Create macro vectors
                    macro_face_ind = surf[node_type.index("face")]
                    macro_cell_ind = macro_cf[macro_face_ind].indices[0]
                    macro_vec = (
                        macro_g.face_centers[:, macro_face_ind]
                        - macro_g.cell_centers[:, macro_cell_ind]
                    ).reshape((-1, 1))

                    # Identify micro vectors that point in the same direction as the
                    # macro vector.
                    same_side = np.sum(macro_vec * micro_vec, axis=0) > 0

                    # Append the micro faces to the list of found indices
                    micro_map[g] = np.hstack((micro_map[g], on_bound[same_side]))

                    # The macro face index needs some more work.
                    if reg.name == "tpfa":
                        # For tpfa-style ia regions, we can use the region index
                        macro_ind_expanded = macro_face_ind * np.ones(
                            same_side.sum(), dtype=np.int
                        )
                    else:  # mpfa
                        # If a ValueError is raised here, there is no 'face' in
                        # node_type, and something is really wrong
                        macro_ind_expanded = macro_face_ind * np.ones(
                            same_side.sum(), dtype=np.int
                        )
                    # Store the information
                    macro_map[g] = np.hstack((macro_map[g], macro_ind_expanded))

        # Finally, we have found all micro faces that lie on the boundary of the macro
        # domain. The information can safely be stored (had we done this in the above
        # loop over surfaces, we would have risked overwriting information on different
        # surfaces).
        for g, _ in gb:
            macro_ind = macro_map[g]
            micro_ind = micro_map[g]
            if len(macro_ind) > 0:
                g.macro_face_ind = macro_ind
                g.face_on_macro_bound = micro_ind

    def _network_boundary_points(self, network):
        boundary_ind = network._decomposition["domain_boundary_points"]
        boundary_points = network._decomposition["points"][:, boundary_ind]

        if boundary_points.shape[0] == 2:
            boundary_points = np.vstack((boundary_points, np.zeros(boundary_ind.size)))

        return boundary_points, boundary_ind

    def _points_in_triangle(self, tri_pts: np.ndarray, p: np.ndarray) -> np.ndarray:
        # Check if points (in 3d) are inside a triangle (boundary surface
        # of the interaciton region).
        # The points can fall outside either because they are not in the plane of the
        # triangle, or because they are outside the triangle (but in the plane)

        # Shortcut if no points are passed.
        if p.size == 0:
            return np.array([], dtype=np.int)

        if p.size < 4:
            p = p.reshape((-1, 1))

        # First check if the points are in the plane of the triangle
        # Vectors between points in triangle
        v1 = tri_pts[:, 1] - tri_pts[:, 0]
        v2 = tri_pts[:, 2] - tri_pts[:, 0]
        # Normal vector
        n = np.array(
            [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]
        )
        # Vector from triangle to all candidate points
        vfc = p - tri_pts[:, 0].reshape((-1, 1))
        # Check
        dot = np.abs(n.dot(vfc))
        in_plane = dot < self.tol

        # Next, check if the points are inside the triangle. We do not limit
        # the test to points in the plane, this does not seem worth the effort.
        # Credit: https://blackpawn.com/texts/pointinpoly/
        v1_dot_v1 = v1.dot(v1)
        v1_dot_v2 = v1.dot(v2)
        v2_dot_v2 = v2.dot(v2)
        v1_dot_vfc = v1.dot(vfc)
        v2_dot_vfc = v2.dot(vfc)

        det = v1_dot_v1 * v2_dot_v2 - v1_dot_v2 * v1_dot_v2

        t1 = (v1_dot_vfc * v2_dot_v2 - v2_dot_vfc * v1_dot_v2) / det
        t2 = (v1_dot_v1 * v2_dot_vfc - v1_dot_v2 * v1_dot_vfc) / det

        # Check if points are inside triangle
        # Need to safeguard the comparisons here due to rounding errors
        inside = np.logical_and.reduce(
            (
                t1 >= -self.tol,
                t2 >= -self.tol,
                t1 <= 1 + self.tol,
                t2 <= 1 + self.tol,
                t1 + t2 <= 1 + self.tol,
            )
        )
        # In triangle if both in the plane and inside
        return np.where(np.logical_and(inside, in_plane))[0]

    def __repr__(self) -> str:
        s = (
            f"Set of GridBuckets in {self.dim} dimensions\n"
            f"Main Bucket contains "
            f"{len(self.gb.grids_of_dimension(self.dim-1))} fractures\n"
            f"In lower dimensions:\n"
        )

        if self.dim == 2:
            s += f"In total {len(self.line_gb)} 1d buckets\n"
        else:
            s += (
                f"In total {len(self.surface_gb)} 2d buckets and "
                f"{len(self.line_gb)} 1d buckets\n"
            )

        return s
