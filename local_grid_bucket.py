#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:19:40 2020

@author: eke001
"""

import numpy as np
import porepy as pp
import meshio

from porepy.utils.setmembership import unique_columns_tol
from porepy.grids.gmsh import mesh_2_grid
from porepy.grids.constants import GmshConstants

from dfm_upscaling import interaction_region as ia_reg

import pdb


class LocalGridBucketSet:
    def __init__(self, dim, reg):

        self.dim = dim
        self.reg = reg
        self.tol = 1e-6

    def construct_local_buckets(self):

        if self.dim == 2:
            self._construct_buckets_2d()
        elif self.dim == 3:
            self._construct_buckets_3d()

    def _construct_buckets_2d(self):

        gb, network, file_name = self.reg.mesh()

        # for 2d problems, the physical (gmsh) tags can also be used to identify
        # individual interaction regions (this follows form how the gmsh .geo file) is
        # set up in gmsh.
        network.decomposition["edges"] = network.decomposition["edges"][[0, 1, 2, 3, 3]]

        self.buckets_2d = [gb]

        self._recover_line_gb(network, file_name)

    def _construct_buckets_3d(self):
        gb, network, file_name = self.reg.mesh()

        self.buckets_2d = [gb]

        decomp = network.decomposition

        edges = decomp["edges"]
        edge_tags = decomp["edge_tags"]

        def edge_indices(subset, edges):
            """ Helper function to find a subset of edges in the full edge set

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
        for edge, node_type in zip(reg.edges, reg.edge_node_type):
            coords = np.zeros((3, 0))
            for e, node in zip(edge, node_type):
                coords = np.hstack((coords, reg._coord(node, e)))

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

        network.decomposition["edges"] = np.vstack(
            (edges, edge_tags, physical_line_counter, ia_reg_edge_numbering)
        )

        self._recover_line_gb(network, file_name)

    def _recover_line_gb(self, network, file_name):
        """ We will use the following keys / items in network.decomposition:

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
        decomp = network.decomposition
        # Recover the full description of the gmsh mesh
        mesh = meshio.read(file_name + ".msh")

        # Invert the meshio field_data so that phys_names maps from the tags that gmsh
        # assigns to XXX, to the physical names
        phys_names = {v[0]: k for k, v in mesh.field_data.items()}

        pts = mesh.points

        gmsh_constants = GmshConstants()
        # Create all 1d grids that correspond to a domain boundary
        g_1d = mesh_2_grid.create_1d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            line_tag=gmsh_constants.PHYSICAL_NAME_DOMAIN_BOUNDARY,
            return_fracture_tips=False,
        )

        # Create 0d grid that corresponds to a domain boundary point.
        # Some of these are in cell centers of the coarse grid, while others are on
        # faces. Only some of the grids are included in the local grid buckets -
        # specifically the cell centers grids may not be needed - but it is easier to
        # create all, and then dump those not needed.
        g_0d_domain_boundary = mesh_2_grid.create_0d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            target_tag_stem=gmsh_constants.PHYSICAL_NAME_BOUNDARY_POINT,
        )
        # Create a mapping from the domain boundary points to the 0d grids
        # The keys are the indexes in the decomposition of the network.
        domain_point_2_g = {}
        for g in g_0d_domain_boundary:
            domain_point_2_g[
                decomp["domain_boundary_points"][g.physical_name_index]
            ] = g

        # Similarly, construct 0d grids for the intersection between a fracture and the
        # edge of a domain
        g_0d_frac_bound = mesh_2_grid.create_0d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            target_tag_stem=gmsh_constants.PHYSICAL_NAME_FRACTURE_BOUNDARY_POINT,
        )
        # A map from fracture points on the domain boundary to the 0d grids.
        # The keys are the indexes in the decomposition of the network.
        frac_bound_point_2_g = {}
        for g in g_0d_frac_bound:
            frac_bound_point_2_g[
                decomp["fracture_boundary_points"][g.physical_name_index]
            ] = g

        # Get the points that form the boundary of the interaction region
        boundary_point_coord, boundary_point_ind = self._network_boundary_points(
            network
        )

        # Find all edges that are marked as a domain_boundary
        bound_edge = decomp["edges"][2] == gmsh_constants.DOMAIN_BOUNDARY_TAG
        # Find the index of edges that are associated with the domain boundary. Each
        # part of the boundary may consist of a single line, or several edges that are
        # split either by fractures, or by other auxiliary points.
        bound_edge_ind = np.unique(decomp["edges"][-1][bound_edge])

        # Mapping from the frac_num, which **should** (TODO!) be equivalent to the
        # edge numbering in decomp[edges][3], thus bound_edge_ind, to 1d grids on the
        # domain boundary.
        bound_edge_ind_2_g = {g.frac_num: g for g in g_1d}

        # Data structure for storage of 1d grids
        buckets_1d = []

        # Loop over the edges in the interaction region
        for ia_edge, node_type in zip(reg.edges, reg.edge_node_type):

            # Recover coordinates of the edge points
            ia_edge_coord = np.zeros((3, 0))
            for e, t in zip(ia_edge, node_type):
                ia_edge_coord = np.hstack((ia_edge_coord, reg._coord(t, e)))

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
            else:
                # This should require dropping the last edge, coord, but most other
                # parts should be the same
                raise NotImplementedError("Have not considered global boundaries yet")

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
            # To get the right grid, first map the midpoint on the ia_edge to the right
            # boundary point, and then access the dictionary of all boundary point grids
            edge_grids_0d = [domain_point_2_g[boundary_point_ind[domain_pt_ia_edge[1]]]]

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
                            edge_grids_1d.append(
                                bound_edge_ind_2_g[decomp["edges"][-2, ind]]
                            )

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
            # Create a grid bucket for this edge
            gb_edge = pp.meshing.grid_list_to_grid_bucket(grid_list)

            # Append the bucket for this ia_edge
            buckets_1d.append(gb_edge)

        # Store all edge buckets for this region
        self.buckets_1d = buckets_1d

    def _match_points(self, p1, p2):
        """ Find occurences of coordinates in the second array within the first array.

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

    def _network_boundary_points(self, network):
        boundary_ind = network.decomposition["domain_boundary_points"]
        boundary_points = network.decomposition["points"][:, boundary_ind]

        if boundary_points.shape[0] == 2:
            boundary_points = np.vstack((boundary_points, np.zeros(boundary_ind.size)))

        return boundary_points, boundary_ind


if __name__ == "__main__":
    from dfm_upscaling.utils import create_grids

    interior_face = 4
    if False:
        g = create_grids.cart_2d()

        p = np.array([[0.7, 1.3], [1.0, 1.5]])
        edges = np.array([[0], [1]])

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]
        # reg = ia_reg.extract_mpfa_regions(g, nodes=[interior_face])[0]
        reg.add_fractures(points=p, edges=edges)

        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()

    else:
        g = create_grids.cart_3d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]
        reg = ia_reg.extract_mpfa_regions(g, nodes=[13])[0]

        frac = pp.Fracture(
            np.array([[0.7, 1.4, 1.4, 0.7], [0.5, 0.5, 1.4, 1.4], [0.2, 0.2, 0.8, 0.8]])
        )
        reg.add_fractures(fractures=[frac])

        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()
