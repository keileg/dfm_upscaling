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
from typing import Dict

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

    def bucket_list(self):
        if self.dim == 2:
            return [self.line_gb, [self.gb]]
        else:
            return [self.line_gb, self.surface_gb, [self.gb]]

    def construct_local_buckets(self):

        if self.dim == 2:
            self._construct_buckets_2d()
        elif self.dim == 3:
            self._construct_buckets_3d()

        self._tag_faces_macro_boundary(self.gb)

    def _construct_buckets_2d(self):

        gb, network, file_name = self.reg.mesh()
        self.network = network
        self.gb = gb
        self.file_name = file_name

        for g, _ in gb:
            if g.dim < self.dim:
                g.from_fracture = True

        gmsh_constants = GmshConstants()

        # We need to define point tags, which are assumed to exist by
        # self._recover_line_gb()
        edges = network.decomposition["edges"]

        # Each point should be classified as either boundary, fracture or fracture and
        # boundary, according to which edges share the point

        # Initialize by a neutral tag
        point_tags = gmsh_constants.NEUTRAL_TAG * np.ones(
            network.decomposition["points"].shape[1], dtype=np.int
        )

        # Find the points of boundary and fracture edges
        boundary_points = np.union1d(
            edges[0, edges[2] == gmsh_constants.DOMAIN_BOUNDARY_TAG],
            edges[1, edges[2] == gmsh_constants.DOMAIN_BOUNDARY_TAG],
        )
        fracture_points = np.union1d(
            edges[0, edges[2] == gmsh_constants.FRACTURE_TAG],
            edges[1, edges[2] == gmsh_constants.FRACTURE_TAG],
        )

        # Split into fracture, boundary or both
        fracture_boundary_points = np.intersect1d(boundary_points, fracture_points)
        only_fracture_points = np.setdiff1d(fracture_points, boundary_points)
        only_boundary_points = np.setdiff1d(boundary_points, fracture_points)

        # Tag accordingly
        point_tags[
            fracture_boundary_points
        ] = gmsh_constants.FRACTURE_LINE_ON_DOMAIN_BOUNDARY_TAG
        point_tags[only_fracture_points] = gmsh_constants.FRACTURE_TAG
        point_tags[only_boundary_points] = gmsh_constants.DOMAIN_BOUNDARY_TAG

        # Store information
        network.decomposition["point_tags"] = point_tags

        # for 2d problems, the physical (gmsh) tags can also be used to identify
        # individual interaction regions (this follows form how the gmsh .geo file is
        # set up).
        network.decomposition["edges"] = network.decomposition["edges"][[0, 1, 2, 3, 3]]

        self._recover_line_gb(network, file_name)

    def _construct_buckets_3d(self):
        gb, network, file_name = self.reg.mesh()

        for g, _ in gb:
            if g.dim < self.dim:
                g.from_fracture = True

        self.gb = gb
        self.network = network
        self.file_name = file_name

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
        for edge, node_type in zip(self.reg.edges, self.reg.edge_node_type):
            coords = np.zeros((3, 0))
            for e, node in zip(edge, node_type):
                coords = np.hstack((coords, self.reg._coord(node, e)))

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
        self._recover_surface_gb(network, file_name)

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
        fracture_boundary_points = np.where(
            decomp["point_tags"] == gmsh_constants.FRACTURE_LINE_ON_DOMAIN_BOUNDARY_TAG
        )[0]

        # Assign the 0d grids an attribute g.from_fracture, depending on whether it
        # coincides with a fracture or is an auxiliary point
        for g in g_0d_frac_bound:
            g.from_fracture = True

        for g in g_0d_domain_boundary:
            g.from_fracture = False

        for g in g_1d:
            g.compute_geometry()
            g.from_fracture = False

        # A map from fracture points on the domain boundary to the 0d grids.
        # The keys are the indexes in the decomposition of the network.
        frac_bound_point_2_g = {}
        for g in g_0d_frac_bound:
            frac_bound_point_2_g[fracture_boundary_points[g.physical_name_index]] = g

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
        for ia_edge, node_type in zip(self.reg.edges, self.reg.edge_node_type):

            # Recover coordinates of the edge points
            ia_edge_coord = np.zeros((3, 0))
            for e, t in zip(ia_edge, node_type):
                ia_edge_coord = np.hstack((ia_edge_coord, self.reg._coord(t, e)))

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
                edge_grids_0d = [
                    domain_point_2_g[boundary_point_ind[domain_pt_ia_edge[1]]]
                ]

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
                            if not g_loc in edge_grids_1d:
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
            # Create a grid bucket for this edge
            gb_edge = pp.meshing.grid_list_to_grid_bucket(grid_list)

            # Append the bucket for this ia_edge
            buckets_1d.append(gb_edge)

        # Store all edge buckets for this region
        for gb in buckets_1d:
            self._tag_faces_macro_boundary(gb)

        self.line_gb = buckets_1d

    def _recover_surface_gb(self, network, file_name):

        # Network decomposition
        decomp = network.decomposition

        # Recover the full description of the gmsh mesh
        mesh = meshio.read(file_name + ".msh")

        # Invert the meshio field_data so that phys_names maps from the tags that gmsh
        # assigns to XXX, to the physical names
        phys_names = {v[0]: k for k, v in mesh.field_data.items()}

        # Mesh points
        pts = mesh.points

        # We need to recover four types of grids:
        #  1) 2d grids on the domain surfaces
        #  2) 1d grids formed by the intersection of domain surfaces and fractures
        #  3) 1d grids formed at the junction between 2d domain surface grids
        #  4) 0d grids formed at the intersection between 1d fracture line grids

        gmsh_constants = GmshConstants()

        # Create all 2d grids that correspond to a domain boundary
        g_2d_all = mesh_2_grid.create_2d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            is_embedded=True,
            network=network,
            surface_tag=gmsh_constants.PHYSICAL_NAME_DOMAIN_BOUNDARY_SURFACE,
        )

        index_offset = min([g.frac_num for g in g_2d_all])

        g_2d = [
            g
            for g in g_2d_all
            if not self.reg.surface_is_boundary[g.frac_num - index_offset]
        ]

        # Map form the frac_num (which by construction in pp.mesh_2_grid will correspond
        # to the number part of the gmsh physical name of the surface polygon) to the
        # corresponding physical grid.
        # Note that we need to be careful when accessing this information below, since
        # the frac_num also will count any fracture surface in the network.
        g_2d_map = {g.frac_num: g for g in g_2d}

        # 1d grids formed on the intersection of fracture surfaces with the domain
        # boundary
        g_1d = mesh_2_grid.create_1d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            line_tag=gmsh_constants.PHYSICAL_NAME_FRACTURE_BOUNDARY_LINE,
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
        g_1d_auxiliary = mesh_2_grid.create_1d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            line_tag=gmsh_constants.PHYSICAL_NAME_DOMAIN_BOUNDARY,
            constraints=ia_edge,
            return_fracture_tips=False,
        )

        # Points that are tagged as both on a fracture and on the domain boundary
        fracture_boundary_points = np.where(
            decomp["point_tags"] == gmsh_constants.FRACTURE_LINE_ON_DOMAIN_BOUNDARY_TAG
        )[0]

        # Create grids for physical points on the boundary surfaces. This may be both
        # on domain edges, in the meeting of surface and fracture polygons, and by the
        # meeting of fracture surfaces within a boundary surface.
        g_0d = mesh_2_grid.create_0d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            target_tag_stem=gmsh_constants.PHYSICAL_NAME_FRACTURE_BOUNDARY_POINT,
        )

        # Assign the 1d and 0d grids an attribute g.from_fracture, depending on whether
        # they coincide with a fracture or are auxiliary
        for g in g_1d:
            g.from_fracture = True

        for g in g_1d_auxiliary:
            g.from_fracture = False

        for g in g_0d:
            g.from_fracture = True

        # Map from the fracture boundary points, in the network decomposition index, to
        # the corresponding 0d grids
        g_0d_map = {}
        for g in g_0d:
            g_0d_map[fracture_boundary_points[g.physical_name_index]] = g

        # We now have all the grids needed. The next step is to group them into surfaces
        # that are divided by the interaction region edges. Specifically, 2d surface
        # grids will be joined if they are divided by an auxiliary 1d grid.

        # Data structures for storing pairs of 2d surface and 1d auxiliary grids
        pairs = []
        # bookkeeping
        num_2d_grids = len(g_2d)

        # Loop over all surface grids, find all auxiliary grids with which is share
        # nodes (a surface grid face should coincide with a 1d cell).
        # This code is borrowed from pp.meshing.grid_list_to_grid_bucket()
        # It is critical that the operation is carried out before splitting of the
        # nodes, or else the local-to-global node numbering is not applicable.
        for hi, hg in enumerate(g_2d):
            # We have to specify the number of nodes per face to generate a
            # matrix of the nodes of each face.
            nodes_per_face = 2
            fn_loc = hg.face_nodes.indices.reshape(
                (nodes_per_face, hg.num_faces), order="F"
            )
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            for li, lg in enumerate(g_1d_auxiliary):
                cell_2_face, cell = pp.fracs.tools.obtain_interdim_mappings(
                    lg, fn, nodes_per_face
                )
                if cell_2_face.size > 0:
                    # We have found a new pair. Adjust the counting of the 1d grid index
                    # with the number of 2d surface grids
                    pairs.append((hi, li + num_2d_grids))

        # To find the isolated components, make the pairs into a graph.
        graph = nx.Graph()
        for couple in pairs:
            graph.add_edge(couple[0], couple[1])

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

        # Create mappings from the surface grids to its embedded 1d and 0d grids
        g_2d_2_frac_g_map = {}
        g_2d_2_0d_g_map = {}

        # Loop over the surface grids, find its embedded lower-dimensional grids
        for si in np.where(network.tags["boundary"])[0]:
            if self.reg.surface_is_boundary[si - index_offset]:
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
            g_2d_2_0d_g_map[g_surf] = [g_0d_map[i] for i in intersection_nodes]

        # Finally, we can collect the surface grid buckets. There will be one for each
        # cluster, identified above.
        # Data structure
        surface_buckets = []

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
                    g0 += list(g_2d_2_0d_g_map[g_surf])
                # .. or a 1d auxiliary grid
                else:
                    # Here we need to adjust the grid index, to account for the
                    # numbering used in defining the pairs above
                    g1 += [g_1d_auxiliary[grid_ind - num_2d_grids]]

            # Make list, make bucket, store it.
            grid_list = [g2, g1, g0]
            gb_loc = pp.meshing.grid_list_to_grid_bucket(grid_list)
            surface_buckets.append(gb_loc)

        # Done!
        for gb in surface_buckets:
            self._tag_faces_macro_boundary(gb)

        self.surface_gb = surface_buckets

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

    def _tag_faces_macro_boundary(self, gb: pp.GridBucket) -> None:
        """
        In the main GridBucket, tag faces that coincide with the boundary of the macro
        domain.

        Returns:
            None.

        """
        reg = self.reg

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
            pts = np.empty((3, 0))
            for ind, node in zip(surf, node_type):
                pts = np.hstack((pts, reg._coord(node, ind)))

            # Loop over all grids in the main gb, and look for faces on the surface.
            for g, _ in gb:
                # Grids of co-dimension > 1 will not be assigned a bc on the macro
                # boundary
                if g.dim < reg.dim - 1:
                    continue

                # Find face centers on the region surface
                fc = g.face_centers
                if reg.dim == 2:
                    dist, _ = pp.distances.points_segments(fc, pts[:, 0], pts[:, 1])
                else:  # reg.dim == 3
                    dist, *_ = pp.distances.points_polygon(fc, pts, tol=self.tol)

                on_bound = np.where(dist < self.tol)[0]

                if on_bound.size > 0:
                    # Append the micro faces to the list of found indices
                    micro_map[g] = np.hstack((micro_map[g], on_bound))
                    # The macro face index needs some more work.
                    if reg.name == "tpfa":
                        # For tpfa-style ia regions, we can use the region index
                        macro_ind = reg.reg_ind * np.ones(on_bound.size, dtype=np.int)
                    else:  # mpfa
                        # in mpfa, look for the face among the surface nodes (there will
                        # be a single one), use this.
                        # If a ValueError is raised here, there is no 'face' in
                        # node_type, and something is really wrong
                        macro_ind = surf[node_type.index("face")] * np.ones(
                            on_bound.size, dtype=np.int
                        )
                    # Store the information
                    macro_map[g] = np.hstack((macro_map[g], macro_ind))

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
        boundary_ind = network.decomposition["domain_boundary_points"]
        boundary_points = network.decomposition["points"][:, boundary_ind]

        if boundary_points.shape[0] == 2:
            boundary_points = np.vstack((boundary_points, np.zeros(boundary_ind.size)))

        return boundary_points, boundary_ind


if __name__ == "__main__":
    from dfm_upscaling.utils import create_grids

    interior_face = 4
    if True:
        g = create_grids.cart_2d()

        p = np.array([[0.7, 1.3], [1.0, 1.5]])
        edges = np.array([[0], [1]])

        reg = ia_reg.extract_tpfa_regions(g, faces=[3])[0]
        reg = ia_reg.extract_mpfa_regions(g, nodes=[3])[0]
        reg.add_fractures(points=p, edges=edges)

        local_gb = LocalGridBucketSet(2, reg)
        local_gb.construct_local_buckets()

    else:
        g = create_grids.cart_3d()
        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]
        #  reg = ia_reg.extract_mpfa_regions(g, nodes=[13])[0]

        f_1 = pp.Fracture(
            np.array([[0.7, 1.4, 1.4, 0.7], [0.5, 0.5, 1.4, 1.4], [0.2, 0.2, 0.8, 0.8]])
        )

        f_2 = pp.Fracture(
            np.array([[0.3, 0.3, 1.4, 1.4], [0.7, 1.4, 1.4, 0.7], [0.1, 0.1, 0.9, 0.9]])
        )

        reg.add_fractures(fractures=[f_1, f_2])

        local_gb = LocalGridBucketSet(3, reg)
        local_gb.construct_local_buckets()

        assert False
