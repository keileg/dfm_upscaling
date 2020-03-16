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

    def construct_local_buckets(self):

        if self.dim == 2:
            self._construct_buckets_2d()

    def _construct_buckets_2d(self):
        gb, network, file_name = reg.mesh()
        
        self.buckets_2d = [gb]

        decomp = network.decomposition

        # Recover the full description of the gmsh mesh
        mesh = meshio.read(file_name + ".msh")

        # Invert the meshio field_data so that phys_names maps from the tags that gmsh
        # assigns to XXX, to the physical names
        phys_names = {v[0]: k for k, v in mesh.field_data.items()}

        pts = mesh.points

        gmsh_constants = GmshConstants()
        # Create all 1d grids
        g_1d = mesh_2_grid.create_1d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            line_tag=gmsh_constants.PHYSICAL_NAME_DOMAIN_BOUNDARY,
            return_fracture_tips=False,
        )

        g_0d_domain_boundary = mesh_2_grid.create_0d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            target_tag_stem=gmsh_constants.PHYSICAL_NAME_BOUNDARY_POINT,
        )

        domain_point_2_g = {}
        for g in g_0d_domain_boundary:
            domain_point_2_g[
                decomp["domain_boundary_points"][g.physical_name_index]
            ] = g

        g_0d_frac_bound = mesh_2_grid.create_0d_grids(
            pts,
            mesh.cells,
            phys_names,
            mesh.cell_data,
            target_tag_stem=gmsh_constants.PHYSICAL_NAME_FRACTURE_BOUNDARY_POINT,
        )

        frac_bound_point_2_g = {}
        for g in g_0d_frac_bound:
            frac_bound_point_2_g[
                decomp["fracture_boundary_points"][g.physical_name_index]
            ] = g

        # Get the points that form the boundary of the interaction region
        boundary_point_coord, boundary_point_ind = self._network_boundary_points(network)

        bound_edge = decomp["edges"][2] == gmsh_constants.DOMAIN_BOUNDARY_TAG
        bound_edge_ind = np.unique(decomp["edges"][3][bound_edge])

        bound_edge_ind_2_g = {g.frac_num: g for g in g_1d}
        
        buckets_1d = []

        # Loop over the edges in the interaction region
        for edge_ind, (edge, node_type) in enumerate(
            zip(reg.edges, reg.edge_node_type)
        ):
            # Recover coordinates of the edge points
            edge_coord = np.zeros((3, 0))
            for e, t in zip(edge, node_type):
                edge_coord = np.hstack((edge_coord, reg._coord(t, e)))

            # Match points in the region with points in the network
            edge_decomp = self._match_points(edge_coord, boundary_point_coord)

            # Next step: Find the g_1d grids that belong to the parts of the edge
            # This is one full edge, in the interaction region sense
            edge_vertexes = np.vstack(
                (
                    boundary_point_ind[edge_decomp][:-1],
                    boundary_point_ind[edge_decomp][1:],
                )
            )

            # The 1d grids that run over an edge
            edge_grids_1d = []

            # The 0d grids that connects the parts of this interaction region edge.
            # These will be located on the middle point of the edge.
            edge_grids_0d = [domain_point_2_g[pi] for pi in edge_vertexes[0, 1:]]

            fracture_point_grids_0d = []

            num_edge_vertexes = edge_vertexes.shape[1]

            # Loop over the parts of this edge. This will be one straight line that forms a
            # part of the domain boundary
            for partial_edge in range(num_edge_vertexes):
                loc_reg_edge = edge_vertexes[:, partial_edge]
                found = False

                # Loop over all boundaries in the decomposition. This should have start and
                # endpoints corresponding to one of the loc_edge_edge, and possible other
                # points in between
                for bi in bound_edge_ind:

                    # edge_components contains the points in the decomposition of the domain that
                    # together form an edge on the domain boundary
                    edge_components = decomp["edges"][:2, decomp["edges"][3] == bi]

                    sorted_edge_components, _ = pp.utils.sort_points.sort_point_pairs(
                        edge_components, is_circular=False
                    )

                    # The end points of this boundary edge, in the decomposition, are
                    # sorted_edge[0, 0] and sorted_edge[1, -1]
                    assert (
                        sorted_edge_components[0, 0] in decomp["domain_boundary_points"]
                    )
                    assert (
                        sorted_edge_components[1, -1]
                        in decomp["domain_boundary_points"]
                    )

                    if (
                        loc_reg_edge[0] == sorted_edge_components[0, 0]
                        and loc_reg_edge[1] == sorted_edge_components[1, -1]
                    ) or (
                        loc_reg_edge[1] == sorted_edge_components[0, 0]
                        and loc_reg_edge[0] == sorted_edge_components[1, -1]
                    ):
                        found = True

                        # Add this grid to the set of 1d grids along the region edge
                        edge_grids_1d.append(bound_edge_ind_2_g[bi])

                        # Add all 0d grids to the set of 0d grids along the region edge
                        # This will be fracture points, should inherit properties from
                        # the fracture.
                        for pi in sorted_edge_components[0, 1:]:
                            if pi in frac_bound_point_2_g.keys():
                                fracture_point_grids_0d.append(frac_bound_point_2_g[pi])

                        break

                assert found

            grid_list = [edge_grids_1d, edge_grids_0d + fracture_point_grids_0d]

            gb_edge = pp.meshing.grid_list_to_grid_bucket(grid_list)
            
            buckets_1d.append(gb_edge)

        self.buckets_1d = buckets_1d
        
        
    def _match_points(self, p1, p2, tol=1e-4):
        p_all = np.hstack((p1, p2))
    
        for p in (p1, p2):
            p_unique, _, _ = unique_columns_tol(p, tol)
            if p_unique.shape[1] != p.shape[1]:
                raise ValueError("Original point sets should be unique")
    
        _, _, new_2_all = unique_columns_tol(p_all, tol)
    
        doubles = np.where(np.bincount(new_2_all) > 1)[0]
    
        pairs_first = []
    
        for ind in doubles:
            hit = np.where(new_2_all == ind)[0]
            assert hit.size == 2
            assert hit[0] < p1.shape[1] and hit[1] >= p1.shape[1]
            pairs_first.append(hit[1] - p1.shape[1])
    
        return pairs_first
    
    
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

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]
        reg = ia_reg.extract_mpfa_regions(g, nodes=[interior_face])[0]
        reg.add_fractures(points=p, edges=edges)
        
        local_buckets = LocalGridBucketSet(2, reg)

    else:
        g = create_grids.cart_3d()
        
    local_buckets.construct_local_buckets()
