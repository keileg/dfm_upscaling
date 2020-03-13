#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:32:39 2020

@author: eke001
"""
import numpy as np
import porepy as pp

import pdb


class InteractionRegion:
    def __init__(self, g: pp.Grid, name: str, reg_ind: int, central_node=None):
        self.g = g
        self.dim = g.dim
        self.name = name

        self.reg_ind = reg_ind

        if self.dim == 2:
            self.pts = np.zeros((2, 0))
            self.edges = np.zeros((2, 0))

            self.fracture_pts = np.zeros((3, 0))
            self.fracture_edges = np.zeros((2, 0), dtype=np.int)
        else:
            self.fractures = []

        if central_node is not None:
            self.node_ind = central_node
            self.node_coord = g.nodes[:, central_node].reshape((-1, 1))

    ####################
    ## Functions related to meshing of fracture networks
    ####################

    def mesh(self):
        """ Create a local mesh for this interaction region.
        """
        if self.dim == 2:
            return self._mesh_2d()
        else:
            return self._mesh_3d()

    def _mesh_2d(self):
        """ To create a local grid bucket in 2d, we should:
            1) Create the bounding surfaces, from self.surfaces
                i) Find coordinates of all surfaces
                ii) Remove duplicate nodes
                iii) Create a FractureNetwork2d object, create a mesh

        """

        # First, build points and edges for the domain boundary
        domain_pts = np.zeros((3, 0))
        domain_edges = np.zeros((2, 0), dtype=np.int)
        edge_2_surf = np.empty([], dtype=np.int)

        for surf_ind, (surf, node_type) in enumerate(
            zip(self.surfaces, self.surface_node_type)
        ):

            e = np.vstack(
                (np.arange(len(node_type) - 1), 1 + np.arange(len(node_type) - 1))
            )
            # The new edges are offset by the number of previous points
            domain_edges = np.hstack((domain_edges, domain_pts.shape[1] + e))

            # Then add new points
            for ind, node in zip(surf, node_type):
                domain_pts = np.hstack((domain_pts, self._coord(node, ind)))

            edge_2_surf = np.hstack((edge_2_surf, surf_ind + np.ones(e.shape[1])))

        # Next, build up the constraints
        # Todo: Expand this with fractures contained within the region
        edge_2_constraint = np.array([], dtype=np.int)
        constraint_edges = np.empty((2, 0), dtype=np.int)
        constraint_pts = np.empty((3, 0))

        for constraint_ind, (constraint, node_type) in enumerate(
            zip(self.constraints, self.constraint_node_type)
        ):
            e = np.vstack(
                (np.arange(len(node_type) - 1), 1 + np.arange(len(node_type) - 1))
            )
            # The new edges are offset by the number of previous points
            constraint_edges = np.hstack(
                (constraint_edges, constraint_pts.shape[1] + e)
            )
            # Then add new points
            for ind, node in zip(constraint, node_type):
                constraint_pts = np.hstack((constraint_pts, self._coord(node, ind)))

            edge_2_constraint = np.hstack(
                (edge_2_constraint, constraint_ind * np.ones(e.shape[1], dtype=np.int))
            )

        # Uniquify points on the domain boundary
        unique_domain_pts, _, all_2_unique = pp.utils.setmembership.unique_columns_tol(domain_pts)
        unique_domain_edges = all_2_unique[domain_edges]
        # Also sort the boundary points to form a circle
        sorted_edges, sort_ind = pp.utils.sort_points.sort_point_pairs(unique_domain_edges)

        constraint_edges += self.fracture_pts.shape[1]

        int_pts = np.hstack((self.fracture_pts, constraint_pts))
        int_edges = np.hstack((self.fracture_edges, constraint_edges))

        # Similarly uniquify points in constraint description

        unique_int_pts, _, a2u = pp.utils.setmembership.unique_columns_tol(int_pts)
        unique_int_edges = a2u[int_edges]

        # Define a fracture network, using the surface specification as boundary,
        # and the constraints as points
        # Fractures will be added as edges
        network = pp.FractureNetwork2d(
            domain=unique_domain_pts[: self.dim, sorted_edges[0]],
            pts=unique_int_pts[: self.dim],
            edges=unique_int_edges,
        )

        mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_bound": 0.5,
            "mesh_size_min": 0.3,
        }

        file_name = "gmsh_upscaling_region_" + str(self.reg_ind)

        gb = network.mesh(
            mesh_args=mesh_args, file_name=file_name, constraints=edge_2_constraint
        )

        return gb, network, file_name

    def _mesh_3d(self):

        boundaries = []
        for surf_ind, (surf, node_type) in enumerate(
            zip(self.surfaces, self.surface_node_type)
        ):
            pts = np.empty((3, 0))

            # Then add new points
            for ind, node in zip(surf, node_type):
                pts = np.hstack((pts, self._coord(node, ind)))

            boundaries.append(pts)

        constraints = []

        for constraint_ind, (constraint, node_type) in enumerate(
            zip(self.constraints, self.constraint_node_type)
        ):
            pts = np.empty((3, 0))
            # Then add new points
            for ind, node in zip(constraint, node_type):
                pts = np.hstack((pts, self._coord(node, ind)))

            constraints.append(pp.Fracture(pts))

        polygons = self.fractures + constraints

        constraint_inds = len(self.fractures) + np.arange(len(constraints))

        network = pp.FractureNetwork3d(polygons)
        network.impose_external_boundary(boundaries)

        mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_bound": 0.5,
            "mesh_size_min": 0.3,
        }

        file_name = "gmsh_upscaling_region_" + str(self.reg_ind)

        gb = network.mesh(
            mesh_args=mesh_args,
            file_name=file_name,
            constraints=constraint_inds,
        )

        return gb, network, file_name

    def _coord(self, node: str, ind: int):
        if node == "cell":
            p = self.g.cell_centers[:, ind].reshape((-1, 1))
        elif node == "face":
            p = self.g.face_centers[:, ind].reshape((-1, 1))
        elif node == "node":
            p = self.g.nodes[:, ind].reshape((-1, 1))
        elif node == "edge":
            p = 0.5 * (self.g.nodes[:, ind].reshape((-1, 1)) + self.node_coord)
        else:
            raise ValueError("Unknown node type " + node)

        return p


    def add_fractures(self, points=None, edges=None, fractures=None):
        if self.dim == 3:
            self.fractures = fractures
        else:
            if points.shape[0] == 2:
                points = np.vstack((points, np.zeros(points.shape[1])))
            self.fracture_pts = points
            self.fracture_edges = edges



def extract_tpfa_regions(g: pp.Grid, faces=None):
    """ Define tpfa-type interaction regions for specified faces in a grid.

    Parameters:
        g (pp.Grid): Grid where interaction regions will be found
        faces (int or np.array, optional): Index of faces for which the regions will be found.
            If not provided, all faces will have their region computed.

    Returns:
        list of np.ndarray: Each list element contains the interaction region
            for one of the specified faces. The array is n x 3, each row defines
            a boundary face in the interaction region, by refering to indexes in
            the grid.
        np.array: Index of the faces in the order used in the first return value.
            Identical to the face array parameter if this is provided.
        tuple of str, length 3: Inform which type of indices the columns in the
            index array referes to. Gives values 'cell' and 'node'.

    """

    # Process input.
    if faces is None:
        faces = np.arange(g.num_faces)
    elif isinstance(faces, int):
        faces = np.array([faces])
    else:
        faces = np.asarray(faces)

    # Get a dense array version of the cell-face map.
    cell_faces = g.cell_face_as_dense()

    # Data structure for the output
    region_list = []

    # The interaction region consists of

    # For TPFA, the interaction regions are centered on the faces in the coarse grid
    # Loop over all specified faces, find their interaction region
    for fi in faces:

        # Find cells neighboring this face
        cells = cell_faces[:, fi]
        # By construction of the dense face-cell relation, cell index of -1
        # indicates outside the outer boundary - skip these cells.
        cells = cells[cells >= 0]

        # Special marker for boundary cells
        on_boundary = cells.size == 1

        # Nodes of the central face
        nodes = g.face_nodes[:, fi].indices
        # Extend the nodes so that we can combine pairs
        nodes_extended = np.hstack((nodes, nodes[0]))

        # Storage for this interaction regions
        tri = []  # Surfaces
        c2c = []  # edges that connect cell centers

        surface_node_type = []
        surface_is_boundary = []
        edge_node_type = []

        # The faces of the interaction region are now formed by the connecting
        # one cell center with two consequtive nodes in the face-node map.
        # This assumes a circular ordering of the face-node relation.
        for ci in cells:
            if g.dim == 2:
                tri.append((ci, nodes[0]))
                surface_node_type.append(("cell", "node"))
                surface_is_boundary.append(False)

                tri.append((ci, nodes[1]))
                surface_node_type.append(("cell", "node"))
                surface_is_boundary.append(False)
            else:
                for ni in range(nodes.size):
                    # The face of the interaction region consists of one cell
                    # and two nodes, in that order.
                    tri.append((ci, nodes_extended[ni], nodes_extended[ni + 1]))
                    surface_node_type.append(("cell", "node", "node"))
                    surface_is_boundary.append(False)

        for n in nodes:
            if on_boundary:
                c2c.append((cells[0], n))
                edge_node_type.append(("cell", "node"))
            else:
                c2c.append((cells[0], n, cells[1]))
                edge_node_type.append(("cell", "node", "cell"))

        if on_boundary:
            if g.dim == 2:
                tri.append(nodes)
                surface_node_type.append(("node", "node"))
                surface_is_boundary.append(True)
            else:
                if nodes.size == 3:
                    tri.append(nodes)
                    surface_node_type.append(("node", "node", "node"))
                    surface_is_boundary.append(True)
                elif nodes.size == 4:
                    tri.append(nodes[:3])
                    surface_node_type.append(("node", "node", "node"))
                    surface_is_boundary.append(True)

                    tri.append((nodes[0], nodes[2], nodes[3]))
                    surface_node_type.append(("node", "node", "node"))
                    surface_is_boundary.append(True)
                else:
                    raise ValueError(
                        "Implementation only covers simplexes and Cartisan grids"
                    )

        # Combine the full set of boundary faces of this interaction region
        # into an np.array, and store it in the global list
        surfaces = np.vstack([t for t in tri])

        edges = np.vstack([c for c in c2c])

        reg = InteractionRegion(g, "tpfa", fi)
        reg.surfaces = surfaces

        # Which type of grid element the boundaries of the interaction regions are
        # formed by. Needed for lookup of geometry (the first index refers to cell center etc.)
        reg.surface_node_type = surface_node_type
        reg.surface_is_boundary = surface_is_boundary

        reg.edges = edges
        reg.edge_node_type = edge_node_type

        if on_boundary:
            reg.constraints = []
            reg.constraint_node_type = []
        else:
            reg.constraints = [nodes]
            reg.constraint_node_type = [nodes.size * ["node"]]

        region_list.append(reg)

    # Probably return an interaction region, or a named tuple
    return region_list


def extract_mpfa_regions(g: pp.Grid, nodes=None):

    if g.dim < 2:
        raise ValueError("Implementation is only valid for 2d and 3d")

    if nodes is None:
        nodes = np.arange(g.num_nodes)
    elif isinstance(nodes, int):
        nodes = np.array([nodes])
    else:
        nodes = np.asarray(nodes)

    # Get a dense array version of the cell-face map.
    cell_faces = g.cell_face_as_dense()

    region_list = []

    cn = g.cell_nodes().tocsr()
    fn = g.face_nodes.tocsr()

    for ni in nodes:

        loc_cells = cn[ni].indices
        loc_faces = fn[ni].indices
        nc = loc_cells.size
        nf = loc_faces.size

        on_boundary = np.any(cell_faces[:, loc_faces] < 0)

        loc_cells_unique_ind_glob = -np.ones(g.num_cells, dtype=np.int)
        loc_cells_unique_ind_glob[loc_cells] = np.arange(nc, dtype=np.int)

        loc_faces_unique_ind_glob = -np.ones(g.num_faces, dtype=np.int)
        loc_faces_unique_ind_glob[loc_faces] = nc + np.arange(nf, dtype=np.int)

        tmp_edges = []
        edge_node_type = []
        surface_node_type = []
        surface_is_boundary = []

        constraints = []
        constraints_node_type = []

        if g.dim == 2:
            tmp_surfaces = []

        else:
            tmp_surfaces = []
            other_node, face_of_edge = _find_edges(g, loc_faces, ni)

        for fi in loc_faces:
            # If we're on a boundary, this will lead to trouble here
            if np.any(cell_faces[:, fi] < 0):
                boundary_face = True
                # This is a boundary face
                edge_node_type.append(("cell", "face"))
                if cell_faces[0, fi] < 0:
                    ci = cell_faces[1, fi]
                else:
                    ci = cell_faces[0, fi]
                tmp_edges.append(np.array([ci, fi]))
            else:
                boundary_face = False
                tmp_edges.append(np.array([cell_faces[0, fi], fi, cell_faces[1, fi]]))
                edge_node_type.append(("cell", "face", "cell"))

            if g.dim == 2:
                if boundary_face:
                    tmp_surfaces.append(np.array([ci, fi]))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)

                    # Add another surface from this face to the central node
                    tmp_surfaces.append(np.array([fi, ni]))
                    surface_node_type.append(("face", "node"))
                    surface_is_boundary.append(True)

                    # No need to append constraints for the boundary, this will be
                    # represented in the local grid anyhow

                else:
                    tmp_surfaces.append((cell_faces[0, fi], fi))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)

                    tmp_surfaces.append((cell_faces[1, fi], fi))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)

                    # This half-face should form a constraint for the meshing
                    constraints.append((fi, ni))
                    constraints_node_type.append(("face", "node"))

            else:  # g.dim == 3
                edge_ind_this_face = other_node[face_of_edge == fi]

                if boundary_face:
                    ci = np.array([ci])
                else:
                    ci = cell_faces[:, fi]

                for ei in edge_ind_this_face:
                    for c in ci:
                        tmp_surfaces.append((c, fi, ei))
                        surface_node_type.append(("cell", "face", "edge"))
                        surface_is_boundary.append(False)

                    if boundary_face:
                        # This face also is part of the local boundary
                        tmp_surfaces.append((fi, ei, ni))
                        surface_node_type.append(("face", "edge", "node"))
                        surface_is_boundary.append(True)

                    else:
                        # If this is not a boundary, the face must still be explicitly
                        # resolved in the mesh
                        constraints.append((fi, ei, ni))
                        constraints_node_type.append(("face", "edge", "node"))

            # End of loop over faces

        # Bounding edges of the interaction region
        edges = np.array(tmp_edges)
        surfaces = np.array(tmp_surfaces)

        reg = InteractionRegion(g, "mpfa", reg_ind=ni, central_node=ni)

        reg.surface_node_type = ("cell", "face")
        reg.edges = edges
        reg.surfaces = surfaces

        reg.edge_node_type = edge_node_type
        reg.surface_is_boundary = surface_is_boundary
        reg.surface_node_type = surface_node_type

        reg.constraints = constraints
        reg.constraint_node_type = constraints_node_type

        region_list.append(reg)

    return region_list


def _find_edges(g, loc_faces, central_node):

    fn_loc = g.face_nodes[:, loc_faces]
    node_ind = fn_loc.indices
    fn_ptr = fn_loc.indptr

    if not np.unique(np.diff(fn_ptr)).size == 1:
        ValueError("Have not implemented grids with varying number of face-nodes")

    num_fn = np.unique(np.diff(fn_ptr))[0]

    sort_ind = np.argsort(node_ind)
    sorted_node_ind = node_ind[sort_ind]

    face_ind = np.tile(loc_faces, (num_fn, 1)).ravel(order="f")
    sorted_face_ind = face_ind[sort_ind]

    not_central_node = np.where(sorted_node_ind != central_node)[0]

    sorted_node_ind_ext = sorted_node_ind[not_central_node]
    sorted_face_ind_ext = sorted_face_ind[not_central_node]

    multiple_occur = np.where(np.bincount(sorted_node_ind_ext) > 1)[0]

    hit = np.in1d(sorted_node_ind_ext, multiple_occur)

    nodes_on_edges = sorted_node_ind_ext[hit]
    face_of_edges = sorted_face_ind_ext[hit]

    return nodes_on_edges, face_of_edges
