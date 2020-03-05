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
    def __init__(self, g: pp.Grid, name: str):
        self.g = g
        self.dim = g.dim
        self.name = name
        pass


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

    region_list = []

    # Loop over all specified faces, find their interaction region
    for fi in faces:

        # Find cells neighboring this face
        cells = cell_faces[:, fi]
        # By construction of the dense face-cell relation, cell index of -1
        # indicates outside the outer boundary - skip these cells.
        cells = cells[cells >= 0]

        on_boundary = cells.size == 1

        # Nodes of the central face
        nodes = g.face_nodes[:, fi].indices
        # Extend so that we can combine pairs
        nodes_extended = np.hstack((nodes, nodes[0]))

        # Storage for this face.
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

        reg = InteractionRegion(g, 'tpfa')
        reg.surfaces = surfaces

        # Which type of grid element the boundaries of the interaction regions are
        # formed by. Needed for lookup of geometry (the first index refers to cell center etc.)
        reg.surface_node_type = surface_node_type
        reg.surface_is_boundary = surface_is_boundary

        reg.edges = edges
        reg.edge_node_type = edge_node_type

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

                else:
                    tmp_surfaces.append((cell_faces[0, fi], fi))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)

                    tmp_surfaces.append((cell_faces[1, fi], fi))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)

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
                        tmp_surfaces.append((fi, ei, ni))
                        surface_node_type.append(("face", "edge", "node"))
                        surface_is_boundary.append(True)

        # Bounding edges of the interaction region
        edges = np.array(tmp_edges)
        surfaces = np.array(tmp_surfaces)

        reg = InteractionRegion(g, 'mpfa')

        reg.surface_node_type = ("cell", "face")
        reg.edges = edges
        reg.surfaces = surfaces

        reg.edge_node_type = edge_node_type
        reg.surface_is_boundary = surface_is_boundary
        reg.surface_node_type = surface_node_type

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
