#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:32:39 2020

@author: eke001
"""
import numpy as np
import porepy as pp
from typing import Union, Tuple, List, Optional

import pdb


class InteractionRegion:
    def __init__(
        self,
        g: pp.Grid,
        name: str,
        reg_ind: int,
        surfaces: np.ndarray,
        surface_node_type: List[Tuple[str]],
        surface_is_boundary: List[bool],
        edges: np.ndarray,
        edge_node_type: List[Tuple[str]],
        constraints: List[Tuple[int]],
        constraint_node_type: List[Tuple[str]],
        macro_face_of_boundary_surface,
    ) -> None:

        self.g = g
        self.dim = g.dim
        self.name = name

        self.reg_ind = reg_ind

        self.surfaces = surfaces
        self.surface_node_type = surface_node_type
        self.surface_is_boundary = surface_is_boundary
        self.edges = edges
        self.edge_node_type = edge_node_type
        self.constraints = constraints
        self.constraint_node_type = constraint_node_type

        self.macro_face_of_boundary_surface = macro_face_of_boundary_surface

        if self.dim == 2:
            self.fracture_pts = np.zeros((3, 0))
            self.fracture_edges = np.zeros((2, 0), dtype=np.int)
        else:
            self.fractures: List[pp.Fractures] = []

        if name == "mpfa":
            self.node_coord = g.nodes[:, reg_ind].reshape((-1, 1))

    ####################
    ## Functions related to meshing of fracture networks
    ####################

    def mesh(
        self,
    ) -> Tuple[pp.GridBucket, Union[pp.FractureNetwork2d, pp.FractureNetwork3d], str]:
        """ Create a local mesh for this interaction region.
        """
        if self.dim == 2:
            return self._mesh_2d()
        else:
            return self._mesh_3d()

    def _mesh_2d(self) -> Tuple[pp.GridBucket, pp.FractureNetwork2d, str]:
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
            domain_pts = np.hstack((domain_pts, self.coords(surf, node_type)))

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
            constraint_pts = self.coords(constraint, node_type)

            edge_2_constraint = np.hstack(
                (edge_2_constraint, constraint_ind * np.ones(e.shape[1], dtype=np.int))
            )

        # Uniquify points on the domain boundary
        unique_domain_pts, _, all_2_unique = pp.utils.setmembership.unique_columns_tol(
            domain_pts
        )
        unique_domain_edges = all_2_unique[domain_edges]
        # Also sort the boundary points to form a circle
        sorted_edges, sort_ind = pp.utils.sort_points.sort_point_pairs(
            unique_domain_edges
        )

        constraint_edges += self.fracture_pts.shape[1]

        int_pts = np.hstack((self.fracture_pts, constraint_pts))
        int_edges = np.hstack((self.fracture_edges, constraint_edges))

        edge_2_constraint += self.fracture_edges.shape[1]

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

    def _mesh_3d(self) -> Tuple[pp.GridBucket, pp.FractureNetwork2d, str]:

        boundaries: List[np.ndarray] = []
        for surf, node_type in zip(self.surfaces, self.surface_node_type):
            boundaries.append(self.coords(surf, node_type))

        constraints: List[np.ndarray] = []

        for constraint, node_type in zip(self.constraints, self.constraint_node_type):
            constraints.append(pp.Fracture(self.coords(constraint, node_type)))

        polygons: List[np.ndarray] = self.fractures + constraints

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
            mesh_args=mesh_args, file_name=file_name, constraints=constraint_inds
        )

        return gb, network, file_name

    def coords(self, indices, node_type) -> np.ndarray:
        indices = np.atleast_1d(np.asarray(indices))
        if isinstance(node_type, str):
            node_type = [node_type]

        pts = np.zeros((3, 0))
        for ind, node in zip(indices, node_type):
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
            pts = np.hstack((pts, p))
        return pts

    def add_fractures(
        self,
        points: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        fractures: Optional[List] = None,
    ) -> None:
        if self.dim == 3:
            self.fractures = fractures
        else:
            if points.shape[0] == 2:
                points = np.vstack((points, np.zeros(points.shape[1])))
            self.fracture_pts = points
            self.fracture_edges = edges

    def coarse_cell_centers(self):

        ci = []
        coords = np.zeros((3, 0))
        for edge, node_type in zip(self.edges, self.edge_node_type):
            for e, node in zip(edge, node_type):
                if node == "cell":
                    ci.append(e)
                    coords = np.hstack((coords, self.coords(e, node)))

        ci = np.array(ci, dtype=np.int)

        unique_ci, ind = np.unique(ci, return_index=True)

        return unique_ci, coords[:, ind]

    ###### Utility functions

    def bounding_box(self):
        """
        Get the bounding box for an interaction region.

        The box is found by taking max and min coordinates over all regions vertexes
        as defined by grid entities in the parent coarse grid.

        Returns:
            np.ndarray, size self.dim x 2: First column is minimum coordinates, second
                is max coordinates

        """
        min_coord = np.ones((self.dim, 1)) * np.inf
        max_coord = -np.ones((self.dim, 1)) * np.inf

        for edge, node_type in zip(self.edges, self.edge_node_type):
            for e, node in zip(edge, node_type):
                min_coord = np.minimum(min_coord, self.coord(node, e))
                max_coord = np.maximum(max_coord, self.coord(node, e))

        return np.hstack((min_coord, max_coord))

    def macro_boundary_faces(self) -> List[int]:
        """
        Find the macro index of faces on the boundary of the macro domain.

        Returns:
            list of int: Index of the macro faces that are in this region, and is a
                domain boundary.

        """
        bf: List[int] = []
        for surf, node_type, is_bound in zip(
            self.surfaces, self.surface_node_type, self.surface_is_boundary
        ):
            # If this is not a boundary surface, we can continue
            if not is_bound:
                continue

            # If tpfa, there will be a single boundary face, identified by the region
            # index
            if self.name == "tpfa":
                bf.append(self.reg_ind)
            else:  # mpfa
                bf.append(surf[node_type.index("face")])

        return bf

    def __str__(self) -> str:
        s = f"Interaction region of type {self.name}\n"
        if self.name == "tpfa":
            s += f"Central face: {self.reg_ind}\n"
        elif self.name == "mpfa":
            s += f"Central node: {self.reg_ind}\n"

        s += f"Region has:\n"
        s += f"{self.edges.shape[0]} 1d edges\n"
        s += f"{self.surfaces.shape[0]} {self.dim - 1} surfaces\n"
        s += (
            f"{sum(self.surface_is_boundary)} surfaces are on the macro domain boundary"
        )

        return s

    def __repr__(self) -> str:
        return self.__str__()


def extract_tpfa_regions(
    g: pp.Grid, faces: np.ndarray = None
) -> List[InteractionRegion]:
    """ Factory method to define tpfa-type interaction regions for specified faces in a
    grid.

    Parameters:
        g (pp.Grid): Grid where interaction regions will be found
        faces (int or np.array, optional): Index of faces for which the regions will be
            found. If not provided, all faces in the gridwill have their region
            computed.

    Returns:
        List of InteractionRegion: One per face in faces.

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

    # For region surfaces that are on the boundary, we store the face index of the
    # corresponding face in the macro grid.
    macro_face_of_surface = []

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
                macro_face_of_surface.append([])

                tri.append((ci, nodes[1]))
                surface_node_type.append(("cell", "node"))
                surface_is_boundary.append(False)
                macro_face_of_surface.append([])

            else:
                for ni in range(nodes.size):
                    # The face of the interaction region consists of one cell
                    # and two nodes, in that order.
                    tri.append((ci, nodes_extended[ni], nodes_extended[ni + 1]))
                    surface_node_type.append(("cell", "node", "node"))
                    surface_is_boundary.append(False)
                    macro_face_of_surface.append([])

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
                    macro_face_of_surface.append(fi)
                elif nodes.size == 4:
                    tri.append(nodes[:3])
                    surface_node_type.append(("node", "node", "node"))
                    surface_is_boundary.append(True)
                    macro_face_of_surface.append(fi)

                    tri.append((nodes[0], nodes[2], nodes[3]))
                    surface_node_type.append(("node", "node", "node"))
                    surface_is_boundary.append(True)
                    macro_face_of_surface.append(fi)

                else:
                    raise ValueError(
                        "Implementation only covers simplexes and Cartisan coarse grids"
                    )

        # Combine the full set of boundary faces of this interaction region
        # into an np.array, and store it in the global list
        surfaces = np.vstack([t for t in tri])

        edges = np.vstack([c for c in c2c])

        if on_boundary:
            constraints = []
            constraint_node_type = []
        else:
            constraints = [nodes]
            constraint_node_type = [nodes.size * ["node"]]

        reg = InteractionRegion(
            g,
            "tpfa",
            fi,
            surfaces,
            surface_node_type,
            surface_is_boundary,
            edges,
            edge_node_type,
            constraints,
            constraint_node_type,
            macro_face_of_surface,
        )

        region_list.append(reg)

    # Probably return an interaction region, or a named tuple
    return region_list


def extract_mpfa_regions(
    g: pp.Grid, nodes: np.ndarray = None
) -> List[InteractionRegion]:
    """ Factory method to define mpfa-type interaction regions for specified faces in a
    grid.

    Parameters:
        g (pp.Grid): Grid where interaction regions will be found
        nodes (int or np.array, optional): Index of nodes for which the regions will be
            found. If not provided, all nodes in the grid will have their region
            computed.

    Returns:
        List of InteractionRegion: One per node in nodes.

    """

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

    fn = g.face_nodes.tocsr()

    for ni in nodes:

        # Faces of this node
        loc_faces = fn[ni].indices

        # Data structure to store local information
        tmp_edges = []  # 1d edges
        edge_node_type = []  # Node type for the edges
        tmp_surfaces = []
        surface_node_type = []
        surface_is_boundary = []
        constraints = []
        constraints_node_type = []

        # For region surfaces that are on the boundary, we store the face index of the
        # corresponding face in the macro grid.
        macro_face_of_surface = []

        if g.dim == 3:
            # In 3d, the boundary of the interaction region will be formed by triangles
            # composed of cell centers, face centers and points on 1d edges in the grid.
            # The latter group is not represented in the grid structure, so we recover
            # these, represented by the index of the other node (not ni) of the edge.
            # Also find all nodes sharing that edge
            other_node, face_of_edge = _find_edges(g, loc_faces, ni)

        # Loop over all local faces. For each of these, we will first construct the 1d
        # edge to the adjacent cell center (if face on boundary) or centers (if face is
        # interior). Second, construct the part of the region boundary surfaces that
        # involves the face. Third, we need to register the part of coarse faces that
        # are inside the regions; these will be used as constraints in the gridding of
        # the region.
        for fi in loc_faces:
            # First the 1d edge. Construction here is the same in 2d and 3d.
            if np.any(cell_faces[:, fi] < 0):
                # This is a boundary face
                boundary_face = True
                # Find index of the adjacent cell
                if cell_faces[0, fi] < 0:
                    ci = cell_faces[1, fi]
                else:
                    ci = cell_faces[0, fi]
                # 1d edge from cell to face center
                tmp_edges.append(np.array([ci, fi]))
                edge_node_type.append(("cell", "face"))
            else:
                # Not a boundary edge
                boundary_face = False
                # Edge from cell, via face to next cell
                tmp_edges.append(np.array([cell_faces[0, fi], fi, cell_faces[1, fi]]))
                edge_node_type.append(("cell", "face", "cell"))

            # Next, part of the boundary of the region. This is quite a bit different
            # from 2d to 3d.
            if g.dim == 2:
                if boundary_face:
                    # One part of the boundary is identical to the 1d edge
                    tmp_surfaces.append(np.array([ci, fi]))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)
                    macro_face_of_surface.append([])

                    # Add another surface from this face to the central node
                    tmp_surfaces.append(np.array([fi, ni]))
                    surface_node_type.append(("face", "node"))
                    # This will be a boundary surface
                    surface_is_boundary.append(True)
                    macro_face_of_surface.append(fi)

                    # No need to append constraints for the boundary, this will be
                    # represented in the local grid anyhow

                else:
                    # The region surface consists of the two parts of the 1d edge
                    tmp_surfaces.append((cell_faces[0, fi], fi))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)
                    macro_face_of_surface.append([])

                    tmp_surfaces.append((cell_faces[1, fi], fi))
                    surface_node_type.append(("cell", "face"))
                    surface_is_boundary.append(False)
                    macro_face_of_surface.append([])

                    # The part of this face that is within the region will be a
                    # constraint for subsequent meshing.
                    constraints.append((fi, ni))
                    constraints_node_type.append(("face", "node"))

            else:  # g.dim == 3
                # Adjacent cells
                if boundary_face:
                    ci = np.array([ci])
                else:
                    ci = cell_faces[:, fi]

                # Loop over all edges of this face
                for ei in other_node[face_of_edge == fi]:
                    for c in ci:
                        # The boundary surface of the region
                        tmp_surfaces.append((c, fi, ei))
                        surface_node_type.append(("cell", "face", "edge"))
                        surface_is_boundary.append(False)
                        macro_face_of_surface.append([])

                    if boundary_face:
                        # This face also is part of the local boundary
                        tmp_surfaces.append((fi, ei, ni))
                        surface_node_type.append(("face", "edge", "node"))
                        surface_is_boundary.append(True)
                        macro_face_of_surface.append(fi)

                        # No need to define a constraint related to this edge

                    else:
                        # If this is not a boundary, the face must still be explicitly
                        # resolved in the mesh
                        constraints.append((fi, ei, ni))
                        constraints_node_type.append(("face", "edge", "node"))

            # End of loop over faces

        # Create ia_reg based on the information.
        reg = InteractionRegion(
            g,
            "mpfa",
            ni,
            np.array(tmp_surfaces),
            surface_node_type,
            surface_is_boundary,
            np.array(tmp_edges, dtype=np.object),
            edge_node_type,
            constraints,
            constraints_node_type,
            macro_face_of_surface,
        )

        region_list.append(reg)

    return region_list


def _find_edges(
    g: pp.Grid, loc_faces: np.ndarray, central_node: int
) -> Union[np.ndarray, np.ndarray]:
    """
    Find the 1d edges around a central node in a 3d grid.

    Args:
        g (pp.Grid): Macro grid.
        loc_faces (np.ndarray): Index of faces that have central_node among their
            vertexes.
        central_node (int): Index of the central node.

    Returns:
        nodes_on_edges (np.ndarray): Index of nodes that form a 1d edge together with
            the central node.
        face_of_edges (np.ndarray): Faces corresponding to the edge.

    Raises:
        ValueError: If not all faces in the grid have the same number of nodes.

    """

    fn_loc = g.face_nodes[:, loc_faces]
    node_ind = fn_loc.indices
    fn_ptr = fn_loc.indptr

    if not np.unique(np.diff(fn_ptr)).size == 1:
        # Fixing this should not be too hard
        raise ValueError("Have not implemented grids with varying number of face-nodes")

    # Number of nodes per face
    num_fn = np.unique(np.diff(fn_ptr))[0]

    # Sort the nodes of the local faces.
    sort_ind = np.argsort(node_ind)
    # The elements in sorted_node_ind (and node_ind) will not be unique
    sorted_node_ind = node_ind[sort_ind]

    # Duplicate the face indices, and make the same sorting as for the nodes
    face_ind = np.tile(loc_faces, (num_fn, 1)).ravel(order="f")
    sorted_face_ind = face_ind[sort_ind]

    # Exclude nodes and faces that correspond to the central node
    not_central_node = np.where(sorted_node_ind != central_node)[0]
    sorted_node_ind_ext = sorted_node_ind[not_central_node]
    sorted_face_ind_ext = sorted_face_ind[not_central_node]

    # Nodes that occur more than once are part of at least two faces, thus there is an
    # edge going from the central to that other node
    # This may not be true for sufficiently degenerated grids (not sure what that means,
    # possibly something with hanging nodes).
    multiple_occur = np.where(np.bincount(sorted_node_ind_ext) > 1)[0]
    hit = np.in1d(sorted_node_ind_ext, multiple_occur)

    # Edges (represented by the node that is not the central one), and the faces of the
    # edges. Note that neither nodes_on_edges nor face_of_edges are unique, however, the
    # combination of an edge and a face should be so.
    nodes_on_edges = sorted_node_ind_ext[hit]
    face_of_edges = sorted_face_ind_ext[hit]

    return nodes_on_edges, face_of_edges
