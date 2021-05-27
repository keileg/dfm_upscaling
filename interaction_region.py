"""
Created on Mon Mar  2 14:32:39 2020

@author: eke001
"""
import os
from pathlib import Path
import numpy as np
import porepy as pp
from porepy.fracs.gmsh_interface import Tags, GmshWriter
from typing import Union, Tuple, List, Optional


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

        self.network = None
        #        if self.dim == 2:
        #            self.fracture_pts = np.zeros((3, 0))
        #            self.fracture_edges = np.zeros((2, 0), dtype=np.int)
        #        else:
        #            self.fractures: List[pp.Fractures] = []

        if name == "mpfa":
            self.node_coord = g.nodes[:, reg_ind].reshape((-1, 1))

        self.file_name = ".gmsh_upscaling_region_" + str(self.reg_ind)

        # True if central node is tip (understood, this is mpfa)
        self.is_tip: bool = False

        # Index of surfaces that are macroscale fractures, see self._mesh_3d() for details.
        # Only needed in 3d.
        self.ind_surf_on_macro_frac: np.ndarray = np.array([])

    def mesh(
        self, mesh_args=None
    ) -> Tuple[pp.GridBucket, Union[pp.FractureNetwork2d, pp.FractureNetwork3d], str]:
        """Create a local mesh for this interaction region."""
        if mesh_args is None:

            # Find the minimum extent of the domain along coordinate axis, use this to
            # guide mesh size arguments
            extent = np.min(np.diff(self.bounding_box(), axis=1)[: self.dim])

            # These are random values, use with care.
            mesh_args = {
                "mesh_size_frac": extent / 3,
                "mesh_size_bound": extent / 3,
                "mesh_size_min": extent / 5,
            }
        if self.dim == 2:
            return self._mesh_2d(mesh_args)
        else:
            return self._mesh_3d(mesh_args)

    def _mesh_2d(self, mesh_args) -> Tuple[pp.GridBucket, pp.FractureNetwork2d]:
        """To create a local grid bucket in 2d, we should:
        1) Create the bounding surfaces, from self.surfaces
            i) Find coordinates of all surfaces
            ii) Remove duplicate nodes
            iii) Create a FractureNetwork2d object, create a mesh

        """
        # First, build points and edges for the domain boundary
        domain_pts = np.zeros((3, 0))
        domain_edges = np.zeros((2, 0), dtype=int)
        edge_2_surf = np.empty([], dtype=int)

        # Empty value for frac_edge - this will be filled in if relevant.
        frac_edge = np.zeros((3, 0))

        # Tag the region surfaces as (macro) fracture or not
        surface_is_fracture: List[bool] = []

        for surf_ind, (surf, node_type) in enumerate(
            zip(self.surfaces, self.surface_node_type)
        ):

            # Fracture tips need some special treatment in mpfa regions:
            # First, the region boundary will contain surfaces from the face centers
            # on the two fracture faces, to the tip node. These edges must be removed,
            # or else the domain for local meshing will contain a slit.
            # Second, the line of the macro fracture must still be represented in the
            # local grid, and the 2d domain must have their faces split along the
            # macro fracture. This is achieved by first adding a micro fracture along
            # what is the macro fractures, construct the full GridBucket, and then
            # remove it towards the end.
            # For Tpfa, no such adjustments are needed.
            if self.name == "mpfa":
                # We will only run into this for surfaces with two components
                if len(node_type) == 2:
                    # Checks
                    assert "face" in node_type
                    fi_edge = node_type.index("face")
                    if (
                        self.g.tags["fracture_faces"][surf[fi_edge]]
                        and "node" in node_type
                    ):
                        ni = node_type.index("node")

                        # Consistency check, for mpfa only the central node should be involved
                        # in the interaction region boundary
                        assert surf[ni] == self.reg_ind

                        # If this node is tagged as a tip in the macro grid, fetch the coordinates
                        # the line from face center to node
                        if self.g.tags["node_is_fracture_tip"][self.reg_ind]:
                            # Not sure if this will work for polyline fractures, where
                            # the tip is not really a tip
                            if self.is_tip:
                                # If we have already encountered the tip node,
                                # Check that the edge is the same as previously found
                                # If frac_edge has its initial empty value, something
                                # is really wrong.
                                assert np.allclose(
                                    frac_edge, self.coords(surf, node_type)
                                )
                            self.is_tip = True
                            # Store the coordinate of the macro fracture.
                            frac_edge = self.coords(surf, node_type)
                            # Store this as fracture surface
                            surface_is_fracture.append(True)

                            # No need to do more for this surface
                            continue
                        else:
                            # This is not a fracture tip, but internal to fracture.
                            # Standard treatment should suffice
                            pass

            e = np.vstack(
                (np.arange(len(node_type) - 1), 1 + np.arange(len(node_type) - 1))
            )
            # The new edges are offset by the number of previous points
            domain_edges = np.hstack((domain_edges, domain_pts.shape[1] + e))

            # Then add new points
            domain_pts = np.hstack((domain_pts, self.coords(surf, node_type)))

            # This is not a fracture surface
            surface_is_fracture.append(False)

            edge_2_surf = np.hstack((edge_2_surf, surf_ind + np.ones(e.shape[1])))

        # Next, build up the constraints
        # Todo: Expand this with fractures contained within the region
        edge_2_constraint = np.array([], dtype=int)
        constraint_edges = np.empty((2, 0), dtype=int)
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
            constraint_pts = np.hstack(
                (constraint_pts, self.coords(constraint, node_type))
            )

            edge_2_constraint = np.hstack(
                (edge_2_constraint, constraint_ind * np.ones(e.shape[1], dtype=int))
            )

        # Uniquify points on the domain boundary
        unique_domain_pts, _, all_2_unique = pp.utils.setmembership.unique_columns_tol(
            domain_pts, tol=self.network.tol
        )
        unique_domain_edges = all_2_unique[domain_edges]
        # Also sort the boundary points to form a circle
        sorted_edges, sort_ind = pp.utils.sort_points.sort_point_pairs(
            unique_domain_edges
        )
        # Store the mapping from the ordering of the domain boundaries, as represented
        # in the FractureNetwork, to the ordering of surfaces in this region.
        # The mapping must be adjusted to ignore surfaces that are macro fractures
        incr_sort_ind = sort_ind
        for i in np.where(surface_is_fracture)[0]:
            hit = incr_sort_ind >= i
            incr_sort_ind[hit] += 1

        self.domain_edges_2_reg_surface = incr_sort_ind

        # Store boolean of which surfaces are macro fractures.
        self.surface_is_macro_fracture: List[bool] = surface_is_fracture

        constraint_edges += self.network.pts.shape[1]

        int_pts = np.hstack((self.network.pts, constraint_pts))
        int_edges = np.hstack((self.network.edges, constraint_edges))
        int_tags = self.network.tags
        for key, value in int_tags.items():
            int_tags[key] = np.hstack((value, [None] * constraint_edges.shape[1]))

        edge_2_constraint += self.network.edges.shape[1]

        # If this is a tip of a macro fracture, add the part of the macro fracture within
        # this network *at the begining of the fracture list* (will be important later)
        if self.is_tip:
            int_pts = np.hstack((frac_edge, int_pts))
            int_edges = np.hstack((np.array([[0], [1]]), 2 + int_edges))
            edge_2_constraint += 1
            # Also expand the tags with a default value for the macro face-cum-fracture
            for key, value in int_tags.items():
                int_tags[key] = np.hstack(([None], value))

        # Similarly uniquify points in constraint description
        unique_int_pts, _, a2u = pp.utils.setmembership.unique_columns_tol(
            int_pts, tol=self.network.tol
        )
        unique_int_edges = a2u[int_edges]

        # Define a fracture network, using the surface specification as boundary,
        # and the constraints as points
        # Fractures will be added as edges
        network_for_meshing = pp.FractureNetwork2d(
            domain=unique_domain_pts[: self.dim, sorted_edges[0]],
            pts=unique_int_pts[: self.dim],
            edges=unique_int_edges,
            tol=self.network.tol,
        )
        network_for_meshing.tags = int_tags
        gmsh_data = network_for_meshing.prepare_for_gmsh(
            mesh_args=mesh_args,
            constraints=edge_2_constraint,
            remove_small_fractures=True,
        )

        decomp = network_for_meshing._decomposition

        physical_points = {}
        for p in decomp["domain_boundary_points"]:
            physical_points[p] = Tags.DOMAIN_BOUNDARY_POINT.value

        #        gmsh_data.physical_points.update(physical_points)

        #        breakpoint()
        gmsh_writer = GmshWriter(gmsh_data)

        # Generate the mesh. Also write .geo file to ease debugging;
        # we will delete this later on.
        gmsh_writer.generate(self.file_name, write_geo=True)

        grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(
            self.file_name, constraints=edge_2_constraint
        )

        # preserve tags for the fractures from the network
        # we are assuming a coherent numeration between the network
        # and the created grids
        frac = np.setdiff1d(
            np.arange(network_for_meshing.edges.shape[1]),
            edge_2_constraint,
            assume_unique=True,
        )
        for idg, g in enumerate(grid_list[1]):
            for key in int_tags:
                if key not in g.tags:
                    g.tags[key] = int_tags[key][frac][idg]

        gb = pp.meshing.grid_list_to_grid_bucket(grid_list)

        #        gb = network.mesh(
        #            mesh_args=mesh_args,
        #            file_name=self.file_name,
        #            constraints=edge_2_constraint,
        #            preserve_fracture_tags=[k for k in int_tags.keys()],
        #        )

        if self.is_tip:
            # For 2d tip nodes, we added a macro fracture to the local fracture
            # network, so that the local 2d grid was split along the macro surface.
            # This fracture should not be part of the local problem, and must
            # be removed.

            # First identify the grid to remove. It should be at the start of the
            # list of fractures.
            gf = gb.grids_of_dimension(self.dim - 1)[0]

            # Check that the grid we have picked out indeed corresponds to the
            # macro scale fracture. This can be removed when we trust the implementation.
            nodes = gf.nodes
            for i in range(2):
                p = frac_edge[:, i].reshape((-1, 1))
                dist = np.min(np.sum(np.power(p - nodes, 2), axis=0))
                assert dist < 1e-8

                # Next remove all lower-dimensional neighbors that are formed by the
                # intersection with other (micro) fractures. Note that this will
                # also remove the interfaces (mortar grids) between intersection and
                # micro fractures. Furthermore, since the micro fractures have been
                # split to accomodate the micro fracture, this will remove the
                # connection of the micro fracture across the micro fracture.
                for low_neigh in gb.node_neighbors(gf, only_lower=True):
                    gb.remove_node(low_neigh)

            # Finally remove the node
            gb.remove_node(gf)
        return gb, network_for_meshing

    def _mesh_3d(self, mesh_args) -> Tuple[pp.GridBucket, pp.FractureNetwork2d]:

        # List of surfaces that make up the region boundary
        boundaries: List[np.ndarray] = []
        # List of region surfaces that form part of a macroscale fracture face.
        macro_frac_surfaces: List[np.ndarray] = []

        ind_surf_on_macro_frac = []

        # Macroscale fracture faces should not be included in the boundary of the
        # interaction region. However, the fractures must be represented in the local
        # grid bucket (really, the faces of the 3d grid along the fracture face
        # must be split). To that end, we first identify which surfaces in the
        # macro region corresponds to macro faces
        if self.name == "mpfa":
            self.is_tip = self.g.tags["node_is_tip_of_some_fracture"][self.reg_ind]
            macro_faces = self.macro_face_ind()

            # This is the list of face indices, which, when encountered in the loop
            # of region surfaces below, will trigger addition to macro_frac_surfaces.
            # Note that the other (split) version of this macro face is simply ignored
            # in the below loop, or else we would have added the fracture twice.
            here_faces = []
            # Also keep track of the faces on both sides of macro fractures, as
            # these will trigger special treatment.
            macro_face_in_region = []

            for fi in macro_faces:
                hit = np.where(self.g.frac_pairs[0] == fi)[0]
                if hit.size > 0 and self.g.frac_pairs[1, hit][0] in macro_faces:
                    here_faces.append(fi)
                    macro_face_in_region.append(fi)
                    macro_face_in_region.append(self.g.frac_pairs[1, hit][0])
        else:
            # This is a tpfa region, where the notion of tip nodes make no sense
            pass

        # Tag the region surfaces as (macro) fracture or not
        surface_is_fracture: List[bool] = []

        for si, (surf, node_type) in enumerate(
            zip(self.surfaces, self.surface_node_type)
        ):
            if self.name == "mpfa":
                if self.is_tip and "node" in node_type:
                    assert "face" in node_type
                    face = node_type.index("face")
                    global_face_ind = surf[face]

                    # If this is a face on a macro face, we may add it (if it is here),
                    # but it will not be appended to the region boundary
                    if global_face_ind in macro_face_in_region:
                        # Register the macro face, but only on the here side, to avoid duplicates
                        if global_face_ind in here_faces:
                            macro_frac_surfaces.append(
                                pp.Fracture(self.coords(surf, node_type))
                            )
                        # Take note that this surface was on a macro fracture face, and was not
                        # added to the boundary of the interaction region.
                        ind_surf_on_macro_frac.append(si)
                        surface_is_fracture.append(True)
                        # Do not add surface to region boundary
                        continue

            # We will make it to here, unless this is a mpfa region on a tip node,
            # and the surfaces is based on what is a macroscale fracture.
            boundaries.append(self.coords(surf, node_type))
            surface_is_fracture.append(False)

        # Store indices of region surfaces not added to the boundary.
        self.ind_surf_on_macro_frac = np.array(ind_surf_on_macro_frac)

        # Store boolean of which surfaces are macro fractures.
        self.surface_is_macro_fracture: List[bool] = surface_is_fracture

        # The ordering of the network boundary surfaces is the same as for the IAreg
        self.domain_edges_2_reg_surface = np.arange(len(boundaries))

        constraints: List[np.ndarray] = []

        for constraint, node_type in zip(self.constraints, self.constraint_node_type):
            constraints.append(pp.Fracture(self.coords(constraint, node_type)))

        # The polygons to be used for meshing are the macro and micro surfaces,
        # and the constraints.
        polygons: List[np.ndarray] = macro_frac_surfaces + self.fractures + constraints

        # Update number of macro faces introduced.
        self.num_macro_frac_faces = len(macro_frac_surfaces)

        constraint_inds = (
            self.num_macro_frac_faces
            + len(self.fractures)
            + np.arange(len(constraints))
        )
        network = pp.FractureNetwork3d(polygons)
        # Impose the boundary on the fracture network.
        # The area threshold is assigned to avoid very small fractures.
        ind_map = network.impose_external_boundary(boundaries, area_threshold=1e-2)
        updated_constraint_inds = []

        for ci in constraint_inds:
            ui = np.where(ind_map == ci)[0]
            assert ui.size == 1
            updated_constraint_inds.append(ui[0])

        # Generate local mesh for the interaction region
        gb = network.mesh(
            mesh_args=mesh_args,
            file_name=self.file_name,
            constraints=updated_constraint_inds,
            write_geo=True,
        )

        # In the construction of the grid bucket, the 3d grid had its faces split along
        # all fractures, hereunder any artifical fracture that was inserted to compensate
        # for a macroscale fracture face. The final step is to delete these grids, and
        # all connection between such grids and other (will be microscale) fractures.
        # In the process, the connectivity within a microscale fracture which crosses a
        # macro fracture is broken, however, this will be compensated in the macroscale
        # fracture-matrix interaction.
        if self.is_tip:
            grids_to_remove = gb.grids_of_dimension(2)[: self.num_macro_frac_faces]
            for rem in grids_to_remove:
                # Pick out intersection lines in the plane of the fake fracture
                neigh_1d = gb.node_neighbors(rem, only_lower=True)
                for neigh in neigh_1d:
                    # Pick out 0d point grids
                    neigh_0d = gb.node_neighbors(neigh, only_lower=True)
                    for point in neigh_0d:
                        # First remove the 0d node. This also deletes all edges between
                        # this node and intersection lines
                        gb.remove_node(point)
                    # All connections to 0d points are deleted, we can delete the line
                    gb.remove_node(neigh)
                # .. and delete the 2d grid.
                gb.remove_node(rem)

        # For 2d grids, make frac_num point to the fracture index in the global list
        # of fractures.
        if len(gb.grids_of_dimension(2)) > 0:
            # Counter for the list of 2d grids in gb
            counter = 0
            for ind in range(ind_map.size):
                if ind < self.num_macro_frac_faces or ind in updated_constraint_inds:
                    # Constraints do not have grids assigned.
                    continue

                # Fracture index, adjust for macro fratures.
                frac_ind = ind_map[ind] - self.num_macro_frac_faces
                gb.grids_of_dimension(2)[counter].frac_num = frac_ind
                counter += 1

        return gb, network

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

    def add_network(self, network) -> None:

        if self.dim == 3:
            # NEED TO BE FIXED
            fracs = []
            for f in network._fractures:
                fracs.append(f.copy())
            self.fractures = fracs
        else:
            self.network = network.copy()
            if self.network.pts.shape[0] == 2:
                self.network.pts = np.vstack(
                    (self.network.pts, np.zeros(self.network.pts.shape[1]))
                )
            # self.fracture_pts = points
            # self.fracture_edges = edges

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

    def bounding_box(self):
        """
        Get the bounding box for an interaction region.

        The box is found by taking max and min coordinates over all regions vertexes
        as defined by grid entities in the parent coarse grid.

        Returns:
            np.ndarray, size self.dim x 2: First column is minimum coordinates, second
                is max coordinates

        """
        min_coord = np.ones((3, 1)) * np.inf
        max_coord = -np.ones((3, 1)) * np.inf

        for edge, node_type in zip(self.edges, self.edge_node_type):
            for e, node in zip(edge, node_type):
                min_coord = np.minimum(min_coord, self.coords(e, node))
                max_coord = np.maximum(max_coord, self.coords(e, node))

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

    def _find_coarse_inds(self, nt: str) -> List[int]:
        ci: List[int] = []
        for surf, node_type in zip(self.surfaces, self.surface_node_type):
            for i, node in enumerate(node_type):
                if node == nt:
                    ci.append(surf[i])

        # Uniquify
        return list(set(ci))

    def macro_cell_inds(self) -> List[int]:
        """Find the index of macro cells included in the interaction region.

        Returns:
            list of int: Index of the macro cells that are in this region.

        """
        return self._find_coarse_inds("cell")

    def macro_face_ind(self) -> List[int]:
        """Find the index of macro faces included in the interaction region.

        Returns:
            list of int: Index of the macro faces that are in this region.

        """

        return self._find_coarse_inds("face")

    def cleanup(self) -> None:
        """Delete files used for local mesh generation for this region."""
        msh = Path(self.file_name + ".msh")
        geo = Path(self.file_name + ".geo_unrolled")
        for file in [msh, geo]:
            if Path.exists(file):
                Path.unlink(file)

    def __str__(self) -> str:
        s = f"Interaction region of type {self.name} for a {self.dim}-dimensional grid.\n"
        if self.name == "tpfa":
            s += f"Central face: {self.reg_ind}\n"
        elif self.name == "mpfa":
            s += f"Central node: {self.reg_ind}\n"

        ci = []
        fi = []
        for (surf, node_type) in zip(self.surfaces, self.surface_node_type):
            for si, nt in zip(surf, node_type):
                if nt == "cell":
                    ci.append(si)
                elif nt == "face":
                    fi.append(si)

        s += f"Region involves {len(set(ci))} macro cells and {len(set(fi))} macro faces.\n"

        s += f"Region has:\n"
        s += f"{self.edges.shape[0]} 1d edges\n"
        s += f"{self.surfaces.shape[0]} {self.dim - 1}d surfaces\n"
        s += f"{sum(self.surface_is_boundary)} surfaces are on the macro domain boundary\n"

        s += f"Region has {len(self.constraints)} internal constraints at macro faces\n"

        bb = self.bounding_box()
        s += f"Min and max coordinates:\n"
        s += f"x: ({bb[0, 0]}, {bb[0, 1]})\n"
        s += f"y: ({bb[1, 0]}, {bb[1, 1]})\n"
        if self.dim == 3:
            s += f"x: ({bb[2, 0]}, {bb[2, 1]})\n"

        return s

    def __repr__(self) -> str:
        return self.__str__()


def extract_tpfa_regions(
    g: pp.Grid, faces: np.ndarray = None
) -> List[InteractionRegion]:
    """Factory method to define tpfa-type interaction regions for specified faces in a
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
    """Factory method to define mpfa-type interaction regions for specified faces in a
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
    # edge going from the central to that other node.
    # Exception: Split faces on the rim of a fracture may give multiple occurences of a node
    # even if the node is not part of the interaction region. This will happen for a
    # Cartesian 3d grid with a fracture of a single cell, the node diagonally from the
    # central node is shared by the split faces, and will be picked up. Correct for
    # this below (and yes, that case was not easy to debug).

    # Count the number of occurences of nodes
    node_counts = np.bincount(sorted_node_ind_ext)
    # Count the number of occurences, considering only nodes associated with fracture faces
    node_counts_fracture_faces = np.bincount(
        sorted_node_ind_ext[g.tags["fracture_faces"][sorted_face_ind_ext]],
        minlength=node_counts.size,
    )
    # There is no need to do special treatment of nodes that are involved in
    # faces not on fractures.
    node_only_on_fracture_faces = np.where(
        np.logical_and(node_counts > 0, node_counts == node_counts_fracture_faces)
    )[0]

    # Loop over all nodes only present on fracture faces. Check if all faces of
    # the node belong to the same split fracture.
    for ni in node_only_on_fracture_faces:
        hit = sorted_node_ind_ext == ni
        # Faces on the node.
        face_hit = sorted_face_ind_ext[hit]

        # For all the fracture faces, find which pair it belongs to.
        frac_face_pair = []
        for f in face_hit:
            frac_face_pair.append(np.where(np.any(g.frac_pairs == f, axis=0))[0][0])

        # If all faces of this node belongs to the same node, it should be excluded.
        if len(set(frac_face_pair)) == 1:
            node_counts[ni] = 0

    # Nodes that occur more than one time should be included among the edges in the region.
    multiple_occur = np.where(node_counts > 1)[0]
    hit = np.in1d(sorted_node_ind_ext, multiple_occur)

    # Edges (represented by the node that is not the central one), and the faces of the
    # edges. Note that neither nodes_on_edges nor face_of_edges are unique, however, the
    # combination of an edge and a face should be so.
    nodes_on_edges = sorted_node_ind_ext[hit]
    face_of_edges = sorted_face_ind_ext[hit]

    return nodes_on_edges, face_of_edges
