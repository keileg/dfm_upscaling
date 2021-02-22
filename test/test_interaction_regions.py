"""
"""

import numpy as np
import unittest
import porepy as pp

from dfm_upscaling.utils import create_grids
from dfm_upscaling import interaction_region as ia_reg
from dfm_upscaling.test import test_utils


class TestRegions(unittest.TestCase):
    """Tests for construction of interaction regions, without macroscale fractures"""

    def test_tpfa_cart_grid_2d_interior_face(self):

        g = create_grids.cart_2d()

        interior_face = 4
        cells = [2, 3]

        fn = g.face_nodes.indices.reshape((2, -1), order="f")
        fn_loc = fn[:, interior_face]

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0]],
                [cells[0], fn_loc[1]],
                [cells[1], fn_loc[0]],
                [cells[1], fn_loc[1]],
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node", "cell"))

    def test_tpfa_cart_grid_2d_boundary_face(self):

        g = create_grids.cart_2d()

        boundary_face = 3
        cells = [2]

        reg = ia_reg.extract_tpfa_regions(g, faces=[boundary_face])[0]

        fn = g.face_nodes.indices.reshape((2, -1), order="f")

        fn_loc = fn[:, boundary_face]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0]],
                [cells[0], fn_loc[1]],
                [fn_loc[0], fn_loc[1]],  # boundary surface
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node"))

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if np.all(np.isin(surf, fn_loc)):
                # This is a boundary surface
                self.assertTrue(bound_info == True)
                self.assertTrue(node_types == ("node", "node"))
            else:
                self.assertTrue(bound_info == False)
                self.assertTrue(node_types == ("cell", "node"))

    def test_tpfa_cart_grid_3d_interior_face(self):

        g = create_grids.cart_3d()

        interior_face = 4
        cells = [2, 3]

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        fn = g.face_nodes.indices.reshape((4, -1), order="f")

        fn_loc = fn[:, interior_face]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0], fn_loc[1]],
                [cells[0], fn_loc[1], fn_loc[2]],
                [cells[0], fn_loc[2], fn_loc[3]],
                [cells[0], fn_loc[3], fn_loc[0]],
                [cells[1], fn_loc[0], fn_loc[1]],
                [cells[1], fn_loc[1], fn_loc[2]],
                [cells[1], fn_loc[2], fn_loc[3]],
                [cells[1], fn_loc[3], fn_loc[0]],
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        known_edges = np.array(
            [
                [cells[0], fn_loc[0], cells[1]],
                [cells[0], fn_loc[1], cells[1]],
                [cells[0], fn_loc[2], cells[1]],
                [cells[0], fn_loc[3], cells[1]],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(known_edges.T, reg.edges.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node", "cell"))

    def test_tpfa_cart_grid_3d_boundary_face(self):

        g = create_grids.cart_3d()

        interior_face = 3
        cells = [2]

        reg = ia_reg.extract_tpfa_regions(g, faces=[interior_face])[0]

        fn = g.face_nodes.indices.reshape((4, -1), order="f")

        fn_loc = fn[:, interior_face]

        known_surfaces = np.array(
            [
                [cells[0], fn_loc[0], fn_loc[1]],
                [cells[0], fn_loc[1], fn_loc[2]],
                [cells[0], fn_loc[2], fn_loc[3]],
                [cells[0], fn_loc[3], fn_loc[0]],
                [fn_loc[0], fn_loc[1], fn_loc[2]],
                [fn_loc[0], fn_loc[2], fn_loc[3]],
            ]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))

        known_edges = np.array(
            [
                [cells[0], fn_loc[0]],
                [cells[0], fn_loc[1]],
                [cells[0], fn_loc[2]],
                [cells[0], fn_loc[3]],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(known_edges.T, reg.edges.T))

        for et in reg.edge_node_type:
            self.assertTrue(et == ("cell", "node"))

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if np.all(np.isin(surf, fn_loc)):
                # This is a boundary surface
                self.assertTrue(bound_info == True)
                self.assertTrue(node_types == ("node", "node", "node"))
            else:
                self.assertTrue(bound_info == False)
                self.assertTrue(node_types == ("cell", "node", "node"))

    def test_mpfa_cart_grid_2d_interior_node(self):
        g = create_grids.cart_2d()

        interior_node = 4
        # cells = [0, 1, 2, 3]
        # faces = [1, 4, 11, 12]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_edges = np.array([[0, 11, 2], [0, 1, 1], [1, 12, 3], [2, 4, 3]])

        known_surfaces = np.array(
            [[0, 1], [1, 1], [0, 11], [2, 11], [2, 4], [3, 4], [3, 12], [1, 12]]
        )

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))
        self.assertTrue(test_utils.compare_arrays(known_edges.T, reg.edges.T))

    def test_mpfa_cart_grid_2d_boundary_node(self):
        g = create_grids.cart_2d()

        interior_node = 3
        cells = [0, 2]
        faces = [0, 3, 11]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_edges = np.array([[0, 0], [0, 11, 2], [2, 3]])

        known_surfaces = np.array([[0, 0], [0, 11], [2, 11], [2, 3], [0, 3], [3, 3]])

        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))
        self.assertTrue(
            test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)
        )

        for i in range(reg.edges.size):
            if len(reg.edges[i]) == 2:
                self.assertTrue(reg.edge_node_type[i] == ("cell", "face"))
            if len(reg.edges[i]) == 3:
                self.assertTrue(reg.edge_node_type[i] == ("cell", "face", "cell"))

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if "node" in node_types:
                # This is a boundary surface
                self.assertTrue(bound_info == True)
                self.assertTrue(node_types == ("face", "node"))
                self.assertTrue(surf[-1] == interior_node)
                self.assertTrue(np.any(surf[0] == faces))
            else:
                self.assertTrue(bound_info == False)
                self.assertTrue(node_types == ("cell", "face"))

    def test_mpfa_cart_grid_3d_internal_node(self):

        g = create_grids.cart_3d()
        interior_node = 13
        # cells = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # faces = [1, 4, 7, 10, 14, 15, 20, 21, 28, 29, 30, 31]
        # other_nodes = [4, 10, 12, 14, 16, 22]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_surfaces = np.asarray(
            [
                [0, 1, 4],
                [0, 1, 10],
                [1, 1, 4],
                [1, 1, 10],
                [0, 14, 4],
                [0, 14, 12],
                [2, 14, 4],
                [2, 14, 12],
                [2, 4, 4],
                [2, 4, 16],
                [3, 4, 4],
                [3, 4, 16],
                [1, 15, 4],
                [1, 15, 14],
                [3, 15, 4],
                [3, 15, 14],  # Done with all lower faces
                [4, 7, 10],
                [4, 7, 22],
                [5, 7, 10],
                [5, 7, 22],
                [4, 20, 12],
                [4, 20, 22],
                [6, 20, 12],
                [6, 20, 22],
                [5, 21, 14],
                [5, 21, 22],
                [7, 21, 14],
                [7, 21, 22],
                [6, 10, 16],
                [6, 10, 22],
                [7, 10, 16],
                [7, 10, 22],  # Done with all upper faces
                [0, 28, 10],
                [0, 28, 12],
                [4, 28, 10],
                [4, 28, 12],
                [1, 29, 10],
                [1, 29, 14],
                [5, 29, 10],
                [5, 29, 14],
                [2, 30, 12],
                [2, 30, 16],
                [6, 30, 12],
                [6, 30, 16],
                [3, 31, 14],
                [3, 31, 16],
                [7, 31, 14],
                [7, 31, 16],
            ]
        )

        known_edges = np.array(
            [
                [0, 1, 1],
                [0, 14, 2],
                [2, 4, 3],
                [1, 15, 3],
                [4, 7, 5],
                [4, 20, 6],
                [5, 21, 7],
                [6, 10, 7],
                [0, 28, 4],
                [1, 29, 5],
                [2, 30, 6],
                [3, 31, 7],
            ]
        )

        self.assertTrue(
            test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T, sort=False)
        )
        self.assertTrue(
            test_utils.compare_arrays(known_edges.T, reg.edges.T, sort=False)
        )

    def test_mpfa_cart_grid_3d_boundary_node(self):
        g = create_grids.cart_3d()
        interior_node = 3
        # cells = [0, 2]
        # faces = [0, 3, 14, 24, 26]
        # other_nodes = [0, 4, 6, 12]

        reg = ia_reg.extract_mpfa_regions(g, interior_node)[0]

        known_edges = np.array([[0, 0], [0, 24], [0, 14, 2], [2, 3], [2, 26]])

        known_surfaces = np.array(
            [
                [0, 0, 0],
                [0, 24, 0],
                [0, 24, 4],
                [0, 14, 4],
                [2, 26, 4],
                [2, 14, 4],
                [0, 0, 12],
                [0, 14, 12],
                [2, 14, 12],
                [2, 3, 12],
                [2, 3, 6],
                [2, 26, 6],  # after this, boundary surfaces
                [0, 0, 3],
                [0, 12, 3],
                [3, 6, 3],
                [3, 12, 3],
                [24, 0, 3],
                [24, 4, 3],
                [26, 4, 3],
                [26, 6, 3],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T))
        self.assertTrue(
            test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)
        )

    def test_regions_different_grids(self):
        # Run through all prepared simple grids, check that the creation does not break
        for g in create_grids.create_grids():
            _ = ia_reg.extract_tpfa_regions(g)
            _ = ia_reg.extract_mpfa_regions(g)


# Below are tests for region constructions in the presence of internal boundaries.


def test_tpfa_2d_internal_boundary():
    gb = pp.meshing.cart_grid([np.array([[0, 1], [1, 1]])], np.array([1, 2]))
    g = gb.grids_of_dimension(2)[0]
    frac_faces = np.where(g.tags["fracture_faces"])[0]

    if g.cell_faces[frac_faces[0], 0] != 0:
        cells = [0, 1]
    else:
        cells = [1, 0]
    nodes = [
        g.face_nodes[:, frac_faces[0]].indices,
        g.face_nodes[:, frac_faces[1]].indices,
    ]

    for i, fi in enumerate(frac_faces):
        reg = ia_reg.extract_tpfa_regions(g, [fi])[0]

        known_surfaces = np.array(
            [
                [cells[i], nodes[i][0]],
                [cells[i], nodes[i][1]],
                [nodes[i][0], nodes[i][1]],
            ]
        )

        assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)

        known_edges = np.array([[cells[i], nodes[i][0]], [cells[i], nodes[i][1]]])

        assert test_utils.compare_arrays(known_edges.T, reg.edges.T)


def test_mpfa_2d_internal_boundary_not_tip():
    gb = pp.meshing.cart_grid([np.array([[1, 5], [1, 1]])], np.array([4, 2]))
    g = gb.grids_of_dimension(2)[0]

    # Nodes with coordinate (2, 1) are 8 and 9, we want the one connected to the
    # lower half of the domain
    fn_8 = g.face_nodes.tocsr()[8].indices
    if np.all(g.face_centers[1, fn_8] >= 1):
        node = 7
    else:
        node = 8

    if g.cell_faces[15, 1] != 0 and g.cell_faces[16, 0] != 0:
        frac_faces = [15, 16]
    elif g.cell_faces[15, 5] != 0 and g.cell_faces[16, 6] != 0:
        frac_faces = [22, 23]
    else:
        raise ValueError("Faces split inconsistent")

    cells = [1, 2]
    faces = [frac_faces[0], 2, frac_faces[1]]

    known_surfaces = np.array(
        [
            [node, faces[0]],
            [faces[0], cells[0]],
            [cells[0], faces[1]],
            [faces[1], cells[1]],
            [cells[1], faces[2]],
            [faces[2], node],
        ]
    )

    known_edges = np.array(
        [[faces[0], cells[0]], [cells[0], faces[1], cells[1]], [cells[1], faces[2]]]
    )

    reg = ia_reg.extract_mpfa_regions(g, node)[0]

    assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
    assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)

    for i in range(reg.edges.size):
        if len(reg.edges[i]) == 2:
            assert reg.edge_node_type[i] == ("cell", "face")
        if len(reg.edges[i]) == 3:
            assert reg.edge_node_type[i] == ("cell", "face", "cell")

    for surf, bound_info, node_types in zip(
        reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
    ):
        if "node" in node_types:
            # This is a boundary surface
            assert bound_info == True
            assert node_types == ("face", "node")
            assert surf[-1] == node
            assert np.any(surf[0] == faces)
        else:
            assert bound_info == False
            assert node_types == ("cell", "face")


def test_mpfa_2d_internal_boundary_tip():
    gb = pp.meshing.cart_grid([np.array([[1, 3], [1, 1]])], np.array([4, 2]))
    g = gb.grids_of_dimension(2)[0]

    node = 6
    cells = [1, 0, 4, 5]

    if g.cell_faces[22, 1] != 0:
        faces = [22, 1, 14, 6, 15]
    else:
        faces = [15, 1, 14, 6, 22]

    surfaces = [
        [faces[0], cells[0]],
        [cells[0], faces[1]],
        [faces[1], cells[1]],
        [cells[1], faces[2]],
        [faces[2], cells[2]],
        [cells[2], faces[3]],
        [faces[3], cells[3]],
        [cells[3], faces[4]],
        [faces[0], node],
        [faces[4], node],
    ]
    known_surfaces = np.array(surfaces)

    known_edges = np.array(
        [
            [faces[0], cells[0]],
            [cells[0], faces[1], cells[1]],
            [cells[1], faces[2], cells[2]],
            [cells[2], faces[3], cells[3]],
            [cells[3], faces[4]],
        ]
    )

    reg = ia_reg.extract_mpfa_regions(g, node)[0]

    assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
    assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)

    for i in range(reg.edges.size):
        if len(reg.edges[i]) == 2:
            assert reg.edge_node_type[i] == ("cell", "face")
        if len(reg.edges[i]) == 3:
            assert reg.edge_node_type[i] == ("cell", "face", "cell")

    for surf, bound_info, node_types in zip(
        reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
    ):
        if "node" in node_types:
            # This is a boundary surface
            assert bound_info == True
            assert node_types == ("face", "node")
            assert surf[-1] == node
            assert np.any(surf[0] == faces)
        else:
            assert bound_info == False
            assert node_types == ("cell", "face")


def test_mpfa_2d_internal_boundary_T_intersection():
    gb = pp.meshing.cart_grid(
        [
            np.array([[1, 3], [1, 1]]),
            np.array([[2, 2], [0, 1]]),
        ],
        np.array([4, 2]),
    )
    g = gb.grids_of_dimension(2)[0]

    nodes = [8, 9, 10]

    cn = g.cell_nodes().tocsr()
    cells = [cn[ni].indices for ni in nodes]

    fn = g.face_nodes.tocsr()
    cf = g.cell_faces.tocsc()

    faces = []

    for n, c in zip(nodes, cells):
        f = cf[:, c].indices
        loc_faces = np.intersect1d(fn[n].indices, f)

        if loc_faces.size == 3:
            # This is the upper side of the intersection, sort the faces
            sort_ind = np.argsort(g.face_centers[0, loc_faces])
            loc_faces = loc_faces[sort_ind]
        faces.append(loc_faces)

    for ind in range(len(nodes)):

        sz = cells[ind].size
        surf = [[faces[ind][i], cells[ind][i]] for i in range(sz)]
        surf += [[cells[ind][i], faces[ind][i + 1]] for i in range(sz)]

        surf += [[faces[ind][0], nodes[ind]]]
        surf += [[faces[ind][-1], nodes[ind]]]

        known_surfaces = np.array(surf)

        if sz == 1:
            known_edges = np.array(
                [
                    [cells[ind][0], faces[ind][0]],
                    [cells[ind][0], faces[ind][1]],
                ]
            )
        else:
            known_edges = np.array(
                [
                    [cells[ind][0], faces[ind][0]],
                    [cells[ind][0], faces[ind][1], cells[ind][1]],
                    [cells[ind][1], faces[ind][2]],
                ]
            )

        reg = ia_reg.extract_mpfa_regions(g, nodes[ind])[0]

        assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
        assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)

        for i in range(reg.edges.shape[0]):
            if len(reg.edges[i]) == 2:
                assert reg.edge_node_type[i] == ("cell", "face")
            if len(reg.edges[i]) == 3:
                assert reg.edge_node_type[i] == ("cell", "face", "cell")

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if "node" in node_types:
                # This is a boundary surface
                assert bound_info == True
                assert node_types == ("face", "node")
                assert surf[-1] == nodes[ind]
                assert np.any(surf[0] == faces[ind])
            else:
                assert bound_info == False
                assert node_types == ("cell", "face")


def test_mpfa_2d_internal_boundary_L_intersection():
    gb = pp.meshing.cart_grid(
        [
            np.array([[1, 3], [1, 1]]),
            np.array([[1, 1], [0, 1]]),
        ],
        np.array([4, 2]),
    )
    g = gb.grids_of_dimension(2)[0]

    nodes = [7, 8]

    cf = g.cell_faces.tocsc()
    fn = g.face_nodes.tocsr()
    cn = g.cell_nodes().tocsr()

    cells = [cn[ni].indices for ni in nodes]

    fn = g.face_nodes.tocsr()
    cf = g.cell_faces.tocsc()

    faces = []

    for n, c in zip(nodes, cells):
        f = cf[:, c].indices
        loc_faces = np.intersect1d(fn[n].indices, f)

        if loc_faces.size == 4:
            ymin = np.argmin(g.face_centers[1, loc_faces])
            xmin = np.argmin(g.face_centers[0, loc_faces])
            ymax = np.argmax(g.face_centers[1, loc_faces])
            xmax = np.argmax(g.face_centers[0, loc_faces])

            loc_faces = loc_faces[[ymin, xmin, ymax, xmax]]

        faces.append(loc_faces)

    for ind in range(len(nodes)):

        sz = cells[ind].size
        surf = [[faces[ind][i], cells[ind][i]] for i in range(sz)]
        surf += [[cells[ind][i], faces[ind][i + 1]] for i in range(sz)]

        surf += [[faces[ind][0], nodes[ind]]]
        surf += [[faces[ind][-1], nodes[ind]]]

        known_surfaces = np.array(surf)

        if sz == 1:
            known_edges = np.array(
                [
                    [cells[ind][0], faces[ind][0]],
                    [cells[ind][0], faces[ind][1]],
                ]
            )
        else:
            known_edges = np.array(
                [
                    [cells[ind][0], faces[ind][0]],
                    [cells[ind][0], faces[ind][1], cells[ind][1]],
                    [cells[ind][1], faces[ind][2], cells[ind][2]],
                    [cells[ind][2], faces[ind][3]],
                ]
            )

        reg = ia_reg.extract_mpfa_regions(g, nodes[ind])[0]

        assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
        assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)

        for i in range(reg.edges.shape[0]):
            if len(reg.edges[i]) == 2:
                assert reg.edge_node_type[i] == ("cell", "face")
            if len(reg.edges[i]) == 3:
                assert reg.edge_node_type[i] == ("cell", "face", "cell")

        for surf, bound_info, node_types in zip(
            reg.surfaces, reg.surface_is_boundary, reg.surface_node_type
        ):
            if "node" in node_types:
                # This is a boundary surface
                assert bound_info == True
                assert node_types == ("face", "node")
                assert surf[-1] == nodes[ind]
                assert np.any(surf[0] == faces[ind])
            else:
                assert bound_info == False
                assert node_types == ("cell", "face")


def test_mpfa_3d_internal_boundary_node_internal_to_fracture():
    gb = pp.meshing.cart_grid(
        [np.array([[0, 2, 2, 0], [1, 1, 1, 1], [0, 0, 2, 2]])], [2, 2, 2]
    )
    g = gb.grids_of_dimension(3)[0]

    fn = g.face_nodes.tocsr()
    if np.all(g.face_centers[1, fn[17].indices] <= 1):
        node = 17
    else:
        node = 18

    edge_nodes = [13]
    edge_pairs = [[5, 6], [15, 16], [19, 20], [29, 30]]
    for e in edge_pairs:
        if np.all(g.face_centers[1, fn[e[0]].indices] <= 1):
            edge_nodes.append(e[0])
        else:
            edge_nodes.append(e[1])

    cells = [0, 1, 4, 5]

    faces_x = [1, 7]
    faces_z = [28, 29]
    faces_y = []
    pairs = [[14, 36], [15, 37], [20, 38], [21, 39]]

    for i, p in enumerate(pairs):
        if g.cell_faces[p[0], cells[i]] != 0:
            faces_y.append(p[0])
        elif g.cell_faces[p[1], cells[i]] != 0:
            faces_y.append(p[1])
        else:
            raise ValueError()

    known_surfaces = np.array(
        [
            [cells[0], faces_x[0], edge_nodes[0]],
            [cells[0], faces_z[0], edge_nodes[0]],
            [cells[1], faces_x[0], edge_nodes[0]],
            [cells[1], faces_z[1], edge_nodes[0]],
            [cells[2], faces_x[1], edge_nodes[0]],
            [cells[2], faces_z[0], edge_nodes[0]],
            [cells[3], faces_x[1], edge_nodes[0]],
            [cells[3], faces_z[1], edge_nodes[0]],  # end of y=0 surfaces
            [cells[0], faces_y[0], edge_nodes[1]],
            [cells[0], faces_x[0], edge_nodes[1]],
            [cells[1], faces_y[1], edge_nodes[1]],
            [cells[1], faces_x[0], edge_nodes[1]],  # end of z=0.5 surface
            [cells[2], faces_y[2], edge_nodes[4]],
            [cells[2], faces_x[1], edge_nodes[4]],
            [cells[3], faces_y[3], edge_nodes[4]],
            [cells[3], faces_x[1], edge_nodes[4]],  # end of z=1.5 surface
            [cells[0], faces_y[0], edge_nodes[2]],
            [cells[0], faces_z[0], edge_nodes[2]],
            [cells[2], faces_y[2], edge_nodes[2]],
            [cells[2], faces_z[0], edge_nodes[2]],  # end of x=0.5 surface
            [cells[1], faces_y[1], edge_nodes[3]],
            [cells[1], faces_z[1], edge_nodes[3]],
            [cells[3], faces_y[3], edge_nodes[3]],
            [cells[3], faces_z[1], edge_nodes[3]],  # end of x=1.5 surface
            [faces_y[0], edge_nodes[1], node],
            [faces_y[0], edge_nodes[2], node],
            [faces_y[1], edge_nodes[1], node],
            [faces_y[1], edge_nodes[3], node],
            [faces_y[2], edge_nodes[4], node],
            [faces_y[2], edge_nodes[2], node],
            [faces_y[3], edge_nodes[4], node],
            [faces_y[3], edge_nodes[3], node],
        ]
    )

    known_edges = np.array(
        [
            [cells[0], faces_x[0], cells[1]],
            [cells[2], faces_x[1], cells[3]],
            [cells[0], faces_z[0], cells[2]],
            [cells[1], faces_z[1], cells[3]],
            [cells[0], faces_y[0]],
            [cells[1], faces_y[1]],
            [cells[2], faces_y[2]],
            [cells[3], faces_y[3]],
        ]
    )

    reg = ia_reg.extract_mpfa_regions(g, node)[0]

    assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
    assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)


def test_mpfa_3d_interal_boundary_node_on_fracture_edge():
    gb = pp.meshing.cart_grid(
        [np.array([[0, 2, 2, 0], [1, 1, 1, 1], [1, 1, 2, 2]])], [2, 2, 2]
    )
    g = gb.grids_of_dimension(3)[0]

    fn = g.face_nodes.tocsr()
    cf = g.cell_faces
    node = 13

    edge_nodes = [10, 16, 4, 12, 14]

    if np.all(g.face_centers[1, fn[23].indices] <= 1):
        edge_nodes.append(23)
        edge_nodes.append(24)
    else:
        edge_nodes.append(24)
        edge_nodes.append(23)

    cells = np.arange(g.num_cells)

    faces_x = np.array([1, 4, 7, 10])

    fy = [14, 15]
    if cf[19, 4] != 0:
        assert cf[20, 5] != 0
        fy += [20, 21, 36, 37]
    else:
        fy += [36, 37, 20, 21]
    faces_y = np.array(fy)

    faces_z = [28, 29, 30, 31]

    known_surfaces = np.array(
        [
            [cells[0], faces_x[0], edge_nodes[2]],
            [cells[0], faces_y[0], edge_nodes[2]],
            [cells[1], faces_x[0], edge_nodes[2]],
            [cells[1], faces_y[1], edge_nodes[2]],
            [cells[2], faces_x[1], edge_nodes[2]],
            [cells[2], faces_y[0], edge_nodes[2]],
            [cells[3], faces_x[1], edge_nodes[2]],
            [cells[3], faces_y[1], edge_nodes[2]],  # bottom
            [cells[4], faces_x[2], edge_nodes[5]],
            [cells[4], faces_y[2], edge_nodes[5]],
            [cells[5], faces_x[2], edge_nodes[5]],
            [cells[5], faces_y[3], edge_nodes[5]],
            [cells[6], faces_x[3], edge_nodes[6]],
            [cells[6], faces_y[4], edge_nodes[6]],
            [cells[7], faces_x[3], edge_nodes[6]],
            [cells[7], faces_y[5], edge_nodes[6]],  # Top
            [cells[0], faces_y[0], edge_nodes[3]],
            [cells[0], faces_z[0], edge_nodes[3]],
            [cells[2], faces_y[0], edge_nodes[3]],
            [cells[2], faces_z[2], edge_nodes[3]],
            [cells[4], faces_y[2], edge_nodes[3]],
            [cells[4], faces_z[0], edge_nodes[3]],
            [cells[6], faces_y[4], edge_nodes[3]],
            [cells[6], faces_z[2], edge_nodes[3]],  # left
            [cells[1], faces_y[1], edge_nodes[4]],
            [cells[1], faces_z[1], edge_nodes[4]],
            [cells[3], faces_y[1], edge_nodes[4]],
            [cells[3], faces_z[3], edge_nodes[4]],
            [cells[5], faces_y[3], edge_nodes[4]],
            [cells[5], faces_z[1], edge_nodes[4]],
            [cells[7], faces_y[5], edge_nodes[4]],
            [cells[7], faces_z[3], edge_nodes[4]],  # right
            [cells[0], faces_x[0], edge_nodes[0]],
            [cells[1], faces_x[0], edge_nodes[0]],
            [cells[0], faces_z[0], edge_nodes[0]],
            [cells[1], faces_z[1], edge_nodes[0]],
            [cells[4], faces_x[2], edge_nodes[0]],
            [cells[5], faces_x[2], edge_nodes[0]],
            [cells[4], faces_z[0], edge_nodes[0]],
            [cells[5], faces_z[1], edge_nodes[0]],  # front
            [cells[2], faces_x[1], edge_nodes[1]],
            [cells[3], faces_x[1], edge_nodes[1]],
            [cells[2], faces_z[2], edge_nodes[1]],
            [cells[3], faces_z[3], edge_nodes[1]],
            [cells[6], faces_x[3], edge_nodes[1]],
            [cells[7], faces_x[3], edge_nodes[1]],
            [cells[6], faces_z[2], edge_nodes[1]],
            [cells[7], faces_z[3], edge_nodes[1]],  # back
            [faces_y[2], edge_nodes[3], node],
            [faces_y[2], edge_nodes[5], node],
            [faces_y[3], edge_nodes[4], node],
            [faces_y[3], edge_nodes[5], node],
            [faces_y[4], edge_nodes[3], node],
            [faces_y[4], edge_nodes[6], node],
            [faces_y[5], edge_nodes[4], node],
            [faces_y[5], edge_nodes[6], node],  # internal
        ]
    )

    known_edges = np.array(
        [
            [cells[0], faces_x[0], cells[1]],
            [cells[2], faces_x[1], cells[3]],
            [cells[4], faces_x[2], cells[5]],
            [cells[6], faces_x[3], cells[7]],
            [cells[0], faces_z[0], cells[4]],
            [cells[1], faces_z[1], cells[5]],
            [cells[2], faces_z[2], cells[6]],
            [cells[3], faces_z[3], cells[7]],
            [cells[0], faces_y[0], cells[2]],
            [cells[1], faces_y[1], cells[3]],
            [cells[4], faces_y[2]],
            [cells[5], faces_y[3]],
            [cells[6], faces_y[4]],
            [cells[7], faces_y[5]],
        ]
    )

    reg = ia_reg.extract_mpfa_regions(g, node)[0]

    assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
    assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)


def test_mpfa_3d_internal_boundary_node_on_fracture_corner():
    gb = pp.meshing.cart_grid(
        [np.array([[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 2, 2]])], [2, 2, 2]
    )
    g = gb.grids_of_dimension(3)[0]

    cf = g.cell_faces
    node = 13

    edge_nodes = [10, 16, 4, 12, 14, 22]

    cells = np.arange(g.num_cells)

    faces_x = np.array([1, 4, 7, 10])

    fy = [14, 15, 20]
    if cf[21, 5] != 0:
        fy += [21, 36]
    else:
        fy += [36, 21]
    faces_y = np.array(fy)

    faces_z = [28, 29, 30, 31]

    known_surfaces = np.array(
        [
            [cells[0], faces_x[0], edge_nodes[2]],
            [cells[0], faces_y[0], edge_nodes[2]],
            [cells[1], faces_x[0], edge_nodes[2]],
            [cells[1], faces_y[1], edge_nodes[2]],
            [cells[2], faces_x[1], edge_nodes[2]],
            [cells[2], faces_y[0], edge_nodes[2]],
            [cells[3], faces_x[1], edge_nodes[2]],
            [cells[3], faces_y[1], edge_nodes[2]],  # bottom
            [cells[4], faces_x[2], edge_nodes[5]],
            [cells[4], faces_y[2], edge_nodes[5]],
            [cells[5], faces_x[2], edge_nodes[5]],
            [cells[5], faces_y[3], edge_nodes[5]],
            [cells[6], faces_x[3], edge_nodes[5]],
            [cells[6], faces_y[2], edge_nodes[5]],
            [cells[7], faces_x[3], edge_nodes[5]],
            [cells[7], faces_y[4], edge_nodes[5]],  # Top
            [cells[0], faces_y[0], edge_nodes[3]],
            [cells[0], faces_z[0], edge_nodes[3]],
            [cells[2], faces_y[0], edge_nodes[3]],
            [cells[2], faces_z[2], edge_nodes[3]],
            [cells[4], faces_y[2], edge_nodes[3]],
            [cells[4], faces_z[0], edge_nodes[3]],
            [cells[6], faces_y[2], edge_nodes[3]],
            [cells[6], faces_z[2], edge_nodes[3]],  # left
            [cells[1], faces_y[1], edge_nodes[4]],
            [cells[1], faces_z[1], edge_nodes[4]],
            [cells[3], faces_y[1], edge_nodes[4]],
            [cells[3], faces_z[3], edge_nodes[4]],
            [cells[5], faces_y[3], edge_nodes[4]],
            [cells[5], faces_z[1], edge_nodes[4]],
            [cells[7], faces_y[4], edge_nodes[4]],
            [cells[7], faces_z[3], edge_nodes[4]],  # right
            [cells[0], faces_x[0], edge_nodes[0]],
            [cells[1], faces_x[0], edge_nodes[0]],
            [cells[0], faces_z[0], edge_nodes[0]],
            [cells[1], faces_z[1], edge_nodes[0]],
            [cells[4], faces_x[2], edge_nodes[0]],
            [cells[5], faces_x[2], edge_nodes[0]],
            [cells[4], faces_z[0], edge_nodes[0]],
            [cells[5], faces_z[1], edge_nodes[0]],  # front
            [cells[2], faces_x[1], edge_nodes[1]],
            [cells[3], faces_x[1], edge_nodes[1]],
            [cells[2], faces_z[2], edge_nodes[1]],
            [cells[3], faces_z[3], edge_nodes[1]],
            [cells[6], faces_x[3], edge_nodes[1]],
            [cells[7], faces_x[3], edge_nodes[1]],
            [cells[6], faces_z[2], edge_nodes[1]],
            [cells[7], faces_z[3], edge_nodes[1]],  # back
            [faces_y[3], edge_nodes[4], node],
            [faces_y[3], edge_nodes[5], node],
            [faces_y[4], edge_nodes[4], node],
            [faces_y[4], edge_nodes[5], node],  # internal
        ]
    )

    known_edges = np.array(
        [
            [cells[0], faces_x[0], cells[1]],
            [cells[2], faces_x[1], cells[3]],
            [cells[4], faces_x[2], cells[5]],
            [cells[6], faces_x[3], cells[7]],
            [cells[0], faces_z[0], cells[4]],
            [cells[1], faces_z[1], cells[5]],
            [cells[2], faces_z[2], cells[6]],
            [cells[3], faces_z[3], cells[7]],
            [cells[0], faces_y[0], cells[2]],
            [cells[1], faces_y[1], cells[3]],
            [cells[4], faces_y[2], cells[6]],
            [cells[5], faces_y[3]],
            [cells[7], faces_y[4]],
        ]
    )

    reg = ia_reg.extract_mpfa_regions(g, node)[0]

    assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
    assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)


def test_mpfa_3d_internal_boundary_on_domain_boundary():
    gb = pp.meshing.cart_grid(
        [np.array([[1, 2, 2, 1], [1, 1, 1, 1], [0, 0, 1, 1]])], [3, 2, 2]
    )
    g = gb.grids_of_dimension(3)[0]

    cf = g.cell_faces
    node = 5

    edge_nodes = [1, 9, 17, 4, 6]

    cells = [0, 1, 3, 4]

    faces_x = [1, 5]
    if cf[20, 1] != 0:
        faces_y = [19, 20, g.num_faces - 1]
    else:
        faces_y = [19, g.num_faces - 1, 20]

    faces_z = [34, 35, 37, 38]

    known_surfaces = np.array(
        [
            [cells[0], faces_y[0], edge_nodes[3]],
            [cells[0], faces_z[0], edge_nodes[3]],
            [cells[2], faces_y[0], edge_nodes[3]],
            [cells[2], faces_z[2], edge_nodes[3]],  # left
            [cells[1], faces_y[1], edge_nodes[4]],
            [cells[1], faces_z[1], edge_nodes[4]],
            [cells[3], faces_y[2], edge_nodes[4]],
            [cells[3], faces_z[3], edge_nodes[4]],  # right
            [cells[0], faces_x[0], edge_nodes[2]],
            [cells[1], faces_x[0], edge_nodes[2]],
            [cells[0], faces_y[0], edge_nodes[2]],
            [cells[1], faces_y[1], edge_nodes[2]],
            [cells[2], faces_x[1], edge_nodes[2]],
            [cells[3], faces_x[1], edge_nodes[2]],
            [cells[2], faces_y[0], edge_nodes[2]],
            [cells[3], faces_y[2], edge_nodes[2]],  # top
            [faces_z[0], edge_nodes[0], node],
            [faces_z[0], edge_nodes[3], node],
            [faces_z[1], edge_nodes[0], node],
            [faces_z[1], edge_nodes[4], node],
            [faces_z[2], edge_nodes[1], node],
            [faces_z[2], edge_nodes[3], node],
            [faces_z[3], edge_nodes[1], node],
            [faces_z[3], edge_nodes[4], node],
            [cells[0], faces_x[0], edge_nodes[0]],
            [cells[0], faces_z[0], edge_nodes[0]],
            [cells[1], faces_x[0], edge_nodes[0]],
            [cells[1], faces_z[1], edge_nodes[0]],  # front
            [cells[2], faces_x[1], edge_nodes[1]],
            [cells[2], faces_z[2], edge_nodes[1]],
            [cells[3], faces_x[1], edge_nodes[1]],
            [cells[3], faces_z[3], edge_nodes[1]],  # back
            [faces_y[1], edge_nodes[2], node],
            [faces_y[1], edge_nodes[4], node],
            [faces_y[2], edge_nodes[2], node],
            [faces_y[2], edge_nodes[4], node],
        ]
    )

    known_edges = np.array(
        [
            [cells[0], faces_x[0], cells[1]],
            [cells[2], faces_x[1], cells[3]],
            [cells[0], faces_y[0], cells[2]],
            [cells[1], faces_y[1]],
            [cells[3], faces_y[2]],
            [cells[0], faces_z[0]],
            [cells[1], faces_z[1]],
            [cells[2], faces_z[2]],
            [cells[3], faces_z[3]],
        ]
    )

    reg = ia_reg.extract_mpfa_regions(g, node)[0]

    assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
    assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)


test_mpfa_3d_internal_boundary_on_domain_boundary()


def test_mpfa_3d_internal_boundary_crossing_internal_interior():
    f1 = np.array([[0, 2, 2, 0], [1, 1, 1, 1], [1, 1, 3, 3]])
    f2 = np.array([[1, 1, 1, 1], [0, 2, 2, 0], [2, 2, 3, 3]])
    gb = pp.meshing.cart_grid([f1, f2], [2, 2, 3])
    g = gb.grids_of_dimension(3)[0]

    cn = g.cell_nodes().tocsr()

    cells = [4, 5, 8, 9]
    if np.all(g.cell_centers[1, cn[23].indices] < 1):
        node = 23
    else:
        node = 24

    edge_nodes = [13, 19]
    if np.all(g.cell_centers[1, cn[21].indices] < 1):
        edge_nodes.append(21)
    else:
        edge_nodes.append(22)
    if np.all(g.cell_centers[1, cn[25].indices] < 1):
        edge_nodes.append(25)
    else:
        edge_nodes.append(26)

    # Top-most edge-coordinate has four coordinates, need two of them
    for ni in range(36, 39):
        if cells[2] in cn[ni].indices:
            edge_nodes.append(ni)
            break

    for ni in range(36, 39):
        if cells[3] in cn[ni].indices:
            edge_nodes.append(ni)
            break

    # Cross our fingers this is the correct face indices after splitting
    faces_x = [7, 56, 13]
    faces_y = [52, 53, 54, 55]
    faces_z = [44, 45]

    known_surfaces = np.array(
        [
            [cells[0], faces_y[0], edge_nodes[0]],
            [cells[0], faces_x[0], edge_nodes[0]],
            [cells[1], faces_y[1], edge_nodes[0]],
            [cells[1], faces_x[0], edge_nodes[0]],  # bottom
            [cells[0], faces_z[0], edge_nodes[2]],
            [cells[0], faces_y[0], edge_nodes[2]],
            [cells[2], faces_z[0], edge_nodes[2]],
            [cells[2], faces_y[2], edge_nodes[2]],  # left
            [cells[1], faces_y[1], edge_nodes[3]],
            [cells[1], faces_z[1], edge_nodes[3]],
            [cells[3], faces_y[3], edge_nodes[3]],
            [cells[3], faces_z[1], edge_nodes[3]],  # right
            [cells[2], faces_x[1], edge_nodes[4]],
            [cells[2], faces_y[2], edge_nodes[4]],
            [cells[3], faces_x[2], edge_nodes[5]],
            [cells[3], faces_y[3], edge_nodes[5]],  # top
            [cells[0], faces_x[0], edge_nodes[1]],
            [cells[0], faces_z[0], edge_nodes[1]],
            [cells[1], faces_x[0], edge_nodes[1]],
            [cells[1], faces_z[1], edge_nodes[1]],
            [cells[2], faces_x[1], edge_nodes[1]],
            [cells[2], faces_z[0], edge_nodes[1]],
            [cells[3], faces_x[2], edge_nodes[1]],
            [cells[3], faces_z[1], edge_nodes[1]],  # front
            [faces_y[0], edge_nodes[0], node],
            [faces_y[0], edge_nodes[2], node],
            [faces_y[1], edge_nodes[0], node],
            [faces_y[1], edge_nodes[3], node],
            [faces_y[2], edge_nodes[2], node],
            [faces_y[2], edge_nodes[4], node],
            [faces_y[3], edge_nodes[3], node],
            [faces_y[3], edge_nodes[5], node],  # back
            [edge_nodes[1], faces_x[1], node],
            [edge_nodes[1], faces_x[2], node],
            [edge_nodes[4], faces_x[1], node],
            [edge_nodes[5], faces_x[2], node],
        ]
    )

    known_edges = np.array(
        [
            [cells[0], faces_x[0], cells[1]],
            [cells[0], faces_z[0], cells[2]],
            [cells[1], faces_z[1], cells[3]],
            [cells[0], faces_y[0]],
            [cells[1], faces_y[1]],
            [cells[2], faces_y[2]],
            [cells[3], faces_y[3]],
            [cells[2], faces_x[1]],
            [cells[3], faces_x[2]],
        ]
    )

    reg = ia_reg.extract_mpfa_regions(g, node)[0]

    assert test_utils.compare_arrays(known_surfaces.T, reg.surfaces.T)
    assert test_utils.compare_arrays_varying_size(known_edges.T, reg.edges.T)


test_mpfa_3d_internal_boundary_crossing_internal_interior()


def test_mpfa_3d_internal_boundaries_crossing_on_edge_of_both_fractures():
    pass


if __name__ == "__main__":
    unittest.main()
