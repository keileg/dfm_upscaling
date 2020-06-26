import numpy as np
import unittest
import porepy as pp

from test import test_utils

import sys

sys.path.insert(0, "../")
from fracture_set import *


class TestFractureSet(unittest.TestCase):
    def test_2d_import(self):
        file_name = "network.csv"
        known_pts, known_edges = self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)

        self.assertTrue(test_utils.compare_arrays(network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(network.edges, known_edges))

    def test_2d_dummy_criterion(self):
        file_name = "network.csv"
        known_pts, known_edges = self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)

        macro_network, micro_network = split_network(network, Criterion.none)

        self.assertTrue(test_utils.compare_arrays(macro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(macro_network.edges, known_edges))

        known_edges = np.empty((2, 0))
        self.assertTrue(test_utils.compare_arrays(micro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(micro_network.edges, known_edges))

    def test_2d_branch_length_criterion(self):
        file_name = "network.csv"
        known_pts, _ = self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)

        macro_network, micro_network = split_network(
            network, Criterion.smaller_than, branch_length=0.3
        )

        known_edges = np.array([[0, 5, 7], [1, 6, 8]])
        self.assertTrue(test_utils.compare_arrays(macro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(macro_network.edges, known_edges))

        known_edges = np.array([[2, 3], [3, 4]])
        self.assertTrue(test_utils.compare_arrays(micro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(micro_network.edges, known_edges))

    def test_2d_isolated_branch_criterion(self):
        file_name = "network.csv"
        known_pts, _ = self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)

        macro_network, micro_network = split_network(network, Criterion.isolated)

        known_edges = np.array([[2, 3], [3, 4]])
        self.assertTrue(test_utils.compare_arrays(macro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(macro_network.edges, known_edges))

        known_edges = np.array([[0, 5, 7], [1, 6, 8]])
        self.assertTrue(test_utils.compare_arrays(micro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(micro_network.edges, known_edges))

    def test_2d_branch_length_or_isolated_branch_criterion(self):
        file_name = "network.csv"
        known_pts, _ = self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)

        criteria = [Criterion.smaller_than, Criterion.isolated]
        macro_network, micro_network = split_network(
            network, criteria, logical_op=np.logical_or, branch_length=0.2
        )

        known_edges = np.array([[2], [3]])
        self.assertTrue(test_utils.compare_arrays(macro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(macro_network.edges, known_edges))

        known_edges = np.array([[0, 3, 5, 7], [1, 4, 6, 8]])
        self.assertTrue(test_utils.compare_arrays(micro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(micro_network.edges, known_edges))

    def test_2d_branch_length_and_isolated_branch_criterion(self):
        file_name = "network.csv"
        known_pts, _ = self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)

        criteria = [Criterion.smaller_than, Criterion.isolated]
        macro_network, micro_network = split_network(
            network, criteria, logical_op=np.logical_and, branch_length=0.35
        )

        known_edges = np.array([[0, 2, 3, 7], [1, 3, 4, 8]])
        self.assertTrue(test_utils.compare_arrays(macro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(macro_network.edges, known_edges))

        known_edges = np.array([[5], [6]])
        self.assertTrue(test_utils.compare_arrays(micro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(micro_network.edges, known_edges))

    def test_2d_branch_length_criterion_network_split(self):
        file_name = "network.csv"
        self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)
        network = network.split_intersections()

        macro_network, micro_network = split_network(
            network, Criterion.smaller_than, branch_length=0.1
        )

        known_edges = np.array([[0, 1, 2, 5, 6, 7], [3, 3, 3, 9, 9, 9]])
        known_pts = np.array(
            [
                [0.25, 0.75, 0.5, 0.5, 0.5, 0.35, 0.65, 0.5, 0.5, 0.5],
                [0.25, 0.25, 0.0, 0.25, 0.35, 0.75, 0.75, 1.0, 0.65, 0.75],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(macro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(macro_network.edges, known_edges))

        known_edges = np.array([[3, 8], [4, 9]])
        self.assertTrue(test_utils.compare_arrays(micro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(micro_network.edges, known_edges))

    def test_2d_isolated_branch_criterion_network_split(self):
        file_name = "network.csv"
        self._2d_network(file_name)
        network = pp.fracture_importer.network_2d_from_csv(file_name)
        network = network.split_intersections()

        macro_network, micro_network = split_network(network, Criterion.isolated)

        known_edges = np.array([[0, 1, 2, 3, 5, 6, 7, 8], [3, 3, 3, 4, 9, 9, 9, 9]])
        known_pts = np.array(
            [
                [0.25, 0.75, 0.5, 0.5, 0.5, 0.35, 0.65, 0.5, 0.5, 0.5],
                [0.25, 0.25, 0.0, 0.25, 0.35, 0.75, 0.75, 1.0, 0.65, 0.75],
            ]
        )
        self.assertTrue(test_utils.compare_arrays(macro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(macro_network.edges, known_edges))

        known_edges = np.empty((2, 0))
        self.assertTrue(test_utils.compare_arrays(micro_network.pts, known_pts))
        self.assertTrue(test_utils.compare_arrays(micro_network.edges, known_edges))

    def _2d_network(self, file_name):
        content = (
            "FID,START_X,START_Y,END_X,END_Y\n"
            "0,0.25,0.25,0.75,0.25\n"
            "1,0.5,0,0.5,0.25\n"
            "2,0.5,0.25,0.5,0.35\n"
            "3,0.35,0.75,0.65,0.75\n"
            "4,0.5,1,0.5,0.65"
        )
        with open(file_name, "w") as out_file:
            out_file.write(content)
        known_pts = np.array(
            [
                [0.25, 0.75, 0.5, 0.5, 0.5, 0.35, 0.65, 0.5, 0.5],
                [0.25, 0.25, 0.0, 0.25, 0.35, 0.75, 0.75, 1.0, 0.65],
            ]
        )
        known_edges = np.array([[0, 2, 3, 5, 7], [1, 3, 4, 6, 8]])
        return known_pts, known_edges


if __name__ == "__main__":
    unittest.main()
