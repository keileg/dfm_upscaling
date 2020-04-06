"""

"""
import numpy as np
import porepy as pp

import interaction_region as ia_reg
import local_problems
from local_grid_bucket import LocalGridBucketSet


class FVDFM(pp.FVElliptic):
    def __init__(self, keyword="flow"):
        super(FVDFM, self).__init__(keyword)

        # Keyword used to identify the micro network to be upscaled.
        self.network_keyword = "micro_network"

        self.cell_variable = "pressure"
        self.mortar_variable = "mortar_flux"

        self.cell_discr = self.cell_variable + "_discr"
        self.mortar_discr = self.mortar_variable + "_discr"

        # method for the discretization (tpfa or mpfa so far)
        self.method = None

    def set_parameters_cell_basis(self, gb):
        """
        Assign parameters for the micro gb. Very simple for now, this must be improved.

        Args:
            gb (TYPE): the micro gb.

        Returns:
            None.

        """

        keyword = "flow"

        # First initialize data
        for g, d in gb:

            param = {}

            if g.dim == gb.dim_max():
                domain_boundary = np.logical_and(
                    g.tags["domain_boundary_faces"],
                    np.logical_not(g.tags["fracture_faces"]),
                )

                boundary_faces = np.where(domain_boundary)[0]
                bc_type = boundary_faces.size * ["dir"]

                bc = pp.BoundaryCondition(g, boundary_faces, bc_type)
                param["bc"] = bc

            pp.initialize_default_data(g, d, self.keyword, param)

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            g1, g2 = gb.nodes_of_edge(e)

            param = {}

            if g1.from_fracture:
                param["normal_diffusivity"] = 1e1

            pp.initialize_data(mg, d, self.keyword, param)

    def set_variables_discretizations_cell_basis(self, gb):
        """
        Assign variables, and set discretizations for the micro gb.

        NOTE: keywords and variable names are hardcoded here. This should be centralized.
        @Eirik we are keeping the same nomenclature, since the gb are different maybe we can
        change it a bit

        Args:
            gb (TYPE): the micro gb.

        Returns:
            None.

        """
        # Use mpfa for the fine-scale problem for now. We may generalize this at some
        # point, but that should be a technical detail.
        fine_scale_dicsr = pp.Mpfa(self.keyword)

        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {self.cell_variable: {"cells": 1, "faces": 0}}
            d[pp.DISCRETIZATION] = {
                self.cell_variable: {self.cell_discr: fine_scale_dicsr}
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.keyword: {}}

        # Loop over the edges in the GridBucket, define primary variables and discretizations
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {self.mortar_variable: {"cells": 1}}
            # The type of lower-dimensional discretization depends on whether this is a
            # (part of a) fracture, or a transition between two line or surface grids.
            if g1.from_fracture:
                mortar_discr = pp.RobinCoupling(
                    self.keyword, fine_scale_dicsr, fine_scale_dicsr
                )
            else:
                mortar_discr = pp.FluxPressureContinuity(
                    self.keyword, fine_scale_dicsr, fine_scale_dicsr
                )

            d[pp.COUPLING_DISCRETIZATION] = {
                self.mortar_discr: {
                    g1: (self.cell_variable, self.cell_discr),
                    g2: (self.cell_variable, self.cell_discr),
                    e: (self.mortar_variable, mortar_discr),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.keyword: {}}

    def discretize(self, g, data):

        # Get the dictionaries for storage of data and discretization matrices
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        # Allocate the data to store matrix entries, that's an efficient
        # way to create a sparse matrix.

        ###### @Eirik, I imagine we can compute this for tpfa and mpfa
        # EK: Use arrays, convert to np.arrays afterwards. I don't want to think about
        # how many orders of maginuted faster the rest of the code must be before this
        #
        I = []
        J = []
        dataIJ = []

        micro_network = parameter_dictionary[self.network_keyword]

        # This for-loop could be parallelized. TODO
        for reg in self._interaction_regions(g):

            # Add the fractures to be upscaled
            self._add_network_to_upscale(reg, micro_network)

            # construct the sequence of local grid buckets
            gb_set = LocalGridBucketSet(g.dim, reg)
            gb_set.construct_local_buckets()

            # First basis functions for local problems
            basis = local_problems.cell_basis_functions(reg, gb_set, self)

            # Call method to transfer basis functions to transmissibilties over coarse
            # edges
            import pdb

            pdb.set_trace()
            print(basis)

            # I[idx] =
            # J[idx] =
            # dataIJ[idx] =
            # idx += 1

        # Construct the global matrix

        mass = sps.coo_matrix((dataIJ, (I, J))).tocsr()

    def _interaction_regions(self, g):
        raise NotImplementedError

    def _add_network_to_upscale(self, reg, network):
        """
        Add the fractures to the local interation region depending on the spatial dimension
        """
        if isinstance(network, pp.FractureNetwork3d):
            reg.add_fractures(fractures=network._fractures)
        elif isinstance(network, pp.FractureNetwork2d):
            reg.add_fractures(points=network.pts, edges=network.edges)
        else:
            raise ValueError


class Tpfa_DFM(FVDFM):
    """
    Define the specific class for tpfa upscaling
    """

    def __init__(self, keyword="flow"):
        super(Tpfa_DFM, self).__init__(keyword)
        self.method = pp.Tpfa

    def _interaction_regions(self, g):
        for fi in range(g.num_faces):
            yield ia_reg.extract_tpfa_regions(g, fi)[0]


class Mpfa_DFM(FVDFM):
    """
    Define the specific class for tpfa upscaling
    """

    def __init__(self, keyword="flow"):
        super(Mpfa_DFM, self).__init__(keyword)
        self.method = pp.Mpfa

    def _interaction_regions(self, g):
        for ni in range(g.num_nodes):
            yield ia_reg.extract_mpfa_regions(g, ni)[0]
