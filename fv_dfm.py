"""

"""
import numpy as np
import porepy as pp
import scipy.sparse as sps

import multiprocessing as mp
from functools import partial

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

    def set_parameters_cell_basis(self, gb, macro_data):
        """
        Assign parameters for the micro gb. Very simple for now, this must be improved.

        Args:
            gb (TYPE): the micro gb.

        Returns:
            None.

        """

        macro_bc = macro_data["bc"]
        # For now, we use constant matrix permeability for the micro domains. Not sure
        # what is the best way to generalize this, probably, we can just skip it.
        permeability = macro_data["permeability"]

        # First initialize data
        for g, d in gb:
            domain_boundary = np.logical_and(
                g.tags["domain_boundary_faces"],
                np.logical_not(g.tags["fracture_faces"]),
            )

            boundary_faces = np.where(domain_boundary)[0]
            if domain_boundary.size > 0:
                bc_type = boundary_faces.size * ["dir"]
            else:
                bc_type = np.empty(0)

            micro_bc = pp.BoundaryCondition(g, boundary_faces, bc_type)
            if hasattr(g, "face_on_macro_bound"):
                micro_ind = g.face_on_macro_bound
                macro_ind = g.macro_face_ind

                micro_bc.is_neu[micro_ind] = macro_bc.is_neu[macro_ind]
                micro_bc.is_dir[micro_ind] = macro_bc.is_dir[macro_ind]

            param = {"bc": micro_bc}

            perm = pp.SecondOrderTensor(kxx=permeability * np.ones(g.num_cells))

            param["second_order_tensor"] = perm

            pp.initialize_default_data(g, d, self.keyword, param)

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            g1, g2 = gb.nodes_of_edge(e)

            param = {}

            if g1.from_fracture:
                param["normal_diffusivity"] = 1e4

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

        micro_network = parameter_dictionary[self.network_keyword]

        num_processes = data.get("num_processes", 1)
        discr_ig = partial(self._discretize_interaction_region, g, micro_network, parameter_dictionary)
        if num_processes == 1:
            # run the code in serial, useful also for debug
            out = [discr_ig(reg) for reg in self._interaction_regions(g)]
        else:
            # run the code in parallel
            with mp.Pool(processes=num_processes) as p:
                out = p.map(discr_ig, list(self._interaction_regions(g)))

        # Data structures to build up the flux discretization as a sparse coo-matrix
        rows_flux, cols_flux, data_flux = [], [], []

        # Data structures for the discretization of boundary conditions
        rows_bound, cols_bound, data_bound = [], [], []

        # unpack all the values
        for fi, ci, trm, ri_bound, ci_bound, trm_bound in out:
            rows_flux += fi
            cols_flux += ci
            data_flux += trm

            rows_bound += ri_bound
            cols_bound += ci_bound
            data_bound += trm_bound

        # Construct the global matrix
        flux = sps.coo_matrix(
            (data_flux, (rows_flux, cols_flux)), shape=(g.num_faces, g.num_cells)
        ).tocsr()

        # Construct the global flux matrix
        bound_flux = sps.coo_matrix(
            (data_bound, (rows_bound, cols_bound)), shape=(g.num_faces, g.num_faces)
        ).tocsr()

        # For Neumann boundaries, we should not use the flux discretization (the flux
        # is known). The boundary discretization is modified to simply reuse the flux.
        bc = parameter_dictionary["bc"]
        neumann_faces = np.where(bc.is_neu)[0]
        pp.fvutils.zero_out_sparse_rows(flux, neumann_faces)
        # Not sure about sign here
        bound_flux = pp.fvutils.zero_out_sparse_rows(bound_flux, neumann_faces, diag=1)

        matrix_dictionary[self.flux_matrix_key] = flux
        matrix_dictionary[self.bound_flux_matrix_key] = bound_flux

        # Empty discretization of vector sources - we will not provide this for the
        # foreseeable future.
        matrix_dictionary[self.vector_source_matrix_key] = sps.csc_matrix(
            (g.num_faces, g.num_cells * g.dim)
        )

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

    def _discretize_interaction_region(self, g, micro_network, parameter_dictionary, reg):

        # Add the fractures to be upscaled
        self._add_network_to_upscale(reg, micro_network)

        # construct the sequence of local grid buckets
        gb_set = LocalGridBucketSet(g.dim, reg)
        gb_set.construct_local_buckets()

        # First basis functions for local problems
        (
            basis_functions,
            cc_assembler,
            cc_bc_values,
        ) = local_problems.cell_basis_functions(
            reg, gb_set, self, parameter_dictionary
        )
        # TODO: Operator to reconstruct boundary pressure from the basis functions

        # Call method to transfer basis functions to transmissibilties over coarse
        # edges
        ci, fi, trm = local_problems.compute_transmissibilies(
            reg,
            gb_set,
            basis_functions,
            cc_assembler,
            cc_bc_values,
            g,
            self,
            parameter_dictionary,
        )

        (
            ci_bound,
            ri_bound,
            trm_bound,
        ) = local_problems.discretize_boundary_conditions(
            reg, gb_set, self, parameter_dictionary, g
        )

        return fi, ci, trm, ri_bound, ci_bound, trm_bound

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
