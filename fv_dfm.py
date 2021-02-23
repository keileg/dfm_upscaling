"""

"""
from typing import Dict
import numpy as np
import porepy as pp
import scipy.sparse as sps

import multiprocessing as mp
from functools import partial

from dfm_upscaling import interaction_region as ia_reg
from dfm_upscaling import local_problems
from dfm_upscaling.local_grid_bucket import LocalGridBucketSet


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

    def set_parameters_cell_basis(self, gb: pp.GridBucket, data: Dict):
        """
        Assign parameters for the micro gb. Very simple for now, this must be improved.

        Args:
            gb (TYPE): the micro gb.

        Returns:
            None.

        """
        # First initialize data
        for g, d in gb:

            d["Aavatsmark_transmissibilities"] = True

            domain_boundary = np.logical_and(
                g.tags["domain_boundary_faces"],
                np.logical_not(g.tags["fracture_faces"]),
            )

            boundary_faces = np.where(domain_boundary)[0]
            if domain_boundary.size > 0:
                bc_type = boundary_faces.size * ["dir"]
            else:
                bc_type = np.empty(0)

            bc = pp.BoundaryCondition(g, boundary_faces, bc_type)
            if hasattr(g, "face_on_macro_bound"):
                micro_ind = g.face_on_macro_bound
                macro_ind = g.macro_face_ind

                bc.is_neu[micro_ind] = data["bc_macro"]["bc"].is_neu[macro_ind]
                bc.is_dir[micro_ind] = data["bc_macro"]["bc"].is_dir[macro_ind]

            param = {"bc": bc}
            perm = data["g_data"](g)["second_order_tensor"]
            param["second_order_tensor"] = perm

            # Use python inverter for mpfa for small problems, where it does not pay off
            # to fire up numba. The set threshold value is somewhat randomly picked.
            if g.num_cells < 100:
                param["mpfa_inverter"] = "python"

            pp.initialize_default_data(g, d, self.keyword, param)

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            g1, g2 = gb.nodes_of_edge(e)

            param = {}

            if not hasattr(g1, "is_auxiliary") or not g1.is_auxiliary:
                #param["normal_diffusivity"] = 1e2
                pass

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
        # In 1d, mpfa will end up calling tpfa, so use this directly, in a hope that
        # the reduced amount of boilerplate will save some time.
        fine_scale_dicsr_1d = pp.Tpfa(self.keyword)

        void_discr = pp.EllipticDiscretizationZeroPermeability(self.keyword)

        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {self.cell_variable: {"cells": 1, "faces": 0}}
            if hasattr(g, "is_auxiliary") and g.is_auxiliary:
                d[pp.DISCRETIZATION] = {
                    self.cell_variable: {self.cell_discr: void_discr}
                }
            #                print(g.cell_centers)
            else:
                if g.dim > 1:
                    d[pp.DISCRETIZATION] = {
                        self.cell_variable: {self.cell_discr: fine_scale_dicsr}
                    }
                else:
                    d[pp.DISCRETIZATION] = {
                        self.cell_variable: {self.cell_discr: fine_scale_dicsr_1d}
                    }

            d[pp.DISCRETIZATION_MATRICES] = {self.keyword: {}}

        # Loop over the edges in the GridBucket, define primary variables and discretizations
        # NOTE: No need to differ between Mpfa and Tpfa here; their treatment of interface
        # discretizations are the same.
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {self.mortar_variable: {"cells": 1}}
            # The type of lower-dimensional discretization depends on whether this is a
            # (part of a) fracture, or a transition between two line or surface grids.
            if hasattr(g1, "is_auxiliary") and g1.is_auxiliary:
                if g2.dim > 1:
                    mortar_discr = pp.FluxPressureContinuity(
                        self.keyword, fine_scale_dicsr, void_discr
                    )
                else:
                    mortar_discr = pp.FluxPressureContinuity(
                        self.keyword, fine_scale_dicsr_1d, void_discr
                    )
            else:
                if g1.dim > 1:
                    mortar_discr = pp.FluxPressureContinuity( #RobinCoupling(
                        self.keyword, fine_scale_dicsr, fine_scale_dicsr
                    )
                elif g1.dim == 1:
                    mortar_discr = pp.FluxPressureContinuity( #RobinCoupling(
                        self.keyword, fine_scale_dicsr, fine_scale_dicsr_1d
                    )
                else:
                    mortar_discr = pp.FluxPressureContinuity( #RobinCoupling(
                        self.keyword, fine_scale_dicsr_1d, fine_scale_dicsr_1d
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
        # The the local mesh arguments
        local_mesh_args = data.get("local_mesh_args", None)

        micro_network = parameter_dictionary[self.network_keyword]
        num_processes = parameter_dictionary.get("num_processes", 1)
        discr_ig = partial(
            self._discretize_interaction_region, g, micro_network, parameter_dictionary, local_mesh_args
        )
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

        # data structures for discretization of pressure trace operator on macro boundaries
        rows_cell_trace, cols_cell_trace, data_cell_trace = [], [], []
        rows_face_trace, cols_face_trace, data_face_trace = [], [], []

        # unpack all the values
        for reg_values in out:
            cell, bound, cell_trace, face_trace = reg_values
            cols_flux += cell[0]
            rows_flux += cell[1]
            data_flux += cell[2]

            cols_bound += bound[0]
            rows_bound += bound[1]
            data_bound += bound[2]

            cols_cell_trace += cell_trace[0]
            rows_cell_trace += cell_trace[1]
            data_cell_trace += cell_trace[2]

            cols_face_trace += face_trace[0]
            rows_face_trace += face_trace[1]
            data_face_trace += face_trace[2]

        # Construct the global matrix
        flux = sps.coo_matrix(
            (data_flux, (rows_flux, cols_flux)), shape=(g.num_faces, g.num_cells)
        ).tocsr()

        # Construct the global flux matrix
        bound_flux = sps.coo_matrix(
            (data_bound, (rows_bound, cols_bound)), shape=(g.num_faces, g.num_faces)
        ).tocsr()

        bound_pressure_cell = sps.coo_matrix(
            (data_cell_trace, (rows_cell_trace, cols_cell_trace)),
            shape=(g.num_faces, g.num_cells),
        ).tocsr()

        bound_pressure_face = sps.coo_matrix(
            (data_face_trace, (rows_face_trace, cols_face_trace)),
            shape=(g.num_faces, g.num_faces),
        ).tocsr()

        # For Neumann boundaries, we should not use the flux discretization (the flux
        # is known). The boundary discretization is modified to simply reuse the flux.
        bc = parameter_dictionary["bc"]
        neumann_faces = np.where(bc.is_neu)[0]
        pp.fvutils.zero_out_sparse_rows(flux, neumann_faces)

        # Set sign of Neumann faces according to the divergence operator. This will
        # counteract minus signs in the divergence during assembly.
        sgn, _ = g.signs_and_cells_of_boundary_faces(neumann_faces)
        # Here we also zero out non-diagonal terms, but these should be zero anyhow
        # (by construction of the coarse problems), but better safe than sorry
        bound_flux = pp.fvutils.zero_out_sparse_rows(
            bound_flux, neumann_faces, diag=sgn
        )

        matrix_dictionary[self.flux_matrix_key] = flux
        matrix_dictionary[self.bound_flux_matrix_key] = bound_flux
        matrix_dictionary[self.bound_pressure_cell_matrix_key] = bound_pressure_cell
        matrix_dictionary[self.bound_pressure_face_matrix_key] = bound_pressure_face

        # Empty discretization of vector sources - we will not provide this for the
        # foreseeable future.
        matrix_dictionary[self.vector_source_matrix_key] = sps.csc_matrix(
            (g.num_faces, g.num_cells * g.dim)
        )
        matrix_dictionary[
            self.bound_pressure_vector_source_matrix_key
        ] = sps.csc_matrix((g.num_faces, g.num_cells * g.dim))

    def _interaction_regions(self, g):
        raise NotImplementedError

    def _add_network_to_upscale(self, reg, network):
        """
        Add the fractures to the local interation region depending on the spatial dimension
        """
        if isinstance(network, pp.FractureNetwork2d) or isinstance(
            network, pp.FractureNetwork3d
        ):
            reg.add_network(network)
        else:
            raise ValueError

    def _discretize_interaction_region(
        self, g, micro_network, parameter_dictionary, local_mesh_args, reg
    ):

        # Add the fractures to be upscaled
        self._add_network_to_upscale(reg, micro_network)

        # construct the sequence of local grid buckets
        gb_set = LocalGridBucketSet(g.dim, reg)
        gb_set.construct_local_buckets()

        # First basis functions for local problems
        (
            basis_functions,
            cc_assembler,  # Assembler for local Nd problem, one per coarse cell center
            cc_bc_values,
            full_assembler_map,  # Full hierarchy of (Nd, .., 0) assemblers
        ) = local_problems.cell_basis_functions(reg, gb_set, self, parameter_dictionary)

        # Call method to transfer basis functions to transmissibilties over coarse
        # edges
        trm_cell = local_problems.compute_transmissibilies(
            reg,
            gb_set,
            basis_functions,
            cc_assembler,
            cc_bc_values,
            g,
            self,
            parameter_dictionary,
        )

        matrix_bound_pressure_cell = (
            local_problems.discretize_pressure_trace_macro_bound(
                g, gb_set, self, cc_assembler, basis_functions
            )
        )

        (
            trm_boundary,
            matrix_bound_pressure_face,
        ) = local_problems.discretize_boundary_conditions(
            reg, gb_set, self, parameter_dictionary, g, full_assembler_map
        )
        # For debugging purposes e have kept the mesh files for this region up to this point
        # but now it should be okay to delete it
        reg.cleanup()

        return (
            trm_cell,
            trm_boundary,
            matrix_bound_pressure_cell,
            matrix_bound_pressure_face,
        )


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
