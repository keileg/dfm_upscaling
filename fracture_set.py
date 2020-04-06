"""
Module contains data structure for handling a global set of fractures.
"""
import numpy as np
import porepy as pp

def split_network(network, selection_criterion, logical_op=np.logical_or, **kwargs):
    """ Construct the network for the macro and micro fractures.
    No modification on the geometry of the fractures is done.
    If the selection_criterion is a vector a logical operator is applied.
    """

    # select the fractures that are micro
    selection_criterion = np.atleast_1d(selection_criterion)
    micro = logical_op.reduce([sc(network, **kwargs) for sc in selection_criterion])
    # what is not micro is macro
    macro = np.logical_not(micro)

    if _is_3d(network):
        # construct the macro fracture network
        macro_fractures = network._fractures[macro]
        macro_network = pp.FractureNetwork3d(macro_fractures, network.domain, network.tol)

        # construct the micro fracture network
        micro_fractures = network._fractures[micro]
        micro_network = pp.FractureNetwork3d(micro_fractures, network.domain, network.tol)

    elif _is_2d(network):
        # construct the macro fracture network
        macro_edges = network.edges[:, macro]
        macro_network = pp.FractureNetwork2d(network.pts, macro_edges, network.domain, network.tol)

        # construct the micro fracture network
        micro_edges = network.edges[:, micro]
        micro_network = pp.FractureNetwork2d(network.pts, micro_edges, network.domain, network.tol)

    else:
        raise ValueError

    return macro_network, micro_network

class Criterion(object):
    """ Static class that stores different possible criteria to select fractures.
    """
    @staticmethod
    def none(network, **kwargs):
        """ Return none of the fractures as micro"""
        if _is_3d(network):
            num_fracs = np.asarray(network._fractures).size

        elif _is_2d(network):
            num_fracs = network.edges.shape[1]

        else:
            raise ValueError

        return np.zeros(num_fracs, dtype=np.bool)

    @staticmethod
    def every(network, **kwargs):
        """ Return all the fractures as micro """
        return np.logical_not(Criterion.none(network, **kwargs))

    @staticmethod
    def smaller_than(network, **kwargs):
        """ All the fractures strictly smaller than a level will be micro """
        if _is_3d(network):
            raise NotImplementedError

        elif _is_2d(network):
            length = network.length()

        else:
            raise ValueError

        return length < kwargs["branch_length"]

    @staticmethod
    def not_smaller_then(network, **kwargs):
        """ All the fractures not strictly smaller than a level will be micro """
        return np.logical_not(Criterion.smaller_than(network, **kwargs))

    @staticmethod
    def isolated(network, **kwargs):
        """ All the fractures that are isolated are micro.
        A fracture is isolated if both point of the associated edges are unique.
        NOTE: no intersection between fractures is computed
        """
        if _is_3d(network):
            raise NotImplementedError

        elif _is_2d(network):
            edges = network.edges
            pts_id = edges.ravel()
            # check if a point is unique
            unique_pts = np.flatnonzero(np.bincount(pts_id) == 1)
            # check if for each edge point is unique
            edges_unique_pts = np.isin(edges, unique_pts)
            # define if an edge is isolated by having both point as unique
            fractures = np.logical_and.reduce(edges_unique_pts)

        else:
            raise ValueError

        return fractures

    @staticmethod
    def not_isolated_branch(network, **kwargs):
        return np.logical_not(self.isolated_branch(network, **kwargs))

def _is_3d(network):
    return isinstance(network, pp.FractureNetwork3d)

def _is_2d(network):
    return isinstance(network, pp.FractureNetwork2d)
