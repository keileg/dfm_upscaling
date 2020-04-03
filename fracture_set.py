"""
Module contains data structure for handling a global set of fractures.
"""
import numpy as np
import porepy as pp

def split_network(network, selection_criterion, logical_op=np.logical_or, **kwargs):
    """ Construct the network for the explicit fractures and the one for the fractures that
    should be upscaled. No modification on the geometry of the fractures is done.
    If the selection_criterion is a vector a logical operator is applied.
    """

    # get the ids of fractures that will be explicitly considered,
    # the others are upscaled
    selection_criterion = np.atleast_1d(selection_criterion)
    upscaled = logical_op.reduce([sc(network, **kwargs) for sc in selection_criterion])
    explicit = np.logical_not(upscaled)

    if _is_3d(network):
        # construct the explicit fracture network
        explicit_fractures = network._fractures[explicit]
        explicit_network = pp.FractureNetwork3d(explicit_fractures, network.domain, network.tol)

        # construct the upscaled fracture network
        upscaled_fractures = network._fractures[upscaled]
        upscaled_network = pp.FractureNetwork3d(upscaled_fractures, network.domain, network.tol)

    elif _is_2d(network):
        # construct the explicit fracture network
        explicit_edges = network.edges[:, explicit]
        explicit_network = pp.FractureNetwork2d(network.pts, explicit_edges, network.domain, network.tol)

        # construct the upscaled fracture network
        upscaled_edges = network.edges[:, upscaled]
        upscaled_network = pp.FractureNetwork2d(network.pts, upscaled_edges, network.domain, network.tol)

    else:
        raise ValueError

    return explicit_network, upscaled_network

class Criterion(object):
    """ Static class that stores different possible criteria to select fractures.
    """
    @staticmethod
    def none(network, **kwargs):
        """ Return none of the fractures as upscaled"""
        if _is_3d(network):
            num_fracs = np.asarray(network._fractures).size

        elif _is_2d(network):
            num_fracs = network.edges.shape[1]

        else:
            raise ValueError

        return np.zeros(num_fracs, dtype=np.bool)

    @staticmethod
    def every(network, **kwargs):
        return np.logical_not(Criterion.none(network, **kwargs))

    @staticmethod
    def smaller_than(network, **kwargs):
        """ All the fractures strictly smaller than level will be upscaled """
        if _is_3d(network):
            pass

        elif _is_2d(network):
            length = network.length()

        else:
            raise ValueError

        return length < kwargs["branch_length"]

    @staticmethod
    def not_smaller_then(network, **kwargs):
        return np.logical_not(Criterion.smaller_than(network, **kwargs))

    @staticmethod
    def isolated(network, **kwargs):
        """ All the fractures that are isolated are upscaled.
        A fracture is isolated if both point of the associated edges are unique.
        NOTE: no intersection between fractures is computed
        """
        if _is_3d(network):
            pass

        elif _is_2d(network):
            edges = network.edges
            pts_id = edges.ravel()
            # check if a point is unique
            unique_pts = np.flatnonzero(np.bincount(pts_id) == 1)
            # check if for each edge point is unique
            edges_unique_pts = np.isin(edges, unique_pts)
            # define if an edge is isolated by having both point as unique
            upscaled_fractures = np.logical_and.reduce(edges_unique_pts)

        else:
            raise ValueError

        return upscaled_fractures

    @staticmethod
    def not_isolated_branch(network, **kwargs):
        return np.logical_not(self.isolated_branch(network, **kwargs))

def _is_3d(network):
    return isinstance(network, pp.FractureNetwork3d)

def _is_2d(network):
    return isinstance(network, pp.FractureNetwork2d)
