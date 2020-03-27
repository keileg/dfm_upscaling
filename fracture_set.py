"""
Module contains data structure for handling a global set of fractures.
"""
import numpy as np
import porepy as pp


class GlobalFractureSet:
    """ Class for storage and treatment of multiscale fractures. 
    
    The class stores all fractures in a domain, and provides methods for dividing
    these into explicit and upscaled representation.
    
    """
    
    def __init__(self, dim, domain, pts=None, edges=None, fractures=None):
        self.dim = dim
        
        if self.dim == 2:
            self.pts = pts
            self.edges = edges
        else:
            self.fractures = fractures
            
    def explicit_fractures(self, selection_criterion):
        pass
    
    def upscaled_fractures(self, selection_criterion):
        pass
            
    def network(self, selection_criterion):
        
        if self.dim == 2:
            p, e = self.explicit_fractures(selection_criterion)
            network = pp.FractureNetwork2d(p, e, self.domain)
            network.impose_external_boundary()
            return network
        
        else:
            fracs = self.explicit_fractures(selection_criterion)
            network = pp.FractureNetwork3d(fracs)
            network.impose_external_boundary(self.domain)
            
            return network