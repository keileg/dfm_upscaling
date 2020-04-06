import numpy as np
from scipy.sparse.linalg import spsolve
import porepy as pp

from fv_dfm import Tpfa_DFM
from fracture_set import split_network, Criterion

class My_Tpfa_DFM(Tpfa_DFM):
    # I might use this guy to define the user-defined set_parameters* functions
    # for micro and macro fracture networks
    def __init__(self, keyword="flow"):
        super(Tpfa_DFM, self).__init__(keyword)

    def set_parameters_cell_basis(self, gb):
        pass

def write_network(file_name):
    content = ("FID,START_X,START_Y,END_X,END_Y\n"
               "0,0,0.1,1,0.9\n"
               #"0,0.25,0.25,0.75,0.25\n"
               #"1,0.5,0,0.5,0.25\n"
               #"2,0.5,0.25,0.5,0.35\n"
               #"3,0.35,0.75,0.65,0.75\n"
               #"4,0.5,1,0.5,0.65"
              )
    with open(file_name, "w") as out_file:
        out_file.write(content)

def main():
    # example in 2d
    file_name = "network.csv"
    write_network(file_name)

    # load the network and split it, we assume 2d
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    global_network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)
    global_network = global_network.split_intersections()

    # select a subsample of the fractures
    macro_network, micro_network = split_network(global_network, Criterion.every)
    # NOTE: the explicit network is empty so far

    # create the macroscopic grid
    mesh_size = 2
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}
    gb = macro_network.mesh(mesh_kwargs)

    # plot the suff
    pp.plot_fractures(micro_network.pts, micro_network.edges, micro_networks.domain)
    pp.plot_grid(gb, info="c", alpha=0)

    # construct the solver
    tpfa_dfm = Tpfa_DFM(micro_network)

    # set parameters and discretization variables
    tpfa_dfm.set_parameters(gb)
    tpfa_dfm.set_variables_discretizations(gb)

    # discretize with the assembler
    assembler = pp.Assembler(gb)
    assembler.discretize()

    A, b = assembler.assemble_matrix_rhs()

    # Solve and distribute
    x = spsolve(A, b)
    assembler.distribute_variable(x)

if __name__ == "__main__":
    main()
