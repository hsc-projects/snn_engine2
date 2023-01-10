from vispy import gloo
import sys

from engine import Engine

from configured_networks.io_network1 import IOSnn1Config
from configured_networks.rate_network0 import RateNetwork0Config
from configured_networks.defaultnetwork0 import DefaultNetwork0Config


# high priority

# TODO: Fix connection probability bug
# TODO: diffusion grid calculation
# TODO: in-engine network - configuration/construction

# medium priority

# TODO: C++/Cuda refactoring (remove snn_construction)
# TODO: 1. minimalistic network; 2. rate network
# TODO: -ax**2 + b connection probability
# TODO: ReLu-ANN
# TODO: configurable segmentation
# TODO: subgroups

# low priority

# # simulation

# TODO: Error Handling
# TODO: Docs
# TODO: pre-synaptic delays
# TODO: resonant cells/groups,
# TODO: better sensory weights,
# TODO: filter-cells,
# TODO: group_info_mesh face sizes
# TODO: better stdp G2G config
# TODO: monitor learning
# TODO: weird synapse counts
# TODO: low neuron count swaps
# TODO: performance hit above 300K neurons
# TODO: testing

# # ui

# TODO: cpp_cuda_backend side group_info_mesh face color actualization
# TODO: group selection via selector box
# TODO: group selection via click


if __name__ == '__main__':

    gloo.gl.use_gl('gl+')
    eng = Engine(IOSnn1Config())
    # eng = Engine(RateNetwork0Config())
    # eng = Engine(DefaultNetwork0Config())
    if sys.flags.interactive != 1:
        eng.run()
