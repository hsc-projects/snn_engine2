from vispy import gloo
import sys

from engine import Engine

from configured_networks.io_network0 import IOSnn0Config
from configured_networks.rate_network0 import RateNetwork0Config
from configured_networks.network0 import Network0Config


# high priority

# TODO: simulate chemical diffusion
# TODO: build docs to ./docs instead of ./docs/html

# medium priority

# TODO: 1. minimalistic network; 2. rate network
# TODO: in-engine configuration/construction
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

# # ui

# TODO: cpp_cuda_backend side group_info_mesh face color actualization
# TODO: group selection via selector box
# TODO: group selection via click


if __name__ == '__main__':

    gloo.gl.use_gl('gl+')
    # eng = Engine(IOSnn0Config())
    # eng = Engine(RateNetwork0Config())
    eng = Engine(Network0Config())
    if sys.flags.interactive != 1:
        eng.run()
