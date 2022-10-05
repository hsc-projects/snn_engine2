import torch

from network.network_structures import NeuronTypes


def sensory_group(n_sensory_neurons, sensory_neuron_ids,
                  filter_neuron_ids_exc, filter_neuron_ids_inh,
                  group, N_flags, N_pos, G_pos, grid):

    N_flags.b_sensory_input[sensory_neuron_ids] = 1
    ux = grid.unit_shape[0]
    if n_sensory_neurons > 1:
        new_x_coords = torch.linspace(G_pos.tensor[group][0] + ux * 0.1,
                                      G_pos.tensor[group][0] + ux * 0.9,
                                      steps=n_sensory_neurons)
    else:
        new_x_coords = G_pos.tensor[group][0] + ux * 0.5
    new_y_coords = G_pos.tensor[group][1]
    new_z_coords = G_pos.tensor[group][2] + (grid.unit_shape[2] / 2)
    N_pos.tensor[sensory_neuron_ids, 0] = new_x_coords
    N_pos.tensor[sensory_neuron_ids, 1] = new_y_coords
    N_pos.tensor[sensory_neuron_ids, 2] = new_z_coords
    N_pos.tensor[sensory_neuron_ids, 10] = 1

    uy = grid.unit_shape[1]
    max_y = G_pos.tensor[group][1] + uy
    # self.N_pos.tensor[filter_neuron_ids_exc, 10] = 1
    N_pos.tensor[filter_neuron_ids_exc, 1] += .33 * uy * (
            (max_y - N_pos.tensor[filter_neuron_ids_exc, 1]) / max_y)
    N_pos.tensor[filter_neuron_ids_inh, 1] += .33 * uy * (
            (max_y - N_pos.tensor[filter_neuron_ids_inh, 1]) / max_y)