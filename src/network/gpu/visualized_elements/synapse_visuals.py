import numpy as np
import torch

from rendering import RenderedCudaObjectNode, CudaLine
from network.gpu.neurons import NeuronRepresentation


# noinspection PyAbstractClass
class SynapseVisual(RenderedCudaObjectNode):

    def __init__(self,
                 scene,
                 view,
                 src_pos,
                 snk_pos,
                 device):

        scene.set_current()

        src_pos = np.reshape(src_pos, (1, 3))

        pos = np.vstack([src_pos, snk_pos])

        n_sinks = len(snk_pos)

        connect = np.zeros((n_sinks, 2))
        connect[:, 1] = np.arange(n_sinks)

        self.line: CudaLine = CudaLine(pos=pos,
                                       color=(1, 1, 1, 0.18),
                                       connect=connect,
                                       antialias=False, width=1, parent=None)

        RenderedCudaObjectNode.__init__(self, [self.line])

        view.add(self)
        scene._draw_scene()

        self.init_cuda_attributes(device=device)

    def init_cuda_attributes(self, device):
        super().init_cuda_attributes(device)
        self.registered_buffers += self.line.registered_buffers


class VisualizedSynapsesCollection:

    def __init__(self, scene, view, device,
                 neurons: NeuronRepresentation,
                 synapses):

        self._scene = scene
        self._view = view
        self._device = device
        self._neurons = neurons
        self._synapses = synapses

        self._dct = {}
        self._reference_counts = {}

    def add_synapse_visual(self, neuron_id, outgoing=True):

        if outgoing is not True:
            raise NotImplementedError('outgoing=False')

        if not isinstance(neuron_id, int):
            raise TypeError('not isinstance(neuron_id, int)')

        if self.b_slot_taken(neuron_id):
            self._reference_counts[neuron_id] += 1
            self._dct[neuron_id].visible = True
            return self._dct[neuron_id]
            # raise ValueError(f'Synapses of {neuron_id} are already visualized.')

        src_pos = self._neurons.pos[neuron_id].cpu()

        snk_ids = self._synapses.N_rep[:, neuron_id].to(dtype=torch.int64)
        snk_ids = snk_ids[snk_ids >= 0]

        snk_pos = self._neurons.pos[snk_ids].cpu()

        self._reference_counts[neuron_id] = 1
        self._dct[neuron_id] = SynapseVisual(
            scene=self._scene,
            view=self._view,
            device=self._device,
            src_pos=src_pos,
            snk_pos=snk_pos
        )
        return self._dct[neuron_id]

    def b_slot_taken(self, neuron_id):
        return (neuron_id in self._dct) and (self._dct[neuron_id] is not None)

    def move_visual(self, prev_neuron_id, new_neuron_id):
        if prev_neuron_id is not None:
            self._reference_counts[prev_neuron_id] -= 1
            if self.b_slot_taken(new_neuron_id):
                if self._reference_counts[prev_neuron_id] <= 0:
                    self._dct[prev_neuron_id].visible = False
                self._reference_counts[new_neuron_id] += 1
                if self._reference_counts[new_neuron_id] == 1:
                    self._dct[new_neuron_id].visible = True
                return self._dct[new_neuron_id]
            elif self._reference_counts[prev_neuron_id] <= 0:
                self._reference_counts[new_neuron_id] = 1
                visual = self._dct[prev_neuron_id]
                self._dct[prev_neuron_id] = None
                self._dct[new_neuron_id] = visual
                return self.refresh_synapse_visual(new_neuron_id)
            else:
                return self.add_synapse_visual(new_neuron_id)
        else:
            return self.add_synapse_visual(new_neuron_id)

    def refresh_synapse_visual(self, neuron_id):
        visual: SynapseVisual = self._dct[neuron_id]
        snk_ids = self._synapses.N_rep[:, neuron_id].to(dtype=torch.int64)
        snk_ids = snk_ids[snk_ids >= 0]
        if len(snk_ids) != len(visual.line.pos_gpu.tensor):
            self.remove_synapse_visual(neuron_id)
            self.add_synapse_visual(neuron_id)
        else:
            src_pos = self._neurons.pos[neuron_id]
            snk_pos = self._neurons.pos[snk_ids]
            visual.line.pos_gpu.tensor[:] = torch.vstack([src_pos, snk_pos])
        visual.visible = True
        return visual

    def remove_synapse_visual(self, neuron_id):
        self._reference_counts[neuron_id] = 0
        self._dct[neuron_id].unregister_registered_buffers()
        self._dct[neuron_id].parent = None
        self._dct[neuron_id] = None

    def unregister_registered_buffers(self):
        for v in self._dct.values():
            if v is not None:
                v.unregister_registered_buffers()
