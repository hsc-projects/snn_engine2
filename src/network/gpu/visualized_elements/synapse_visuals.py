import numpy as np
import torch

from rendering import RenderedCudaObjectNode, CudaLine


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
                                       # color=(0, 0, 0, 1),
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

    def __init__(self, scene, view, device, neurons, synapses):

        self._scene = scene
        self._view = view
        self._device = device
        self._neurons = neurons
        self._synapses = synapses

        self._dct = {}

    def add_synapse_visual(self, neuron_id, outgoing=True):

        if outgoing is not True:
            raise NotImplementedError('outgoing=False')

        if not isinstance(neuron_id, int):
            raise TypeError('not isinstance(neuron_id, int)')

        if (neuron_id in self._dct) and (self._dct[neuron_id] is not None):
            raise ValueError(f'Synapses of {neuron_id} are already visualized.')

        src_pos = self._neurons.pos[neuron_id].cpu()
        snk_pos = self._neurons.pos[self._synapses.N_rep[neuron_id].to(dtype=torch.int64, device='cpu')].cpu()

        self._dct[neuron_id] = SynapseVisual(
            scene=self._scene,
            view=self._view,
            device=self._device,
            src_pos=src_pos,
            snk_pos=snk_pos
        )

    def refresh_synapse_visual(self, neuron_id):
        self.remove_synapse_visual(neuron_id)
        self.add_synapse_visual(neuron_id)

    def remove_synapse_visual(self, neuron_id):
        self._dct[neuron_id].unregister_registered_buffers()
        self._dct[neuron_id].parent = None
        self._dct[neuron_id] = None

    def unregister_registered_buffers(self):
        for v in self._dct.values():
            if v is not None:
                v.unregister_registered_buffers()
