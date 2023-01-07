from typing import Optional

from engine.content.widgets.collapsible_widget.collapsible_widget import CollapsibleWidget
from engine.content.neuron_properties_collapsible import NeuronIDFrame

from network import SpikingNeuralNetwork
from network.gpu.visualized_elements.synapse_visuals import VisualizedSynapsesCollection, SynapseVisual


class SynapseCollapsible(CollapsibleWidget):

    def __init__(self, parent, network: SpikingNeuralNetwork,
                 title, neuron_id):

        super().__init__(parent=parent, title=title)

        self.id_frame = NeuronIDFrame(self, network.network_config.N)
        self.id_frame.spinbox.setValue(neuron_id)

        self.id_frame.spinbox.valueChanged.connect(self.update_neuron_id)

        self._last_id = neuron_id
        self.visual_collection: Optional[VisualizedSynapsesCollection] = network.synapse_arrays.visualized_synapses

        self.add(self.id_frame)

        self.visual_collection.add_synapse_visual(neuron_id=neuron_id)

    def update_neuron_id(self):
        new_id = self.id_frame.spinbox.value()
        self.visual_collection.move_visual(prev_neuron_id=self._last_id, new_neuron_id=new_id)
        self._last_id = new_id

    @property
    def visual(self):
        return self.visual_collection._dct[self.id_frame.spinbox.value()]


class SynapseCollapsibleContainer(CollapsibleWidget):

    def __init__(self, title='Visualized Synapses', parent=None):
        CollapsibleWidget.__init__(self, title=title, parent=parent)

        self.interfaced_synapses_dct = {}

    def add_interfaced_synapse(self, network: SpikingNeuralNetwork, neuron_id: int, title=None):

        index = len(self.interfaced_synapses_list)

        if title is None:
            title = 'VisualizedSynapses' + str(index)

        syn_collapsible = SynapseCollapsible(self, network=network, title=title, neuron_id=neuron_id)

        self.interfaced_synapses_dct[title] = syn_collapsible

        self.add(syn_collapsible)
        self.toggle_collapsed()
        self.toggle_collapsed()

        return syn_collapsible

    @property
    def interfaced_synapses_list(self):
        return list(self.interfaced_synapses_dct.values())
