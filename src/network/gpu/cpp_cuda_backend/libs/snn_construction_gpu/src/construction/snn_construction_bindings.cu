#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/numpy.h>
#include <pybind11/include/pybind11/stl.h>

#include <construction/snn_representation.cuh>


namespace py = pybind11;


SnnRepresentation make_SnnRepresentation(
    const int N,
    const int G,
    const int S,
    const int D,
    std::shared_ptr<CuRandStates> curand_states,
    const long N_pos_dp,
    const long G_group_delay_counts_dp, 
    const long G_flags_dp, 
    const long G_props_dp, 
    const long N_rep_dp, 
    const long N_rep_buffer_dp,
    const long N_rep_pre_synaptic_dp,
    const long N_rep_pre_synaptic_idcs_dp,
    const long N_rep_pre_synaptic_counts_dp,
    const long N_delays_dp, 
    const long N_flags_dp,
    const long N_weights_dp,
    const long L_winner_take_all_map_dp,
    const int max_n_winner_take_all_layers,
    const int max_winner_take_all_layer_size
){

    float* N_pos = reinterpret_cast<float*> (N_pos_dp);
    int* G_group_delay_counts = reinterpret_cast<int*> (G_group_delay_counts_dp);
    int* G_flags = reinterpret_cast<int*> (G_flags_dp);
    float* G_props = reinterpret_cast<float*> (G_props_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    int* N_rep_buffer = reinterpret_cast<int*> (N_rep_buffer_dp);
    int* N_rep_pre_synaptic = reinterpret_cast<int*> (N_rep_pre_synaptic_dp);
    int* N_rep_pre_synaptic_idcs = reinterpret_cast<int*> (N_rep_pre_synaptic_idcs_dp);
    int* N_rep_pre_synaptic_counts = reinterpret_cast<int*> (N_rep_pre_synaptic_counts_dp);
    
    int* N_delays = reinterpret_cast<int*> (N_delays_dp);

    int* N_flags = reinterpret_cast<int*> (N_flags_dp);
    
    float* N_weights = reinterpret_cast<float*> (N_weights_dp);
    int* L_winner_take_all_map = reinterpret_cast<int*> (L_winner_take_all_map_dp);

    
    return SnnRepresentation(
        N,
        G,
        S,
        D,
        curand_states->states,
        N_pos,
        G_group_delay_counts,
        G_flags,
        G_props, 
        N_rep, 
        N_rep_buffer,
        N_rep_pre_synaptic, 
        N_rep_pre_synaptic_idcs, 
        N_rep_pre_synaptic_counts, 
        N_delays, 
        N_flags, 
        N_weights,
        L_winner_take_all_map,
        max_n_winner_take_all_layers,
        max_winner_take_all_layer_size
    );
}


PYBIND11_MODULE(snn_construction2_gpu, m)
    {
        
    py::class_<SnnRepresentation, std::shared_ptr<SnnRepresentation>>(m, "SnnRepresentation_")
    .def_readonly("N", &SnnRepresentation::N)
    .def_readonly("G", &SnnRepresentation::G)
    .def_readonly("S", &SnnRepresentation::S)
    .def_readonly("D", &SnnRepresentation::D)
    .def("fill_N_rep", &SnnRepresentation::fill_N_rep_python, 
        py::arg("cc_src"),
        py::arg("cc_snk"),
        py::arg("G_rep"),
        py::arg("G_neuron_counts"),
        py::arg("G_autapse_indices"),
        py::arg("G_relative_autapse_indices"),
        py::arg("has_autapses"),
        py::arg("gc_location"),
        py::arg("gc_conn_shape"),
        py::arg("cc_syn"),
        py::arg("sort_keys"),
        py::arg("N_flags_row_group") = 2,
        py::arg("verbose") = false)
    .def("swap_groups", &SnnRepresentation::swap_groups_python)
    .def("actualize_N_rep_pre_synaptic", &SnnRepresentation::actualize_N_rep_pre_synaptic)
    .def("remove_synapses_to_group", &SnnRepresentation::remove_all_synapses_to_group, 
        py::arg("group"))
    .def("nullify_all_weights_to_group", &SnnRepresentation::nullify_all_weights_to_group, 
        py::arg("group"))
    .def("__repr__",
        [](const SnnRepresentation &rep) {
            return "SnnRepresentation(N=" + std::to_string(rep.N) + ")";
        });
    m.def("SnnRepresentation", &make_SnnRepresentation,
        py::arg("N"),
        py::arg("G"),
        py::arg("S"),
        py::arg("D"),
        py::arg("curand_states_p"),
        py::arg("N_pos"),
        py::arg("G_group_delay_counts"),
        py::arg("G_flags"),
        py::arg("G_props"),
        py::arg("N_rep"),
        py::arg("N_rep_buffer"),
        py::arg("N_rep_pre_synaptic"),
        py::arg("N_rep_pre_synaptic_idcs"),
        py::arg("N_rep_pre_synaptic_counts"),
        py::arg("N_delays"),
        py::arg("N_flags"),
        py::arg("N_weights"),
        py::arg("L_winner_take_all_map"),
        py::arg("max_n_winner_take_all_layers"),
        py::arg("max_winner_take_all_layer_size")
    );

}