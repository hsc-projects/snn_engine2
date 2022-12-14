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


void fill_N_flags_group_id_and_G_neuron_count_per_type_python(
    const int N, 
    const int G, 
    const long N_pos_dp,
    const py::tuple& N_pos_shape,
    long N_flags_dp,
    const py::tuple& G_shape,
    long G_neuron_counts_dp,
    const int N_pos_n_cols,
    const int N_flags_row_type,
	const int N_flags_row_group
){
    const float* N_pos = reinterpret_cast<float*> (N_pos_dp);
	int* N_flags = reinterpret_cast<int*> (N_flags_dp);
	int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);

    fill_N_flags_group_id_and_G_neuron_count_per_type(
        N, 
        G, 
        N_pos, 
        N_pos_shape[0].cast<int>(), N_pos_shape[1].cast<int>(), N_pos_shape[2].cast<int>(),
        N_flags, 
        G_neuron_counts,
        G_shape[0].cast<int>(), G_shape[1].cast<int>(), G_shape[2].cast<int>(),
        N_pos_n_cols,
        N_flags_row_type,
        N_flags_row_group
    );
}


void fill_G_neuron_count_per_delay_python(
	const int S,
	const int D,
	const int G,
	const long G_delay_distance_dp,
	long G_neuron_counts_dp
){
    
    const int* G_delay_distance = reinterpret_cast<int*> (G_delay_distance_dp);
	int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);

    fill_G_neuron_count_per_delay(
        S,
        D,
        G,
        G_delay_distance,
        G_neuron_counts
    );
}



void fill_G_exp_ccsyn_per_src_type_and_delay_python(
	const int S,
	const int D,
	const int G,
    const long G_neuron_counts_dp,
    long G_conn_probs_dp,
	long G_exp_ccsyn_per_src_type_and_delay_dp
){
    
	const int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);
	float* G_conn_probs = reinterpret_cast<float*> (G_conn_probs_dp);
	int* G_exp_ccsyn_per_src_type_and_delay = reinterpret_cast<int*> (G_exp_ccsyn_per_src_type_and_delay_dp);

    fill_G_exp_ccsyn_per_src_type_and_delay(
        S,
        D,
        G,
        G_neuron_counts,
        G_conn_probs,
        G_exp_ccsyn_per_src_type_and_delay
    );
}



void sort_N_rep_python(
	const int N,
	const int S,
	long sort_keys_dp,
	long N_rep_dp	
){
    int* sort_keys = reinterpret_cast<int*> (sort_keys_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    sort_N_rep(N, S, sort_keys, N_rep);
}

void reindex_N_rep_python(
	const int N,
	const int S,
	const int D,
	const int G,
	const long N_flags_dp,
	const long cc_src_dp,
	const long cc_snk_dp,
	const long G_rep_dp,
	const long G_neuron_counts_dp,
	const long G_group_delay_counts_dp,
    const py::tuple& gc_location,
    const py::tuple& gc_conn_shape,
	long cc_syn_dp,
	long N_delays_dp,
    long sort_keys_dp,
	long N_rep_dp,
    const int N_flags_row_group = 2,
	bool verbose = 0
)
{
    const int* N_flags = reinterpret_cast<int*> (N_flags_dp);
    const int* cc_src = reinterpret_cast<int*> (cc_src_dp);
    const int* cc_snk = reinterpret_cast<int*> (cc_snk_dp);
    const int* G_rep = reinterpret_cast<int*> (G_rep_dp);
    const int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);
    const int* G_group_delay_counts = reinterpret_cast<int*> (G_group_delay_counts_dp);
    int* cc_syn = reinterpret_cast<int*> (cc_syn_dp);
    int* N_delays = reinterpret_cast<int*> (N_delays_dp);
    int* sort_keys = reinterpret_cast<int*> (sort_keys_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    reindex_N_rep(
        N, 
        S, 
        D, 
        G,
        N_flags,
        cc_src, 
        cc_snk,
        G_rep, 
        G_neuron_counts, 
        G_group_delay_counts,
        gc_location[0].cast<int>(), gc_location[1].cast<int>(),
        gc_conn_shape[0].cast<int>(), gc_conn_shape[1].cast<int>(),
        cc_syn,
        N_delays,
        sort_keys,
        N_rep,
        N_flags_row_group,
        verbose
    );
}

void fill_N_rep_groups_python(
	int N,
	int S,
	const long N_flags_dp,
	const long N_rep_dp,
	const long N_rep_groups_dp,
    int N_flags_row_group
){
    const int* N_flags = reinterpret_cast<int*> (N_flags_dp);
    const int* N_rep = reinterpret_cast<int*> (N_rep_dp);
    int* N_rep_groups = reinterpret_cast<int*> (N_rep_groups_dp);

    fill_N_rep_groups(
		N,
		S,
		N_flags,
		N_rep,
		N_rep_groups,
		N_flags_row_group
    );
}




PYBIND11_MODULE(snn_construction_gpu, m)
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
    m.def("fill_N_flags_group_id_and_G_neuron_count_per_type", 
          &fill_N_flags_group_id_and_G_neuron_count_per_type_python, 
          py::arg("N"),
          py::arg("G"),
          py::arg("N_pos"),
          py::arg("N_pos_shape"),
          py::arg("N_flags"),
          py::arg("G_shape"),
          py::arg("G_neuron_counts"),
          py::arg("N_pos_n_cols") = 13, 
          py::arg("N_flags_row_type") = 1,
          py::arg("N_flags_row_group") = 2
    );
    
    m.def("fill_G_neuron_count_per_delay", 
          &fill_G_neuron_count_per_delay_python, 
          py::arg("S"),
          py::arg("D"),
          py::arg("G"),
          py::arg("G_delay_distance"),
          py::arg("G_neuron_counts")
    );

    m.def("fill_G_exp_ccsyn_per_src_type_and_delay", 
          &fill_G_exp_ccsyn_per_src_type_and_delay_python, 
          py::arg("S"),
          py::arg("D"),
          py::arg("G"),
          py::arg("G_neuron_counts"),
          py::arg("G_conn_probs"),
          py::arg("G_exp_ccsyn_per_src_type_and_delay")
    );

    m.def("sort_N_rep", 
          &sort_N_rep_python, 
          py::arg("N"),
          py::arg("S"),
          py::arg("sort_keys"),
          py::arg("N_rep"));
    
    m.def("reindex_N_rep", 
          &reindex_N_rep_python, 
          py::arg("N"),
          py::arg("S"),
          py::arg("D"),
          py::arg("G"),
          py::arg("N_flags"),
          py::arg("cc_src"),
          py::arg("cc_snk"),
          py::arg("G_rep"),
          py::arg("G_neuron_counts"),
          py::arg("G_group_delay_counts"),
          py::arg("gc_location"),
          py::arg("gc_conn_shape"),
          py::arg("cc_syn"),
          py::arg("N_delays"),
          py::arg("sort_keys"),
          py::arg("N_rep"),
          py::arg("N_flags_row_group") = 2,
          py::arg("verbose") = false);

    m.def("fill_N_rep_groups", 
          &fill_N_rep_groups_python, 
          py::arg("N"),
          py::arg("S"),
          py::arg("N_flags"),
          py::arg("N_rep"),
          py::arg("N_rep_groups"),
          py::arg("N_flags_row_group")
    );

    py::class_<CuRandStates, std::shared_ptr<CuRandStates>>(m, "CuRandStates_") //, py::dynamic_attr())
    .def(py::init<int>())
    .def_readonly("n_states", &CuRandStates::n_states)
    .def_readonly("states", &CuRandStates::states)
    .def("__repr__",
        [](const CuRandStates &cs) {
            return "CuRandStates_(" + std::to_string(cs.n_states) + ")";
        }
    );

    py::class_<CuRandStatesPointer>(m, "CuRandStates") //, py::dynamic_attr())
    .def(py::init<int>())
    .def_property_readonly("n_states", &CuRandStatesPointer::n_states)
    .def("ptr", &CuRandStatesPointer::ptr)
    .def("__repr__",
        [](const CuRandStatesPointer &cs) {
            return "CuRandStates(" + std::to_string(cs.ptr_->n_states) + ")";
        }
    );
}