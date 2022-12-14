#include <utils/curand_states.cuh>
#include <utils/launch_parameters.cuh>

#include <pybind11/include/pybind11/pybind11.h>
namespace py = pybind11;


void fill_N_flags_group_id_and_G_neuron_count_per_type(
    int N, 
    int G, 
    const float* N_pos,
    int N_pos_shape_x, int N_pos_shape_y, int N_pos_shape_z,
    int* N_flags,
    int* G_neuron_counts,
    int G_shape_x, int G_shape_y, int G_shape_z,
    int N_pos_n_cols = 13,
    int N_flags_row_type = 0,
    int N_flags_row_group = 1
);


void fill_G_neuron_count_per_delay(
	int S,
	int D,
	int G,
	const int* G_delay_distance,
	int* G_neuron_counts
);


void fill_G_exp_ccsyn_per_src_type_and_delay(
	int S,
	int D,
	int G,
	const int* G_neuron_counts,
	float* G_conn_probs,
    int* G_exp_ccsyn_per_src_type_and_delay,
	bool verbose = 0
);

void sort_N_rep(
	int N,
	int S,
	int* sort_keys,
	int* N_rep,
	bool vebose = true
);


void reindex_N_rep(
	int N,
	int S,
	int D,
	int G,
	const int* N_flags,
	const int* cc_src,
	const int* cc_snk,
	const int* G_rep,
	const int* G_neuron_counts,
	const int* G_group_delay_counts,
	int gc_location0,
	int gc_location1,
	int gc_conn_shape0,
	int gc_conn_shape1,
	const int* cc_syn,
	int* N_delays,
	int* sort_keys,
	int* N_rep,
	const int N_flags_row_group,
	bool verbose
);


void fill_N_rep_groups(
	int N,
	int S,
	const int* N_flags,
	const int* N_rep,
	int* N_rep_groups,
    int N_flags_row_group = 2
);

struct SnnRepresentation
{
    int N;
    int G;
    int S;
    int D;

    curandState* rand_states;
    
    float* N_pos; 
    int* G_group_delay_counts; 
    int* G_flags; 
    float* G_props; 
    int* N_rep; 
    int* N_rep_buffer; 
    int* N_rep_pre_synaptic; 
    int* N_rep_pre_synaptic_idcs; 
    int* N_rep_pre_synaptic_counts; 
    int* N_delays; 
    
    int* N_flags; 
    float* N_weights;
    
    int* L_winner_take_all_map; 
    int max_n_winner_take_all_layers;
    int max_winner_take_all_layer_size;

    std::chrono::steady_clock::time_point t0;

    SnnRepresentation(
        int N_,
        int G_,
        int S_,
        int D_,

        curandState* rand_states_,

        float* N_pos_,
        int* G_group_delay_counts_,
        int* G_flags_, 
        float* G_props_, 
        int* N_rep_, 
        int* N_rep_buffer_,
        int* N_rep_pre_synaptic_, 
        int* N_rep_pre_synaptic_idcs_, 
        int* N_rep_pre_synaptic_counts_, 
        int* N_delays_, 

        int* N_flags_,
        float* N_weights_,
        int* L_winner_take_all_map_,
        int max_n_winner_take_all_layers_,
        int max_winner_take_all_layer_size_
    );

    void fill_N_rep(
        const int* cc_src,
        const int* cc_snk,
        const int* G_rep,
        const int* G_neuron_counts,
        int* G_autapse_indices,
        int* G_relative_autapse_indices,
        bool has_autapses,
        int gc_location0,
        int gc_location1,
        int gc_conn_shape0,
        int gc_conn_shape1,
        const int* cc_syn,
        int* sort_keys,
        const int N_flags_row_group,
        bool verbose
    );

    void fill_N_rep_python(
        const long cc_src_dp,
        const long cc_snk_dp,
        const long G_rep_dp,
        const long G_neuron_counts_dp,
        long G_autapse_indices_dp,
        long G_relative_autapse_indices_dp,
        bool has_autapses,
        const py::tuple& gc_location,
        const py::tuple& gc_conn_shape,
        long cc_syn_dp,
        long sort_keys_dp,
        const int N_flags_row_group,
        bool verbose
    );
    

    void swap_groups(
        long* neurons, int n_neurons, 
        long* groups, int n_groups, 
        int* neuron_group_indices,
        int* G_swap_tensor, const int G_swap_tensor_shape_1,
        float* swap_rates_inh, float* swap_rates_exc,
        int* group_neuron_counts_inh, int* group_neuron_counts_exc, int* group_neuron_counts_total,
        int* G_delay_distance,
        int* N_relative_G_indices, int* G_neuron_typed_ccount,
        int* neuron_group_counts,
        int print_idx
    );
    void swap_groups_python(
        long neurons, int n_neurons, 
        long groups, int n_groups, 
        long neuron_group_indices,
        long G_swap_tensor, const int G_swap_tensor_shape_1,
        long swap_rates_inh, long swap_rates_exc,
        long group_neuron_counts_inh, long group_neuron_counts_exc, long group_neuron_counts_total, 
        long G_delay_distance,
        long N_relative_G_indices, long G_neuron_typed_ccount,
        long neuron_group_counts,
        int print_idx
    );


    void actualize_N_rep_pre_synaptic();

    void remove_all_synapses_to_group(int group);
    void nullify_all_weights_to_group(int group);

};