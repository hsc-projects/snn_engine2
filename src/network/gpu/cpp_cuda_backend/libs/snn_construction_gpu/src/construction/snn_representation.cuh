#include <utils/curand_states.cuh>
#include <utils/launch_parameters.cuh>


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