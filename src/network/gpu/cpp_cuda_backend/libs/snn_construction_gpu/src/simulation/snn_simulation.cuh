#include <utils/curand_states.cuh>
#include <utils/launch_parameters.cuh>


struct SnnSimulation
{
    int N;
    int G;
    int S;
    int D;
    int T;

    int n_voltage_plots;
    int voltage_plot_length;
    float* voltage_plot_data;
    int* voltage_plot_map;
    bool b_update_voltage_plot = true;
    
    int n_scatter_plots;
    int scatter_plot_length;
    float* scatter_plot_data;
    int* scatter_plot_map;

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
    float* N_states; 
    
    float* N_weights;
    
    float* fired; 
    int* last_fired; 
    
    float* firing_times_write;
    float* firing_times_read;
    float* firing_times;

    int* firing_idcs_write;
    int* firing_idcs_read;
    int* firing_idcs;
    
    int* firing_counts_write;
    int* firing_counts;
    int* G_firing_count_hist;

    bool stdp_active = false;
    int* G_stdp_config0;
    int* G_stdp_config1;
    int* G_stdp_config_current;

    float* G_avg_weight_inh;
    float* G_avg_weight_exc;
    int* G_syn_count_inh;
    int* G_syn_count_exc;
    
    LaunchParameters lp_update_state;
    LaunchParameters lp_update_voltage_plot;
    LaunchParameters lp_update_scatter_plot;

    cusparseHandle_t fired_handle;
	
	cusparseSpMatDescr_t firing_times_sparse;
	cusparseDnMatDescr_t firing_times_dense;

	void* fired_buffer{nullptr};
	
	int n_fired = 0;
	size_t fired_buffer_size = 0;
	int n_fired_total = 0;
	int n_fired_total_m1 = 0;
	int n_fired_0 = 0;
	int n_fired_m1 = 0;
		
	int firing_counts_idx = 1;
	int firing_counts_idx_m1 = 1;
	// int firing_counts_idx_end = 1;

	int reset_firing_times_ptr_threshold;
	int reset_firing_count_idx_threshold;
	int n_fired_m1_to_end = 0;

    int t = 0;

    int no_stdp_time_threshold = 20;

    bool resetting = false;

    std::chrono::steady_clock::time_point t0;
    // std::chrono::steady_clock::time_point t1;
    uint update_duration;

    int* L_winner_take_all_map; 
    int max_n_winner_take_all_layers;
    int max_winner_take_all_layer_size;

    int chem_grid_w;
    int chem_grid_h;
    int chem_grid_d;
    float* C_old; 
    float* C_new; 
    float* C_source; 
    const int chem_block_size = 8;
    float chem_k_val = 0.75f; 
	float chem_depreciation = 0.1f;
    LaunchParameters lp_update_chemical_contrations;
    bool b_update_chemical_contrations = false;


    SnnSimulation(
        int N_,
        int G_,
        int S_,
        int D_,
        int T,

        int n_voltage_plots_,
        int voltage_plot_length_,
        float* voltage_plot_data_,
        int* voltage_plot_map_,

        int n_scatter_plots_,
        int scatter_plot_length_,
        float* scatter_plot_data_,
        int* scatter_plot_map_,
        
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
        float* N_states_,
        float* N_weights_,
        float* fired_,
        int* last_fired_,
        float* firing_times_,
        int* firing_idcs_,
        int* firing_counts_,
        int* G_firing_count_hist_,
        
        int* G_stdp_config0_,
        int* G_stdp_config1_,
        float* G_avg_weight_inh_,
        float* G_avg_weight_exc_,
        int* G_syn_count_inh_,
        int* G_syn_count_exc_,
        int* L_winner_take_all_map_,
        int max_n_winner_take_all_layers_,
        int max_winner_take_all_layer_size_,

        float* C_old_,
        float* C_new_, 
        float* C_source_, 
        int chem_grid_w_,
        int chem_grid_h_,
        int chem_grid_d_,
        float chem_k_val_, 
        float chem_depreciation_
    );



    
    void update_plots();
    void update_chemical_contrations();
    void print_info(bool bprint_idcs = false, bool bprint_nfiring_times = false);
    void update(bool b_stdp, bool verbose);


    void _update_sim_pointers();

    void set_stdp_config(int stdp_config_id, bool activate = true);


    // void actualize_N_rep_pre_synaptic();

    void calculate_avg_group_weight();

    // void remove_all_synapses_to_group(int group);
    // void nullify_all_weights_to_group(int group);

    void set_b_update_chemical_contrations(bool b_update_chemical_contrations_);
    bool get_b_update_chemical_contrations();

};