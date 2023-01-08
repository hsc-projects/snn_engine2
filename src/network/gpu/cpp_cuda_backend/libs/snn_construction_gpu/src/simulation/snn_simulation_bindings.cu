#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/numpy.h>
#include <pybind11/include/pybind11/stl.h>

#include <simulation/snn_simulation.cuh>


namespace py = pybind11;


SnnSimulation make_SnnSimulation(
    const int N,
    const int G,
    const int S,
    const int D,
    const int T,
    const int n_voltage_plots,
    const int voltage_plot_length,
    const long voltage_plot_data_dp,
    const long voltage_plot_map_dp,
    const int n_scatter_plots,
    const int scatter_plot_length,
    const long scatter_plot_data_dp,
    const long scatter_plot_map_dp,
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
    const long N_states_dp,
    const long N_weights_dp,
    const long fired_dp,
    const long last_fired_dp,
    const long firing_times_dp,
    const long firing_idcs_dp,
    const long firing_counts_dp,

    const long G_firing_count_hist_dp,
    const long G_stdp_config0_dp,
    const long G_stdp_config1_dp,
    const long G_avg_weight_inh_dp,
    const long G_avg_weight_exc_dp,
    const long G_syn_count_inh_dp,
    const long G_syn_count_exc_dp,
    const long L_winner_take_all_map_dp,
    const int max_n_winner_take_all_layers,
    const int max_winner_take_all_layer_size,

    const long C_old_dp,
    const long C_new_dp, 
    const long C_source_dp, 
    const int chem_grid_w,
    const int chem_grid_h,
    const int chem_grid_d,
    const float chem_k_val, 
	const float chem_depreciation
){
    float* voltage_plot_data = reinterpret_cast<float*> (voltage_plot_data_dp);
    int* voltage_plot_map = reinterpret_cast<int*> (voltage_plot_map_dp);
    float* scatter_plot_data = reinterpret_cast<float*> (scatter_plot_data_dp);    
    int* scatter_plot_map = reinterpret_cast<int*> (scatter_plot_map_dp);    

    float* N_pos = reinterpret_cast<float*> (N_pos_dp);
    int* G_group_delay_counts = reinterpret_cast<int*> (G_group_delay_counts_dp);
    int* G_flags = reinterpret_cast<int*> (G_flags_dp);
    float* G_props = reinterpret_cast<float*> (G_props_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    int* N_rep_buffer = reinterpret_cast<int*> (N_rep_buffer_dp);
    int* N_rep_pre_synaptic = reinterpret_cast<int*> (N_rep_pre_synaptic_dp);
    int* N_rep_pre_synaptic_idcs = reinterpret_cast<int*> (N_rep_pre_synaptic_idcs_dp);
    int* N_rep_pre_synaptic_counts = reinterpret_cast<int*> (N_rep_pre_synaptic_counts_dp);
    // int* N_rep_pre_synaptic_counts = reinterpret_cast<int*> (N_rep_dp);
    
    int* N_delays = reinterpret_cast<int*> (N_delays_dp);

    int* N_flags = reinterpret_cast<int*> (N_flags_dp);
    float* N_states = reinterpret_cast<float*> (N_states_dp);
    
    float* N_weights = reinterpret_cast<float*> (N_weights_dp);

    float* fired = reinterpret_cast<float*> (fired_dp);
    int* last_fired = reinterpret_cast<int*> (last_fired_dp);
    float* firing_times = reinterpret_cast<float*> (firing_times_dp);
    int* firing_idcs = reinterpret_cast<int*> (firing_idcs_dp);
    int* firing_counts = reinterpret_cast<int*> (firing_counts_dp);
    int* G_firing_count_hist = reinterpret_cast<int*> (G_firing_count_hist_dp);

    int* G_stdp_config0 = reinterpret_cast<int*> (G_stdp_config0_dp);
    int* G_stdp_config1 = reinterpret_cast<int*> (G_stdp_config0_dp);


    float* G_avg_weight_inh = reinterpret_cast<float*> (G_avg_weight_inh_dp);
    float* G_avg_weight_exc = reinterpret_cast<float*> (G_avg_weight_exc_dp);
    int* G_syn_count_inh = reinterpret_cast<int*> (G_syn_count_inh_dp);
    int* G_syn_count_exc = reinterpret_cast<int*> (G_syn_count_exc_dp);

    int* L_winner_take_all_map = reinterpret_cast<int*> (L_winner_take_all_map_dp);

    float* C_old = reinterpret_cast<float*> (C_old_dp);
    float* C_new = reinterpret_cast<float*> (C_new_dp);
    float* C_source = reinterpret_cast<float*> (C_source_dp);

    
    return SnnSimulation(
        N,
        G,
        S,
        D,
        T,
        n_voltage_plots,
        voltage_plot_length,
        voltage_plot_data,
        voltage_plot_map,
        n_scatter_plots,
        scatter_plot_length,
        scatter_plot_data,
        scatter_plot_map,
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
        N_states,
        N_weights,
        fired,
        last_fired,
        firing_times,
        firing_idcs,
        firing_counts,
        G_firing_count_hist,
        G_stdp_config0,
        G_stdp_config1,
        G_avg_weight_inh,
        G_avg_weight_exc,
        G_syn_count_inh,
        G_syn_count_exc,
        L_winner_take_all_map,
        max_n_winner_take_all_layers,
        max_winner_take_all_layer_size,

        C_old,
        C_new,
        C_source,
        chem_grid_w,
        chem_grid_h,
        chem_grid_d,
        chem_k_val, 
        chem_depreciation
    );
}



PYBIND11_MODULE(snn_simulation_gpu, m)
    {
        
    m.def("print_random_numbers", &print_random_numbers2);
    
    py::class_<SnnSimulation, std::shared_ptr<SnnSimulation>>(m, "SnnSimulation_") //, py::dynamic_attr())
    //.def(py::init<int>())
    .def_readonly("N", &SnnSimulation::N)
    .def_readonly("G", &SnnSimulation::G)
    .def_readonly("S", &SnnSimulation::S)
    .def_readonly("D", &SnnSimulation::D)
    .def_readonly("t", &SnnSimulation::t)
    .def_readonly("update_duration", &SnnSimulation::update_duration)
    .def_readwrite("stdp_active", &SnnSimulation::stdp_active)
    .def_property("bupdate_chemical_contrations", 
        &SnnSimulation::get_b_update_chemical_contrations,
        &SnnSimulation::set_b_update_chemical_contrations
    )
    .def("update", &SnnSimulation::update)
    .def("update_chemical_contrations", &SnnSimulation::update_chemical_contrations)
    // .def("swap_groups", &SnnSimulation::swap_groups_python)
    .def("set_stdp_config", &SnnSimulation::set_stdp_config, 
        py::arg("stdp_config_id"), 
        py::arg("activate") = true)
    // .def("actualize_N_rep_pre_synaptic", &SnnSimulation::actualize_N_rep_pre_synaptic)
    .def("calculate_avg_group_weight", &SnnSimulation::calculate_avg_group_weight)
    // .def("remove_synapses_to_group", &SnnSimulation::remove_all_synapses_to_group, 
    //     py::arg("group"))
    // .def("nullify_all_weights_to_group", &SnnSimulation::nullify_all_weights_to_group, 
    //     py::arg("group"))
    .def("__repr__",
        [](const SnnSimulation &sim) {
            return "SnnSimulation(N=" + std::to_string(sim.N) + ")";
        });
    m.def("SnnSimulation", &make_SnnSimulation,
        py::arg("N"),
        py::arg("G"),
        py::arg("S"),
        py::arg("D"),
        py::arg("T"),
        py::arg("n_voltage_plots"),
        py::arg("voltage_plot_length"),
        py::arg("voltage_plot_data"),
        py::arg("voltage_plot_map"),
        py::arg("n_scatter_plots"),
        py::arg("scatter_plot_length"),
        py::arg("scatter_plot_data"),
        py::arg("scatter_plot_map"),
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
        py::arg("N_states"),
        py::arg("N_weights"),
        py::arg("fired"),
        py::arg("last_fired"),
        py::arg("firing_times"),
        py::arg("firing_idcs"),
        py::arg("firing_counts"),
        py::arg("G_firing_count_hist"),
        py::arg("G_stdp_config0"),
        py::arg("G_stdp_config1"),
        py::arg("G_avg_weight_inh"),
        py::arg("G_avg_weight_exc"),
        py::arg("G_syn_count_inh"),
        py::arg("G_syn_count_exc"),
        py::arg("L_winner_take_all_map"),
        py::arg("max_n_winner_take_all_layers"),
        py::arg("max_winner_take_all_layer_size"),
        py::arg("C_old"),
        py::arg("C_new"),
        py::arg("C_source"),
        py::arg("chem_grid_w") = 0,
        py::arg("chem_grid_h") = 0,
        py::arg("chem_grid_d") = 0,
        py::arg("chem_k_val") = .75f,
        py::arg("chem_depreciation") = 0.1f
    );
}