#pragma once

#include <utils/cuda_header.h>


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


void fill_N_rep(
	int N,
	int S,
	int D,
	int G,
	curandState* curand_states,
	const int n_curand_states,
	const int* N_flags,
	const int* cc_src,
	const int* cc_snk,
	const int* G_rep,
	const int* G_neuron_counts,
	const int* G_group_delay_counts,
	int* G_autapse_indices,
	int* G_relative_autapse_indices,
	bool has_autapses,
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