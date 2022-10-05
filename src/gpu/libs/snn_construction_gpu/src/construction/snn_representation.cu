#include <construction/snn_representation.cuh>


SnnRepresentation::SnnRepresentation(
    const int N_,
    const int G_,
    const int S_,
    const int D_,

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
){
    
	N = N_;
	G = G_;
	S = S_;
	D = D_;

	rand_states = rand_states_;

	N_pos = N_pos_;
	G_group_delay_counts = G_group_delay_counts_;
    G_flags = G_flags_; 
    G_props = G_props_; 
    N_rep = N_rep_;

	N_rep_buffer = N_rep_buffer_;
    N_rep_pre_synaptic = N_rep_pre_synaptic_; 
    N_rep_pre_synaptic_idcs = N_rep_pre_synaptic_idcs_; 
    N_rep_pre_synaptic_counts = N_rep_pre_synaptic_counts_;

    N_delays = N_delays_;

	N_flags = N_flags_;
	
	N_weights = N_weights_;

	L_winner_take_all_map = L_winner_take_all_map_;

    max_n_winner_take_all_layers = max_n_winner_take_all_layers_;
    max_winner_take_all_layer_size = max_winner_take_all_layer_size_;
}



__device__ void roll_copy(
	
	int* write_array, int* read_array, 
	int write_col, int read_col, 
	int write_row_start, 
	int n_write_array_cols, int n_read_array_cols,
	const int copy_length, 
	const int read_offset, 
	bool bprint){
	
	int write_idx;
	int read_idx;
	//int roll_mod = abs(swap_snk_N_s_start - swap_src_N_s_start);

	for (int s=0; s < abs(copy_length); s++){

		write_idx = write_col + (write_row_start + s) * n_write_array_cols;
		read_idx = read_col + (write_row_start + ((s +  read_offset) % copy_length)) * n_read_array_cols;

		if (bprint){
			printf("\nwrite_array[%d, %d]=%d -> read_array[%d, %d]=%d", 
			write_row_start + s, write_col, 
			write_array[write_idx], 
			write_row_start + ((s +  read_offset) % copy_length), read_col,
			read_array[read_idx] );
		}

		write_array[write_idx] = read_array[read_idx];

	}	

}


__device__ void shift_values_row_wise_(
	int shift_start_offset,
	int* array0, int* array1,
	int col0, int col1,
	int n_cols0, int n_cols1,
	int end_row,
	int swap_dir,
	bool bprint
){
	int idx_end0 = col0 + (end_row) * n_cols0;
	int idx_end1 = col1 + (end_row) * n_cols1;
	int value0;

	if (swap_dir == 1){
		for (int k=shift_start_offset; k < 0; k++){
			value0 = array0[idx_end0 + (k+swap_dir) * n_cols0];
			if (value0 > 0){
	
				if (bprint){
					printf("\nN_rep[%d, %d] = %d -> %d, G_swap_index[%d, %d] = %d -> %d",
						end_row + k, idx_end0, array0[idx_end0 + k * n_cols0], value0,
						end_row + k, idx_end1, array1[idx_end1 + k * n_cols1], array1[idx_end1 + (k+swap_dir) * n_cols1]
					);
		
				}
	
				array0[idx_end0 + k * n_cols0] = value0;
				array1[idx_end1 + k * n_cols1] = array1[idx_end1 + (k+swap_dir) * n_cols1];
			}
		
		}
	}
	
	else if (swap_dir == -1){
		for (int k=0; k > shift_start_offset; k--){
			value0 = array0[idx_end0 + (k+swap_dir) * n_cols0];
			if (value0 > 0){
	
				if (bprint){
					printf("\nN_rep[%d, %d] = %d -> %d, G_swap_index[%d, %d] = %d -> %d",
						end_row + k, idx_end0, array0[idx_end0 + k * n_cols0], value0,
						end_row + k, idx_end1, array1[idx_end1 + k * n_cols1], array1[idx_end1 + (k+swap_dir) * n_cols1]
					);
		
				}
	
				array0[idx_end0 + k * n_cols0] = value0;
				array1[idx_end1 + k * n_cols1] = array1[idx_end1 + (k+swap_dir) * n_cols1];
			}
		
		}
	}

}


__device__ void generate_synapses(
	const int N,
	const int n,
	const int neuron_idx,
	int* N_rep,
	int* G_swap_tensor,
	int& swap_src_N_s_start, int& swap_snk_N_s_start,
	int& swap_src_G_count, int& swap_snk_G_count,
	const int max_snk_count,
	curandState &local_state,
	int G_swap_tensor_shape_1, 
	const int swap_type,
	const int index_offset,
	const int relative_index_offset,
	const int swap_dir,
	bool bprint
){
	
	int snk_N;
	int min_G_swap_snk = G_swap_tensor[neuron_idx + swap_snk_N_s_start * G_swap_tensor_shape_1];
	int max_G_swap_snk = G_swap_tensor[neuron_idx + (swap_snk_N_s_start + swap_snk_G_count - 1) * G_swap_tensor_shape_1];
	if (swap_snk_G_count == 0){
		min_G_swap_snk = max_snk_count + relative_index_offset;
		max_G_swap_snk = -1;
	}
	float r;

	int s_end = swap_src_N_s_start + swap_src_G_count;
	

	for (int s=swap_src_N_s_start; s < s_end; s++){
	// for (int s=swap_src_N_s_start; s < swap_src_N_s_start + 2; s++){
		
		r = curand_uniform(&local_state);

		snk_N = __float2int_rd(r * __int2float_rn(max_snk_count)) + relative_index_offset;
			
				
		if (bprint) printf("\n[%d, %d] new=%d (%f), t=%d, s=%d, [%d, %d], [offset = %d - %d]", 
						   n, neuron_idx, snk_N, r, swap_type, s,
						   min_G_swap_snk, max_G_swap_snk, 
						   index_offset, relative_index_offset);

		if (swap_snk_G_count < max_snk_count)
		{	
			bool found = false;
			int i = 0;	
			int j = 0;
			int swap_idx = neuron_idx + (swap_snk_N_s_start)  * G_swap_tensor_shape_1;
			int G_swap0;
			int G_swap_m1;

			int write_row;

			int last_write_mode = 0;
			int write_mode = 0;

			// while ((!found) && (j < 40)){
			while ((!found) && (j < G_swap_tensor_shape_1)){
				
				write_mode = 0;
				//write_row = s - s_offset;
				// write = -i;
				swap_idx = neuron_idx + (swap_snk_N_s_start + i )  * G_swap_tensor_shape_1;
				
				G_swap0 = G_swap_tensor[swap_idx];
				G_swap_m1 = G_swap_tensor[swap_idx - G_swap_tensor_shape_1];


				if((snk_N < min_G_swap_snk) || (swap_snk_G_count == 0)){
				
	
					min_G_swap_snk = snk_N;
					
					if (swap_dir == 1){
						write_row = swap_snk_N_s_start - 1;
					} else {
						write_row = swap_snk_N_s_start;
					}
					write_mode = 1;

					if (swap_snk_G_count == 0){
						max_G_swap_snk = snk_N;
					}
					// G_swap_tensor[swap_idx - G_swap_tensor_shape_1] = G_swap0;	
					// G_swap_tensor[swap_idx] = snk_N;		
				}
				else if((snk_N > max_G_swap_snk)){
					write_mode = 2;
					// if (swap_snk_G_count == 0){
					// 	min_G_swap_snk = snk_N;
					// }
					max_G_swap_snk = snk_N;
					if (swap_dir == 1){
						write_row = swap_snk_N_s_start + swap_snk_G_count - 1;
					} else {
						write_row = swap_snk_N_s_start + swap_snk_G_count;
					}
					

				}
				else if ((G_swap_m1 < snk_N) && (snk_N < G_swap0)){
					write_mode = 3;
					if (swap_dir == 1){
						write_row = swap_snk_N_s_start + i - 1;
					} else {
						write_row = swap_snk_N_s_start + i;
					}
					
				}

				found = write_mode > 0;

				if (found){
					if (swap_dir == 1){
						swap_snk_N_s_start -= 1;
					} else {
						swap_src_N_s_start += 1;
					}
					
					// write = snk_N;
					// s_offset++;
					swap_snk_G_count++;
					swap_src_G_count--;
					// G_swap_tensor[neuron_idx + (write_row) * G_swap_tensor_shape_1] = write;
					break;}
				

				if ((snk_N == G_swap0)){
					snk_N = (snk_N + 1) % max_snk_count;
				}

				// if (bprint || (j >= 30)) {
				if (bprint) {
					printf("\n[%d, %d] + new=%d[i=%d, write_mode=%d] G_swap_m1=%d, G_swap0=%d, [%d, %d], max_snk_count=%d, (%d), swap_snk_G_count=%d, s=%d", 
						n, neuron_idx, snk_N, i, write_mode, G_swap_m1, G_swap0, 
						min_G_swap_snk,
						max_G_swap_snk,
						max_snk_count, swap_type, swap_snk_G_count, s);
				}
				
				i = (i + 1) % swap_snk_G_count;
				j++;

				// if (j >= 10){
				// 	printf("\nn=%d; new=%d[%d] G_swap0=%d, max_snk_count=%d, (%d), s=%d", 
				// 		   n, snk_N, i, G_swap0, max_snk_count, swap_type, s);
				// }
			}

			// if (bprint || (j >= 30)) {
			if (false) {
				printf("\n[%d, %d] (found j=%d, mod:%d->%d) N_rep[%d, %d]=%d (%d) [%d (snk_N) + %d - %d]", 
					n, neuron_idx, j, last_write_mode, write_mode,
					write_row, n, N_rep[n + (write_row) * N], N_rep[n + (swap_snk_N_s_start-1) * N], snk_N,
					index_offset, relative_index_offset);
			}

			//|| (j >= 30)
			if ((swap_dir > 0) && (write_mode > 1)){
				shift_values_row_wise_(
					swap_snk_N_s_start - write_row - 1,
					N_rep, G_swap_tensor,
					n, neuron_idx,
					N, G_swap_tensor_shape_1,
					write_row,
					swap_dir,
					bprint
				);
			} else if ((swap_dir < 0) && (write_mode > 0) && (write_mode != 2)){
				shift_values_row_wise_(
					write_row - swap_src_N_s_start - 1,
					N_rep, G_swap_tensor,
					n, neuron_idx,
					N, G_swap_tensor_shape_1,
					swap_src_N_s_start,
					swap_dir,
					bprint
				);
			}


			N_rep[n + (write_row) * N] = snk_N + index_offset - relative_index_offset;
			G_swap_tensor[neuron_idx + (write_row) * G_swap_tensor_shape_1] = snk_N;

			// bprint || (j >= 30)

			if (bprint  ) {
				printf("\n[%d, %d] (found j=%d, mod:%d->%d) N_rep[%d, %d]=%d (%d) [%d (snk_N) + %d - %d]", 
					n, neuron_idx, j, last_write_mode, write_mode,
					write_row, n, N_rep[n + (write_row) * N], N_rep[n + (swap_snk_N_s_start-1) * N], snk_N,
					index_offset, relative_index_offset);
			}
			last_write_mode = write_mode;
		} 

		// 	swap_snk_G_count++;
	}

}



__global__ void swap_groups_(
	const long* neurons, const int n_neurons, 
	const long* groups, const int n_groups,
	const int* neuron_group_indices,
	int* G_swap_tensor, const int G_swap_tensor_shape_1,
	const float* swap_rates,
	const int* group_neuron_counts_inh, const int* group_neuron_counts_exc, const int* group_neuron_counts_total, 
	const int* G_delay_distance,
	const int* N_relative_G_indices, const int* G_neuron_typed_ccount,
	int N,
	int G,
	int S,
	int D,
	int* N_flags,
	int* N_rep,
	int* N_delays,
	curandState* randstates,
	int* neuron_group_counts,
	const int expected_snk_type,
	const int print_idx,
	const int N_flags_row_type = 1,
	const int N_flags_row_group = 2
){
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (neuron_idx < n_neurons){

		// bool bprint = (neuron_idx == min(print_idx, n_neurons- 1));
		bool bprint = false;

		const int n = neurons[neuron_idx];
		
		const int group_index = neuron_group_indices[neuron_idx];
		const int snk_group_index = group_index + 2 * n_groups;

		const int swap_src_G = groups[group_index];
		const int src_G = groups[snk_group_index - n_groups];
		const int swap_snk_G = groups[snk_group_index];

		const float swap_rate = swap_rates[group_index];

		const int total_src_G_count = group_neuron_counts_total[group_index];
		const int total_snk_G_count = group_neuron_counts_total[snk_group_index];

		if (bprint){		
			printf("\n\nswap_src %d (%d), src_G %d %d (%d), swap_snk %d (%d)  neuron_group_indices[%d] = %d\n", 
			swap_src_G, total_src_G_count,
			N_flags[n + N_flags_row_group * N], src_G, group_neuron_counts_total[snk_group_index - n_groups],
			swap_snk_G, total_snk_G_count, neuron_idx, (int)neuron_group_indices[neuron_idx]);
		}

		int snk_N;
		int snk_type;
		int snk_G;

		int swap_delay_src = G_delay_distance[swap_src_G + src_G * G];
		int swap_delay_snk = G_delay_distance[swap_snk_G + src_G * G];

		int s_start = N_delays[n + min(swap_delay_src, swap_delay_snk) * N];
		int s_end =  N_delays[n + (max(swap_delay_src, swap_delay_snk) + 1) * N];

		int swap_src_N_s_start = s_start;
		int swap_snk_N_s_start = s_start;

		int swap_src_G_count = 0;
		int swap_snk_G_count = 0;

		for (int s=s_start; s < s_end; s++)
		{
			
			snk_N = N_rep[n + s * N];
			snk_type = N_flags[snk_N * 2];
			

			if (snk_type == expected_snk_type){
				
				
				snk_G = N_flags[snk_N + N_flags_row_group * N];

				if (snk_G == swap_src_G)
				{
					
					if (swap_src_G_count == 0){
						swap_src_N_s_start = s;
					}
					swap_src_G_count += 1;

					G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = -total_src_G_count-N_rep[n + s * N]; //-2;
					if (bprint) printf("\n(%d) n_snk=%d, (snk_G=%d)  (s=%d) %d %d, src_counts=[%d, ]", 
						n, N_rep[n + s * N], snk_G, s, snk_G == swap_src_G, snk_G == swap_snk_G, 
						swap_src_G_count);
					
					N_rep[n + s * N] = -1;
					
				}
				else if (snk_G == swap_snk_G)
				{
					if (snk_type == expected_snk_type){
						if (swap_snk_G_count == 0){
							swap_snk_N_s_start = s;
						}
						swap_snk_G_count += 1;
					} 

					G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_relative_G_indices[snk_N];
					
					if (bprint) printf("\n(%d) n_snk=%d, (snk_G=%d)  (s=%d) %d %d, snk_N_rel=%d", 
						n, N_rep[n + s * N], snk_G, s, 
						snk_G == swap_src_G, snk_G == swap_snk_G, 
						N_relative_G_indices[snk_N]);
				} 
				else if((swap_src_G_count > 0) || (swap_snk_G_count > 0)){
					G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_rep[n + s * N];	
				}
			} 
			else if((swap_src_G_count > 0) || (swap_snk_G_count > 0))
			{
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_rep[n + s * N];	
			}
			
		}

		if (swap_snk_G_count == 0){
			swap_snk_N_s_start = swap_src_N_s_start + swap_src_G_count;
		}
		// if (swap_snk_G_count_exc == 0){
		// 	swap_snk_N_s_start_exc += 1;
		// }

		if (swap_rate < 1.f){
			s_end = swap_src_N_s_start + swap_src_G_count;

			swap_src_G_count = __float2int_rd (__int2float_rn(swap_src_G_count) * swap_rate);

			for (int s=swap_src_N_s_start + swap_src_G_count; s < s_end; s++)
			{
				snk_N = - G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] - total_src_G_count;
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = snk_N;
				N_rep[n + s * N] =  snk_N;
			}
		}


		if (bprint){
			printf("\n\nrow intervals: src=[%d, %d (+%d)) snk=[%d, %d (+%d)), swap_rate=%f\n", 
				   swap_src_N_s_start, swap_src_N_s_start + swap_src_G_count, 
				   swap_src_G_count, 
				   swap_snk_N_s_start, swap_snk_N_s_start + swap_snk_G_count,
				   swap_snk_G_count, swap_rate);
					//    printf("exc: src=[%d, +%d] snk=[%d, +%d]\n", 
					//    swap_src_N_s_start_exc, swap_src_G_count_exc, swap_snk_N_s_start_exc, swap_snk_G_count_exc);
		}

		if (swap_src_G_count > 0){

			int distance = max(swap_snk_N_s_start - (swap_src_N_s_start + swap_src_G_count), 
						       min(0, swap_snk_N_s_start + swap_snk_G_count - swap_src_N_s_start));

			int swap_dir = 1 * (swap_snk_N_s_start > swap_src_N_s_start) + -1 * (swap_snk_N_s_start < swap_src_N_s_start);
			
			if (distance != 0){

				// if (swap_dir == 1){
					roll_copy(
						N_rep, G_swap_tensor, 
						n, neuron_idx, 
						min(swap_src_N_s_start, swap_snk_N_s_start + swap_snk_G_count), 
						N, G_swap_tensor_shape_1, 
						(swap_snk_N_s_start - swap_src_N_s_start) * (swap_dir == 1) + (swap_dir == -1) * (-distance + swap_src_G_count), 
						swap_src_G_count * (swap_dir == 1) - distance *  (swap_dir == -1), 
						bprint);
				// } else {
				// 	roll_copy(
				// 		N_rep, G_swap_tensor, 
				// 		n, neuron_idx, 
				// 		swap_snk_N_s_start, 
				// 		N, G_swap_tensor_shape_1, 
				// 		swap_snk_N_s_start - swap_src_N_s_start, 
				// 		swap_dir * swap_src_G_count, 
				// 		bprint);
				// }


				swap_src_N_s_start += distance;

				if (bprint) {printf("\n\nswap_src_N_s_start=%d, distance=%d\n", swap_src_N_s_start, distance);}

			}


			curandState local_state = randstates[neuron_idx];

			int max_snk_count;
			int index_offset;
			int relative_index_offset;
			if (expected_snk_type == 1){
				max_snk_count = group_neuron_counts_inh[snk_group_index];
				index_offset = G_neuron_typed_ccount[swap_snk_G];
				relative_index_offset = 0;
			}
			else if (expected_snk_type == 2){
				max_snk_count = group_neuron_counts_exc[snk_group_index];
				index_offset = G_neuron_typed_ccount[G + swap_snk_G];
				relative_index_offset = group_neuron_counts_inh[snk_group_index];
			}

			if (swap_src_G_count > 0){
				generate_synapses(
					N, n,
					neuron_idx, N_rep,
					G_swap_tensor,
					swap_src_N_s_start, swap_snk_N_s_start,
					swap_src_G_count, swap_snk_G_count,
					max_snk_count,
					local_state,
					G_swap_tensor_shape_1,
					expected_snk_type,
					index_offset,
					relative_index_offset,
					swap_dir,
					bprint
				);
			}

			randstates[neuron_idx] = local_state;
		}

		bool count = true;

		if (count){

			int swap_src_G_count = 0;
			int swap_snk_G_count = 0;
		
			for (int s=0; s < S; s++){
				snk_G = N_flags[N_rep[n + s * N] + N_flags_row_group * N]; 
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = snk_G;
				swap_src_G_count += (snk_G == swap_src_G);
				swap_snk_G_count += (snk_G == swap_snk_G);
			}	
			neuron_group_counts[neuron_idx] = swap_src_G_count;
			neuron_group_counts[neuron_idx + G_swap_tensor_shape_1] = swap_snk_G_count;

			if (swap_delay_src != swap_delay_snk){
				
				int d1 = max(swap_delay_src, swap_delay_snk);
				int count0 = swap_delay_src;

				for (int d=min(swap_delay_src, swap_delay_snk); d < d1; d++){
					N_delays[n + d * N] -= count0;
				}
			}
		}
	
	}
}


void SnnRepresentation::swap_groups(
	long* neurons, const int n_neurons, 
	long* groups, const int n_groups, 
	int* neuron_group_indices,
	int* G_swap_tensor, const int G_swap_tensor_shape_1,
	float* swap_rates_inh, float* swap_rates_exc,
	int* group_neuron_counts_inh, int* group_neuron_counts_exc, int* group_neuron_counts_total,
	int* G_delay_distance,
	int* N_relative_G_indices, int* G_neuron_typed_ccount,
	int* neuron_group_counts,
	const int print_idx
)
{
	LaunchParameters lp_swap_groups = LaunchParameters(n_neurons, (void *)swap_groups_);

	//printf("\nswap groups %d, %d\n", n_groups, n_neurons);

	swap_groups_ KERNEL_ARGS2(lp_swap_groups.grid3, lp_swap_groups.block3)(
		neurons, n_neurons,
		groups, n_groups,
		neuron_group_indices,
		G_swap_tensor, G_swap_tensor_shape_1,
		swap_rates_inh,
		group_neuron_counts_inh, group_neuron_counts_exc, group_neuron_counts_total,
		G_delay_distance,
		N_relative_G_indices, G_neuron_typed_ccount,
		N,
		G,
		S,
		D,
		N_flags,
		N_rep,
		N_delays,
		rand_states,
		neuron_group_counts,
		1,
		print_idx
	);

	checkCudaErrors(cudaDeviceSynchronize());

	swap_groups_ KERNEL_ARGS2(lp_swap_groups.grid3, lp_swap_groups.block3)(
		neurons, n_neurons,
		groups, n_groups,
		neuron_group_indices,
		G_swap_tensor, G_swap_tensor_shape_1,
		swap_rates_exc,
		group_neuron_counts_inh, group_neuron_counts_exc, group_neuron_counts_total,
		G_delay_distance,
		N_relative_G_indices, G_neuron_typed_ccount,
		N,
		G,
		S,
		D,
		N_flags,
		N_rep,
		N_delays,
		rand_states,
		neuron_group_counts,
		2,
		print_idx
	);

	checkCudaErrors(cudaDeviceSynchronize());
}

void SnnRepresentation::swap_groups_python(
	long neurons, const int n_neurons, 
	long groups, const int n_groups, 
	const long neuron_group_indices,
	const long G_swap_tensor, const int G_swap_tensor_shape_1,
	const long swap_rates_inh, const long swap_rates_exc,
	const long group_neuron_counts_inh, const long group_neuron_counts_exc, const long group_neuron_counts_total,
	const long G_delay_distance, 
	const long N_relative_G_indices, const long G_neuron_typed_ccount,
	long neuron_group_counts,
	const int print_idx
)
{
	swap_groups(reinterpret_cast<long*> (neurons), n_neurons, 
				reinterpret_cast<long*> (groups), n_groups, 
				reinterpret_cast<int*> (neuron_group_indices),
				reinterpret_cast<int*> (G_swap_tensor), G_swap_tensor_shape_1,
				reinterpret_cast<float*> (swap_rates_inh), reinterpret_cast<float*> (swap_rates_exc),
				reinterpret_cast<int*> (group_neuron_counts_inh), reinterpret_cast<int*> (group_neuron_counts_exc), reinterpret_cast<int*> (group_neuron_counts_total),
				reinterpret_cast<int*> (G_delay_distance),
				reinterpret_cast<int*> (N_relative_G_indices), reinterpret_cast<int*> (G_neuron_typed_ccount),
				reinterpret_cast<int*> (neuron_group_counts),
				print_idx
				
	);
}


__global__ void reset_N_rep_pre_synaptic_arrays(
	const int N,
	const int S,
	int* Buffer,
	int* N_rep_pre_synaptic,
	int* N_rep_pre_synaptic_idcs,
	int* N_rep_pre_synaptic_counts
){
	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 
	if (src_N < N){
		for (int s = 0; s < S; s++){
			Buffer[src_N + s * N] = -1;
			N_rep_pre_synaptic[src_N + s * N] = -1;
			N_rep_pre_synaptic_idcs[src_N + s * N] = -1;
		}

		if (src_N == 0){
			N_rep_pre_synaptic_counts[0] = 0;
		}
		N_rep_pre_synaptic_counts[src_N + 1] = 0;

	}

}


__global__ void reset_N_rep_snk_counts(
	const int N,
	int* N_rep_pre_synaptic_counts
){
	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (src_N < N){

		if (src_N == 0){
			N_rep_pre_synaptic_counts[0] = 0;
		}

		N_rep_pre_synaptic_counts[src_N + 1] = 0;
	}
}


__global__ void fill_N_rep_snk_counts(
	const int N,
	const int S,
	int* N_rep,
	int* N_rep_pre_synaptic_counts
){
	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 
	int snk_N;
	
	if (src_N < N){

		for (int s = 0; s < S; s++){
			snk_N = N_rep[src_N + s * N];

			if (snk_N == -1){
				printf("\n %d", src_N);
			}

			atomicAdd(&N_rep_pre_synaptic_counts[snk_N + 1], 1);
		}
	}
}


__global__ void fill_unsorted_N_rep_pre_synaptic_idcs(
	const int N,
	const int S,
	int* N_rep,
	int* SortBuffer,
	int* N_rep_pre_synaptic_idcs,
	int* N_rep_pre_synaptic_counts
){

	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 

	if (src_N < N){

		int snk_N;
		int write_idx;
	
		int synapse_idx;

		for (int s = 0; s < S; s++){
			
			synapse_idx = src_N + s * N;

			snk_N = N_rep[synapse_idx];
			write_idx = N_rep_pre_synaptic_counts[snk_N];
			
			while (synapse_idx != -1){
				
				synapse_idx = atomicExch(&N_rep_pre_synaptic_idcs[write_idx], synapse_idx);
				SortBuffer[write_idx] = snk_N;
				write_idx++;
			}

			atomicAdd(&N_rep_pre_synaptic_counts[snk_N],1);

		}
	}

}


__global__ void fill_N_rep_pre_synaptic(
	const int N,
	const int S,
	int* N_rep,
	int* N_rep_pre_synaptic,
	int* N_rep_pre_synaptic_idcs,
	int* N_rep_pre_synaptic_counts
){

	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 

	if (src_N < N){

		int snk_N;
		int write_idx;
	
		int synapse_idx;

		for (int s = 0; s < S; s++){
			
			synapse_idx = src_N + s * N;

			snk_N = N_rep[synapse_idx];
			write_idx = N_rep_pre_synaptic_counts[snk_N];
			
			while (N_rep_pre_synaptic_idcs[write_idx] != synapse_idx){
				write_idx++;
			}

			N_rep_pre_synaptic[write_idx] = src_N;

		}
	}

}


void sort_N_rep_sysnaptic(
	const int N,
	const int S,
	int* sort_keys_buffer,
	int* N_rep_pre_synaptic_idcs,
	int* N_rep_pre_synaptic_counts,
	const bool verbose = true
){

	auto sort_keys_buffer_dp = thrust::device_pointer_cast(sort_keys_buffer);
	auto N_rep_dp = thrust::device_pointer_cast(N_rep_pre_synaptic_idcs);
	auto N_rep_counts_dp = thrust::device_pointer_cast(N_rep_pre_synaptic_counts);

	int n_sorted = 0;
	int N_batch_size = 50000;
	int S_batch_size;

	std::string msg;
	if (verbose) {
		msg = "sorted: 0/" + std::to_string(N);
		std::cout << msg;
	}

	while (n_sorted < N){
			
	 	if (n_sorted + N_batch_size > N){
	 		N_batch_size = N - n_sorted;
		} 

		// printf("\nN_batch_size=%d", N_batch_size);

		S_batch_size = N_rep_counts_dp[n_sorted + N_batch_size] - N_rep_counts_dp[n_sorted];

		// printf("\nS_batch_size=%d\n", S_batch_size);

	 	thrust::stable_sort_by_key(N_rep_dp, N_rep_dp + S_batch_size, sort_keys_buffer_dp);
	 	thrust::stable_sort_by_key(sort_keys_buffer_dp, sort_keys_buffer_dp + S_batch_size, N_rep_dp);
		
	 	n_sorted += N_batch_size;
	 	sort_keys_buffer_dp += S_batch_size;
	 	N_rep_dp += S_batch_size;

	 	if (verbose) { 
	 		std::cout << std::string(msg.length(),'\b');
	 		msg = "sorted: " + std::to_string(n_sorted) + "/" + std::to_string(N);
	 		std::cout << msg;
	 	}
	}

	if (verbose) printf("\n");

}


void SnnRepresentation::actualize_N_rep_pre_synaptic(){

	LaunchParameters launch_pars = LaunchParameters(N, (void *)reset_N_rep_pre_synaptic_arrays);

	checkCudaErrors(cudaDeviceSynchronize());

	reset_N_rep_pre_synaptic_arrays KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep_buffer,
		N_rep_pre_synaptic,
		N_rep_pre_synaptic_idcs,
		N_rep_pre_synaptic_counts
	);

	checkCudaErrors(cudaDeviceSynchronize());

	fill_N_rep_snk_counts KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep,
		N_rep_pre_synaptic_counts
	);

	thrust::device_ptr<int> count_dp = thrust::device_pointer_cast(N_rep_pre_synaptic_counts);

	checkCudaErrors(cudaDeviceSynchronize());

	thrust::inclusive_scan(thrust::device, count_dp, count_dp + N + 1, count_dp);

	checkCudaErrors(cudaDeviceSynchronize());
	printf("\nfill (unsorted) N_rep_pre_synaptic_idcs...");

	fill_unsorted_N_rep_pre_synaptic_idcs KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep,
		N_rep_buffer,
		N_rep_pre_synaptic_idcs,
		N_rep_pre_synaptic_counts
	);

	checkCudaErrors(cudaDeviceSynchronize());

	reset_N_rep_snk_counts KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		N_rep_pre_synaptic_counts
	);	

	checkCudaErrors(cudaDeviceSynchronize());

	fill_N_rep_snk_counts KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep,
		N_rep_pre_synaptic_counts
	);

	checkCudaErrors(cudaDeviceSynchronize());

	thrust::inclusive_scan(thrust::device, count_dp, count_dp + N + 1, count_dp);

	checkCudaErrors(cudaDeviceSynchronize());

	sort_N_rep_sysnaptic(N, S, N_rep_buffer, N_rep_pre_synaptic_idcs, N_rep_pre_synaptic_counts);

	checkCudaErrors(cudaDeviceSynchronize());

	printf("\nfill N_rep_pre_synaptic...");

	fill_N_rep_pre_synaptic KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep,
		N_rep_pre_synaptic,
		N_rep_pre_synaptic_idcs,
		N_rep_pre_synaptic_counts
	);

	checkCudaErrors(cudaDeviceSynchronize());


	// std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	// std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	printf(" done.\n");
	//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}





__global__ void remove_all_synapses_to_group_(
	const int N,
	const int S,
	const int* N_flags,
	int* N_rep,
	int* N_rep_pre_synaptic,
	int* N_rep_pre_synaptic_idcs,
	int* N_rep_pre_synaptic_counts,
	const int group,
	const int N_flags_row_group = 2,
	const int delete_synapse_value = -2
){
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (n < N)
	{
		for (int s = 0; s < S; s++){
			if (N_flags[N_rep[n  + s * N] + N_flags_row_group * N] == group){
				N_rep[n  + s * N] = -2;
			}
		}

		if (N_flags[n + N_flags_row_group * N] == group){		
			int s_end2 = N_rep_pre_synaptic_counts[n + 1];
			for (int s2 = N_rep_pre_synaptic_counts[n]; s2 < s_end2; s2++){
				N_rep_pre_synaptic[N_rep_pre_synaptic_idcs[s2]] = -2;
			}
		}
	}
}

void SnnRepresentation::remove_all_synapses_to_group(const int group){

	checkCudaErrors(cudaDeviceSynchronize());

	LaunchParameters launch_pars = LaunchParameters(N, (void *)remove_all_synapses_to_group_);

	remove_all_synapses_to_group_ KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N, S,
		N_flags, 
		N_rep,
		N_rep_pre_synaptic,
		N_rep_pre_synaptic_idcs,
		N_rep_pre_synaptic_counts,
		group
	);

	checkCudaErrors(cudaDeviceSynchronize());

}


__global__ void nullify_all_weights_to_group_(
	const int N,
	const int S,
	const int* N_flags,
	int* N_rep,
	float* N_weights,
	const int group,
	const int N_flags_row_group = 2,
	const int delete_synapse_value = -2
){
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (n < N)
	{
		for (int s = 0; s < S; s++){
			if (N_flags[N_rep[n  + s * N] + N_flags_row_group * N] == group){
				N_weights[n  + s * N] = 0.f;
			}
		}

	}
}

void SnnRepresentation::nullify_all_weights_to_group(const int group){

	checkCudaErrors(cudaDeviceSynchronize());

	LaunchParameters launch_pars = LaunchParameters(N, (void *)nullify_all_weights_to_group_);

	nullify_all_weights_to_group_ KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N, S,
		N_flags, 
		N_rep,
		N_weights,
		group
	);

	checkCudaErrors(cudaDeviceSynchronize());

}
