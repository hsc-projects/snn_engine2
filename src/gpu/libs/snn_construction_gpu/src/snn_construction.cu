#include <snn_construction.cuh>


void __global__ fill_N_flags_group_id_and_G_neuron_count_per_type_(
	const int N,
    const int G,
	const float* N_pos, 
	float N_pos_shape_x, float N_pos_shape_y, float N_pos_shape_z,
    const int N_pos_n_cols,
	int* N_flags,
    const int N_flags_row_type,
	const int N_flags_row_group,

	const float G_shape_x,
	const float G_shape_y,
	const float G_shape_z,
	const float min_G_shape,

	int* G_neuron_counts  // NOLINT(readability-non-const-parameter)
)
{
	const int n = blockIdx.x * blockDim.x + threadIdx.x;  // NOLINT(bugprone-narrowing-conversions)
	
	if (n < N)
	{
		const int x = __float2int_rd(((N_pos[n * N_pos_n_cols] / N_pos_shape_x) * G_shape_x));  
		const int y = __float2int_rd(((N_pos[n * N_pos_n_cols + 1] / N_pos_shape_y) * G_shape_y));
		const int z = __float2int_rd(((N_pos[n * N_pos_n_cols + 2] / N_pos_shape_z) * G_shape_z));

		const int group = x + G_shape_x * y + G_shape_x * G_shape_y * z;

		// assign neuron to location-based group 
		N_flags[n + N_flags_row_group * N] = group;
		
		// count group: row <> neuron type (1 or 2), column <> group id
		atomicAdd(&G_neuron_counts[group + G * (N_flags[n + N_flags_row_type * N] - 1)], 1);	
	}
}


void fill_N_flags_group_id_and_G_neuron_count_per_type(
    const int N, 
    const int G, 
    const float* N_pos,
	const int N_pos_shape_x, const int N_pos_shape_y, const int N_pos_shape_z,
    int* N_flags,
	int* G_neuron_counts,
    const int G_shape_x, const int G_shape_y, const int G_shape_z,
	const int N_pos_n_cols,
	const int N_flags_row_type,
	const int N_flags_row_group
)
{	
	// Assign location-based group ids to neurons w.r.t. their positions.

	// N: 				number of neurons
	// G: 				number of location-based groups

	// N_pos: device 	pointer to the position array
	// N_pos_n_cols: 	number of columns of N_pos

	// G_shape_*:		number of location-based groups along the *-axis
	// 
	//
	// Example:
	// 
	//	G = 8
	//	N_pos_n_cols = 3
	//	N_pos = [[0,0,0], 
	//			 [1,1,0]]
	//		
	//	N_flags_row_group = 2
	//
	// 	N_flags = 	[[...], [0,1],   [0,0]]	  -> 	N_flags = 	[[...], [0,1], [0,2]]
	//			        				 
	//	
	//	G_neuron_counts = 	[[0,0,0],	-> 	[[1,0,0],
	//						 [0,0,0]]		 [0,0,1]]
	//	


	cudaDeviceSynchronize();
	LaunchParameters launch(N, (void *)fill_N_flags_group_id_and_G_neuron_count_per_type_); 
	
	int min_G_shape = std::min(std::min(G_shape_x, G_shape_y), G_shape_z);

	fill_N_flags_group_id_and_G_neuron_count_per_type_ KERNEL_ARGS2(launch.grid3, launch.block3) (
		N,
        G,
		N_pos,
		static_cast<float>(N_pos_shape_x), static_cast<float>(N_pos_shape_y), static_cast<float>(N_pos_shape_z),
        N_pos_n_cols,
		N_flags,
		N_flags_row_type,
		N_flags_row_group,
		// LG_neuron_counts.dp,
		static_cast<float>(G_shape_x), static_cast<float>(G_shape_y), static_cast<float>(G_shape_z),
		static_cast<float>(min_G_shape),
		G_neuron_counts
	);
	
	cudaDeviceSynchronize();
	printf("\n");
}


__device__ float sigmoidal_connection_probability(
	const float delay,
	const float max_delay,
	const float alpha = 1.f,
	const float beta = 1.f,
	const float gamma = 0.125f
)
{
	const float inv_max_delay = (1.f / max_delay);
	const float normalized_delay = delay * inv_max_delay;
	
	const float sigmoid = 1.f - (1.f / (1.f + expf(-(alpha * delay - 1.f))));
	const float offset = gamma * (1.f - powf(normalized_delay, 2.f));

	return fminf(1.f, beta * inv_max_delay * (sigmoid + offset));
}


__global__ void fill_G_neuron_count_per_delay_(
		const float fS,
		const int D,
		const float fD,
		const int G,
		const int* G_delay_distance,
		int* G_neuron_counts
)
{
	// connection probabilities from inh -> exc
	// connection probabilities from exc -> (inh & exc)

	const int g = threadIdx.x + blockDim.x * blockIdx.x;

	if (g < G)
	{
		const int g_row = g * G;
		
		int delay = 0;
		int count_inh = 0;
		int count_exc = 0;

		const int ioffs_inh = 2 * G + g;
		const int ioffs_exc = (2 + D) * G + g;
	
		for (int h = 0; h < G; h++)
		{

			delay = G_delay_distance[g_row + h];

			count_inh = G_neuron_counts[h];
			count_exc = G_neuron_counts[h + G];

			atomicAdd(&G_neuron_counts[delay * G + ioffs_inh], count_inh);
			atomicAdd(&G_neuron_counts[delay * G + ioffs_exc], count_exc);
		}
	}
}


void fill_G_neuron_count_per_delay(
	const int S,
	const int D,
	const int G,
	const int* G_delay_distance,
	int* G_neuron_counts
)
{	
	cudaDeviceSynchronize();
	LaunchParameters launch(G, (void *)fill_G_neuron_count_per_delay_); 

	fill_G_neuron_count_per_delay_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		static_cast<float>(S),
		D,
		static_cast<float>(D),
		G,
		G_delay_distance,
		G_neuron_counts
	);
	
	cudaDeviceSynchronize();
	printf("\n");
}


__device__ void expected_syn_count(
	const float fD,
	const int D, 
	const int G, 
	const int* G_neuron_counts,
	const int ioffs_inh,
	const int ioffs_exc,

	const float alpha_inh, const float beta_inh, const float gamma_inh,
	const float alpha_exc, const float beta_exc, const float gamma_exc,
	
	float* exp_cnt_inh, float* exp_cnt_exc,
	const int group,
	const bool verbose = 1,
	const int print_group = 1
)
{
	*exp_cnt_inh = 0;
	*exp_cnt_exc = 0;

	
	for (int delay = 0; delay < D; delay++)
	{
		const int idx = (delay)*G;
		// # inh targets (exc)
		const float n_inh_targets = __int2float_rn(G_neuron_counts[ioffs_exc + idx]);
		// # exc targets (inh & exc)
		float n_exc_targets = n_inh_targets + __int2float_rn(G_neuron_counts[ioffs_inh + idx]);

		if ((delay == 0) && (G_neuron_counts[ioffs_inh - G] > 0))
		{
			// only exc neurons will have a technical probability > 0 to form an autapse
			n_exc_targets-= 1.f;
		}

		const float fdelay = __int2float_rn(delay);

		const float prob_inh = sigmoidal_connection_probability(fdelay, fD, alpha_inh, beta_inh, gamma_inh);
		const float prob_exc = sigmoidal_connection_probability(fdelay, fD, alpha_exc, beta_exc, gamma_exc);
		if (n_inh_targets > 0){
			*exp_cnt_inh += roundf(n_inh_targets * prob_inh + .5);
		}
		if (n_exc_targets > 0){
			*exp_cnt_exc += roundf(n_exc_targets * prob_exc + .5);
		}
		if ((verbose) && (group == print_group)){
			printf("\ninh expected_syncount = %f (++ %f)", *exp_cnt_inh, n_inh_targets * prob_inh);
			printf("\n(exc) a=%f,b=%f,g=%f", alpha_exc, beta_exc, gamma_exc);
			printf("\nexc expected_syncount = %f (++ %f)", *exp_cnt_exc, roundf(n_exc_targets * prob_exc + .5));
		}

	}
}

__device__ void prob_improvement(
	int* mode,
	float* alpha,
	float* beta,
	float* gamma,
	const float expected_count,
	const float error,
	const float fS,
	const float fD,
	const float alpha_delta,
	const float beta_delta,
	const float gamma_delta
	// const int group
)
{
	if (*mode == 0)
	{
		// if (group == 0) printf("\n(%d) beta=%f", group, *beta);
		// if (group == 0) printf("\n(%d) beta_delta=%f", group, beta_delta);
		*beta = fminf(*beta * fmaxf(fminf(fS / (expected_count), 1.f + beta_delta), 1.f - beta_delta), fD * (1- *gamma));
		*mode = 1;
		// if (group == 0) printf("\n(%d) beta=%f", group, *beta);
	}
	else if (*mode == 1)
	{
		// if (group == 0) printf("\n(%d) alpha=%f", group, *alpha);
		*alpha = fmaxf(*alpha + fmaxf(fminf( ( expected_count - fS) / fS, alpha_delta), -alpha_delta),
			0.05f);
		*mode = 0;
		// if (group == 0) printf("\n(%d) alpha=%f", group, *alpha);
	}

	if (error > (fS * 0.1f))
	{
		// if (group == 0) printf("\n(%d) gamma=%f", group, *gamma);
		*gamma = fminf(*gamma * fmaxf(fminf(fS / (expected_count), 1.f + gamma_delta), 1.f - gamma_delta), .3f);
		// if (group == 0) printf("\n(%d) gamma=%f", group, *gamma);
	}
	
}


__device__ int roughly_optimize_connection_probabilites(
	const float fS,
	const float fD,
	const int D,
	const int G,
	const int* G_neuron_counts, 
	const int ioffs_inh, const int ioffs_exc,
	float* p_alpha_inh, float* p_beta_inh, float* p_gamma_inh,
	float* p_alpha_exc, float* p_beta_exc, float* p_gamma_exc,
	const float alpha_delta, const float beta_delta, const float gamma_delta,
	const int group,
	const bool verbose = 1,
	const int print_group = 1
){
	
	int j = 0;

	float exp_cnt_inh = 0.f;
	float exp_cnt_exc = 0.f;
		
	int mode_inh = 0;
	int mode_exc = 0;

	float error_inh = fS;
	float error_exc = fS;
	const float p = (1. / fS);


	while (((error_inh > p) || (error_exc > p)) && (j < 300))
	{
		expected_syn_count(
			fD, 
			D, 
			G, 
			G_neuron_counts,
			ioffs_inh, ioffs_exc,
			*p_alpha_inh, *p_beta_inh, *p_gamma_inh,
			*p_alpha_exc, *p_beta_exc, *p_gamma_exc,
			&exp_cnt_inh, &exp_cnt_exc,
			group,
			verbose, print_group
		);

		error_inh = fabsf(exp_cnt_inh - fS);
		error_exc = fabsf(exp_cnt_exc - fS);
		
		j++;
		
		if ((error_inh > p))
		{
			prob_improvement(&mode_inh,
				p_alpha_inh, p_beta_inh, p_gamma_inh,
			 	exp_cnt_inh, error_inh,
			 	fS, fD,
			 	alpha_delta, beta_delta, gamma_delta
				//, group 
			);
		}
		if ((error_exc > p))
		{
			prob_improvement(&mode_exc,
				p_alpha_exc, p_beta_exc, p_gamma_exc,
				exp_cnt_exc, error_exc,
				fS, fD,
				alpha_delta, beta_delta, gamma_delta
				//, group
			);
		}

		if ((verbose) && (group == print_group))
		{
			printf("\n\n0 (%d, %d) expected_count_inh %f, expected_count_exc %f, modes %d, %d",
				group, j, exp_cnt_inh, exp_cnt_exc, 
				mode_inh, mode_exc);
			// if ((error_inh > p))
				printf("\n1 (%d, %d) alpha_inh %f, beta_inh %f , gamma_inh %f  \nerror=%f",
					group, j, *p_alpha_inh, *p_beta_inh, *p_gamma_inh,  exp_cnt_inh - fS);
			// if ((error_exc > p))
				printf("\n2 (%d, %d) alpha_exc %f, beta_exc %f , gamma_exc %f  \nerror=%f",
					group, j, *p_alpha_exc, *p_beta_exc, *p_gamma_exc,  exp_cnt_exc - fS);
		}

	}

	return j;
}

__global__ void fill_G_exp_ccsyn_per_src_type_and_delay_(
	const int S,
	const float fS,
	const int D,
	const float fD,
	const int G,
	const int* G_neuron_counts,
	float* G_conn_probs,
	int* G_exp_ccsyn_per_src_type_and_delay,
	bool verbose = 0,
	int print_group = 1
)
{
	// connection probabilities from inh -> exc
	// connection probabilities from exc -> (inh & exc)

	const int g = threadIdx.x + blockDim.x * blockIdx.x;

	if (g < G)
	{
		const int ioffs_inh = 2 * G + g;
		const int ioffs_exc = (2 + D) * G + g;

		float alpha_inh = 4.f;
		float alpha_exc = 2.f;
		float beta_inh = 1.f + fD / 3.f;
		float beta_exc = 1.f;
		float gamma_inh = .01f;
		float gamma_exc = .05f;

		const float alpha_delta = 0.04f;
		const float beta_delta = 0.04f;
		const float gamma_delta = 0.001f;



		const int opt_runs = roughly_optimize_connection_probabilites(
				fS,
				fD, 
				D,
				G,
				G_neuron_counts, 
				ioffs_inh, ioffs_exc,
				&alpha_inh, &beta_inh, &gamma_inh,
				&alpha_exc, &beta_exc, &gamma_exc,
				alpha_delta, beta_delta, gamma_delta, 
				g, 
				verbose);

		if ((g < 10) && (opt_runs > 98) || ((g == print_group) && (verbose))) {
			printf("\n(GPU: optimize_connection_probabilites) group(%d, opt_runs) = %d", g, opt_runs);
		}
		// if ((verbose) && (g == print_group)) {
		// 	printf("\nalpha_inh = %f, beta_inh = %f, gamma_inh = %f", alpha_inh, beta_inh, gamma_inh);
		// 	printf("\nalpha_exc = %f, beta_exc = %f, gamma_exc = %f", alpha_exc, beta_exc, gamma_exc);
		// }

		int expected_synapses_inh = 0;
		int expected_synapses_exc = 0;

		// int delay_with_min_exp_inh_syn_ge1 = 0
		int delay_with_max_inh_targets = 0;
		int exp_inh_syn_with_max_targets = 0;
		int max_inh_targets = 0;
		// int delay_with_min_exp_exc_syn_ge1 = 0
		int delay_with_max_exc_targets = 0;
		int exp_exc_syn_with_max_targets = 0;
		int max_exc_targets = 0;

		int idx = 0;
		int exc_syn_count = 0;
		int inh_syn_count = 0;

		for (int delay = 0; delay < D; delay++)
		{
			const float fdelay = __int2float_rn(delay);
			float prob_inh = sigmoidal_connection_probability(fdelay, fD, alpha_inh, beta_inh, gamma_inh);
			float prob_exc = sigmoidal_connection_probability(fdelay, fD, alpha_exc, beta_exc, gamma_exc);

			G_conn_probs[(g)*D + delay] = prob_inh;
			G_conn_probs[(G * D) + (g * D) + delay] = prob_exc;

			idx = delay * G;
			const int n_inh_targets = G_neuron_counts[idx + ioffs_exc];
			int n_exc_targets = n_inh_targets + G_neuron_counts[idx + ioffs_inh];
			const float f_n_inh_targets = __int2float_rn(n_inh_targets);
			float f_n_exc_targets = __int2float_rn(n_exc_targets);

			if ((delay == 0) && (G_neuron_counts[ioffs_inh - G] > 0))
			{
				// only exc neurons will have a technical probability > 0 to form an autapse
				n_exc_targets-=1;
				f_n_exc_targets-= 1.f;
			}

			inh_syn_count = min(__float2int_ru(prob_inh * f_n_inh_targets), n_inh_targets);
			expected_synapses_inh += inh_syn_count;
			G_exp_ccsyn_per_src_type_and_delay[g + idx + G] = expected_synapses_inh;
			
			idx += (D + 1) * G;
			exc_syn_count = __float2int_ru(prob_exc * f_n_exc_targets);
			expected_synapses_exc += exc_syn_count;
			G_exp_ccsyn_per_src_type_and_delay[g + idx + G] = expected_synapses_exc;

			if ((n_inh_targets > max_inh_targets)){
				exp_inh_syn_with_max_targets = inh_syn_count;
				delay_with_max_inh_targets = delay;
				max_inh_targets = n_inh_targets;
			}
			if ((n_exc_targets > max_exc_targets)){
				exp_exc_syn_with_max_targets = exc_syn_count;
				delay_with_max_exc_targets = delay;
				max_exc_targets = n_exc_targets;
			}

			// expected_synapses_inh += min(__float2int_ru(prob_inh * f_n_inh_targets), n_inh_targets);
			if ((verbose) && (g == print_group)) {
				printf("\nexp inh %f", prob_inh * f_n_inh_targets);
				printf("\nexp exc %f -> %d | %f (sum=%d)", 
					prob_exc * f_n_exc_targets, 
					min(__float2int_ru(prob_exc * f_n_exc_targets), n_exc_targets),
					roundf(prob_exc * f_n_exc_targets + .5),
					expected_synapses_exc
				);  
			}
		}
		
		// int res_inh = G_exp_ccsyn_per_src_type_and_delay[g + idx - (D * G)];
		// int res_exc = G_exp_ccsyn_per_src_type_and_delay[g + idx + G];

		if ((expected_synapses_inh != S)){
			int add = S - expected_synapses_inh;
			if (expected_synapses_inh > S){
				if (exp_inh_syn_with_max_targets < 1)  
				{
					add = 0;
					printf("\n(GPU: optimize_connection_probabilites) delay_inh(g=%d, exp_too_low=%d, max_targets=%d) = %d", 
					       g, exp_inh_syn_with_max_targets, max_inh_targets, delay_with_max_inh_targets);
				}
			} else if (exp_inh_syn_with_max_targets >= max_inh_targets){
				add = 0;
				printf("\n(GPU: optimize_connection_probabilites) delay_inh(g=%d, exp_too_high=%d, max_targets=%d) = %d", 
					   g, exp_inh_syn_with_max_targets, max_inh_targets, delay_with_max_inh_targets);
			}
			if (add != 0){
				for (int delay = delay_with_max_inh_targets; delay < D; delay++){
					G_exp_ccsyn_per_src_type_and_delay[g + (delay + 1) * G] += add;
				}
			}
			// printf("\n(%d) %d -> %d ", g, expected_synapses_inh, G_exp_ccsyn_per_src_type_and_delay[g + idx + G]);
		}

		if (expected_synapses_exc != S){
			int add = S - expected_synapses_exc;
			if (expected_synapses_exc > S){
				if (exp_exc_syn_with_max_targets < 1)  
				{
					add = 0;
					printf("\n(GPU: optimize_connection_probabilites) delay_exc(g=%d, exp_too_low=%d, max_targets=%d) = %d", 
						   g, exp_exc_syn_with_max_targets, max_exc_targets, delay_with_max_exc_targets);
				}
			} else if (exp_exc_syn_with_max_targets >= max_exc_targets){
				add = 0;
				printf("\n(GPU: optimize_connection_probabilites) delay_exc(g=%d, exp_too_high=%d, max_targets=%d) = %d", 
					   g, exp_exc_syn_with_max_targets, max_exc_targets, delay_with_max_exc_targets);
			} 
			if (add != 0){
				for (int delay = delay_with_max_exc_targets; delay < D; delay++){
					G_exp_ccsyn_per_src_type_and_delay[ g + (delay + 2 + D) * G] += add;
				}
			}
			if (G_exp_ccsyn_per_src_type_and_delay[g + (2 * D + 1) * G] != S){
				printf("\n(GPU: optimize_connection_probabilites) add(g=%d, exp=%d, max_targets=%d) = %d (%d, %d)", 
					   g, exp_exc_syn_with_max_targets, max_exc_targets, add, expected_synapses_exc,
					   G_exp_ccsyn_per_src_type_and_delay[g + (2 * D + 1) * G]);
			}

		} 

		if ((verbose) && (g == print_group)) {
			printf("\nres_inh = %d", expected_synapses_inh);
			printf("\nres_exc = %d", expected_synapses_exc);
		}
	}
}



void fill_G_exp_ccsyn_per_src_type_and_delay(
	const int S,
	const int D,
	const int G,
	const int* G_neuron_counts,
	float* G_conn_probs,
	int* G_exp_ccsyn_per_src_type_and_delay,
	bool verbose
)
{	
	cudaDeviceSynchronize();
	LaunchParameters launch(G, (void *)fill_G_exp_ccsyn_per_src_type_and_delay_); 

	fill_G_exp_ccsyn_per_src_type_and_delay_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		S,
		static_cast<float>(S),
		D,
		static_cast<float>(D),
		G,
		G_neuron_counts,
		G_conn_probs,
		G_exp_ccsyn_per_src_type_and_delay,
		verbose
	);
	
	cudaDeviceSynchronize();
	printf("\n");
}


__device__ int relative_typed_delay_rep_index(
	const int N_autapse_idx,
	const int G_rep_idx0,
	const int G_rep_idx1,
	const int g_N_count,
	const int* G_rep,
	const int n_groups,
	const int* cc_snk,
	bool verbose

)
{
	if (g_N_count == 0)
	{
		return -1;
	}

	int G_rep_idx = G_rep_idx0;
	int g = G_rep[G_rep_idx];
	int Ng_start = cc_snk[g];
	const int Ng_last = cc_snk[G_rep[G_rep_idx1] +1];

	
	if ((N_autapse_idx < Ng_start) || (N_autapse_idx >= Ng_last))
	{
		return -1;
		if (verbose)
		{
			printf(
			"(search, not in range) g=(%d), n=%d, G_rep[%d: %d], Ng_start=%d, Ng_last=%d\n", 
			g, N_autapse_idx, G_rep_idx0, G_rep_idx1, Ng_start, Ng_last);
		}
	}

	int result = N_autapse_idx;
	result -= Ng_start;
	int Ng_next = cc_snk[g + 1];

	if (verbose)
	{
		printf("(search) g=(%d), n=%d, Ng_start=%d, Ng_next=%d\n", g, N_autapse_idx, Ng_start, Ng_next);
	}
	//if (bprint)
	//{
	//	printf("\n  search (%d) g_start_col %d, n_g_search %d, g=%d",
	//		N_autapse_idx, g_search_start_col, n_g_search, g );
	//	printf("\n  (%d)  [%d], src_G %d, g = %d [%d, %d]... %d]",
	//		N_autapse_idx, result, src_G, g, start_col_next_group, end_col_next_group, last_col);
	//}
	
	bool found = (N_autapse_idx >= Ng_start) && (N_autapse_idx < Ng_next);

	int Ng_prev = Ng_next;
	
	while ((!found) && (G_rep_idx < G_rep_idx1))
	{
		G_rep_idx++;
		
		g = G_rep[G_rep_idx];
		Ng_start = cc_snk[g];
		Ng_next = cc_snk[g + 1];

		result -= (Ng_start - Ng_prev);

		found = (N_autapse_idx >= Ng_start) && (N_autapse_idx < Ng_next);
		if (verbose)
		{
			printf("(search, found=%d) g=(%d), n=%d, Ng_start=%d, Ng_next=%d\n", found, g, N_autapse_idx, Ng_start, Ng_next);
		}
	}
	return result * found + (-1) * (!found);
}


__global__ void fill_relative_autapse_indices_(
	const int D,
	const int G,
	const int* cc_src,
	const int* cc_snk,
	const int* G_rep,
	const int* G_delay_counts,
	int* G_autapse_indices,
	int* G_relative_autapse_indices,
	bool verbose = 0,
	int print_group = 1
)
{
	const int g = blockIdx.x * blockDim.x + threadIdx.x;  // NOLINT(bugprone-narrowing-conversions, cppcoreguidelines-narrowing-conversions)
	
	if (g < G)
	{
		
		const int N_autapse_idx = cc_src[g];
		const int g_N_count = cc_src[g + 1] - N_autapse_idx;
	
		for (int d=0; d < D; d++)
		{
			const int g_rep_col0 = G_delay_counts[g * (D + 1) + d];
			const int g_rep_col1 = G_delay_counts[g * (D + 1) + d+1]-1;
			const int G_rep_idx0 = g * G + g_rep_col0;
			const int G_rep_idx1 = g * G + g_rep_col1;
			const int n_groups = g_rep_col1 - g_rep_col0 +1;

			if (verbose && (g == print_group))
			{
				printf("g=(%d), n=%d, d=%d, g_rep_cols=[%d, %d], idcs=[%d,%d], groups=[%d, ...,%d]\n", 
					g, N_autapse_idx, d, 
					g_rep_col0, g_rep_col1,
					G_rep_idx0, G_rep_idx1,
					G_rep[G_rep_idx0], G_rep[G_rep_idx1]);
			}

			const int relative_autapse_index = relative_typed_delay_rep_index(
				N_autapse_idx,
				G_rep_idx0,
				G_rep_idx1,
				g_N_count,
				G_rep,
				n_groups,
				cc_snk,
				verbose && (g == print_group)
			);

			G_relative_autapse_indices[g + d * G] = relative_autapse_index;

			if (relative_autapse_index != -1){
				G_autapse_indices[g + d * G] = N_autapse_idx;
			} else {
				G_autapse_indices[g + d * G] = -1;
			}

			if (verbose && (g == print_group))
			{
				printf("g=(%d), n=%d, d=%d, N=%d, rN=%d\n", 
					g, N_autapse_idx, d, 
					G_autapse_indices[g + d * G],
					relative_autapse_index);
			}

			//if (bprint)
			//{
			//	printf("\nres: %d << %d, g = %d, d = %d\n---------\n", relative_self_index[g + d * G], self_index[g + d * G],g,d);
			//}

		}
	}
}

__forceinline__ __device__ int random_uniform_int(curandState *local_state, const float min, const float max)
{
	return __float2int_rd(fminf(min + curand_uniform(local_state) * (max - min + 1.f), max));
}

__device__ int random_uniform_int_with_exclusion(
	curandState *local_state, 
	const float minf, 
	const float maxf,
	const float maxf0,
	const bool exclude,
	const int autapse_idx,
	const int n, 
	const int s
){
	int new_sink = __float2int_rd(fminf(minf + curand_uniform(local_state) * (maxf - minf + 1.f), maxf));
	int i = 0;
	if (exclude){
		while ((new_sink==autapse_idx) && (i<50))
		{
			new_sink = __float2int_rd(fminf(minf + curand_uniform(local_state) * (maxf - minf + 1.f), maxf));
			i++;
		}

		if (i < 50){ return new_sink; } else { 
			printf("\n Loop-Warning [autapse_2](%d, %d) range=[%f, %f] -> [%f, %f], autapse_idx=%d, sink=%d", 
				   n, s, 0.f, maxf0, minf, maxf, autapse_idx, new_sink); 
		}
	}
}


__device__ void print_array(int* arr, int r, int c, int col0, int sep_row){
	
	printf("\n\n");
	int v;
	for (int i=0; i < c; i++)
	{
		v = col0 + i;
		printf("[%d]", v);
		if (v < 10){
			printf(" ");
		}
	}
	printf("\n");
	for (int j=0; j < r; j++)
	{
		if (j == sep_row){
			for (int i=0; i < c; i++)
			{
				printf("----");
			}
			printf("\n");
		}
		for (int i=0; i < c; i++)
		{
			v = arr[i + j * c];
			printf(" %d ", v);
			if (v < 10){
				printf(" ");
			}
		}
		printf("\n");
	}
}


__device__ bool duplicated_int(
	const int min_hit, const int max_hit,
	const int row_idx0,
	const int n, const int s, 
	const float maxf0, const float minf, const float maxf, 
	const int delay_col0, const int delay_col1, const int autapse_idx, const int new_sink,
	int* N_rep,
	const int k
){
	int i = delay_col0;

	if ((k==45)){
		printf("\n Loop-Warning [duplicated](%d, %d) range=[%f, %f] -> [%f, %f], delay_cols=[%d,%d], autapse_idx=%d, sink=%d", 
			n, s, 0.f, maxf0, minf, maxf, delay_col0, delay_col1, autapse_idx, new_sink);}
	
	// check if the drawn integer has alredy been set

	if ((min_hit < new_sink) && (new_sink < max_hit)) {
		while (i < s) {
			if (N_rep[row_idx0 + i] == new_sink) { 
				if ((k >= 45)){
					printf("\nLoop-Warning [duplicated (%d)](%d, %d), col0=%d, write_idx=%d, k=%d) rep=%d sink=%d", 
					true, n, s, i, row_idx0 + i, k, N_rep[row_idx0 + i], new_sink);}
				return true; }
			i++;
		}
	} else if ((new_sink == min_hit) || (new_sink == max_hit)){ 
		if ((k >= 45)){
			printf("\nLoop-Warning [duplicated (%d) hit](%d, %d), col0=%d, write_idx=%d, k=%d) rep=%d sink=%d", 
			true, n, s, i, row_idx0 + i, k, N_rep[row_idx0 + i], new_sink);}
		return true; }
	return false;
}


__global__ void k_set_locally_indexed_connections(
	const int N,
	const int S,
	const int D,
	const int G,
	curandState* curand_states,
	const int* N_flags,
	const int* cc_src,
	const int* G_neuron_counts,
	const int* G_relative_autapse_indices,
	bool has_autapses,
	const int gc_location0,
	const int gc_location1,
	const int gc_conn_shape0,
	const int gc_conn_shape1,
	//const float init_weight,
	//float* weights,
	const int* cc_syn,
	int* N_delays,
	int* sort_keys,
	int* N_rep,
	const int N_flags_row_group,
	bool verbose
)
{
	extern __shared__ int sh_delays[];
	int* n_targets = &sh_delays[(D+1) * blockDim.x];

	const int n = gc_location0 + blockIdx.x * blockDim.x + threadIdx.x;

	//{ printf("(%d, 0) = %d\n", n, N_rep[n * S]);}
	
	if (n < gc_location0 + gc_conn_shape0)
	{
		curandState local_state = curand_states[n];
		
		const int src_G = N_flags[n + N_flags_row_group * N];
		int tdx = threadIdx.x;
		const int row_idx0 = n * S;

		N_delays[n] = 0;

		// if (n == gc_location0){print_array(sh_delays, 2 * D + 1, blockDim.x, gc_location1);}
		
		sh_delays[tdx] = gc_location1;

		for (int d=1; d<D+1; d++)
		{
			int end_rep_col = cc_syn[src_G + d * G];
			
			sh_delays[tdx + d * blockDim.x] = gc_location1 + end_rep_col;
			n_targets[tdx + (d-1)* blockDim.x] = G_neuron_counts[src_G + (d-1) * G];
			
			N_delays[n + d * N] += end_rep_col;
			
		}

		if ((verbose) && (n == gc_location0)){ print_array(sh_delays, 2 * D + 1, blockDim.x, gc_location0, D + 1); }

		int sort_key = row_idx0 + gc_location1; // + gc_location1 + max(0, (D - S) * n);
		
		// [delay_col0, delay_col1]: column-interval in which to write sink neurons
		int delay = 0;
		int delay_col0 = sh_delays[tdx];
		int delay_col1 = sh_delays[tdx + blockDim.x];
		int n_rep_cols = delay_col1 - delay_col0;
		
		// [min, max]: interval from which to draw an integer ('sink'-neuron)
		int min = 0;
		int max = n_targets[tdx] - 1;
		float maxf = __int2float_rn(max);
		float minf = 0.f;
		float maxf0 = maxf;
		
		int autapse_idx = -1;
		if (has_autapses){ 
			autapse_idx = G_relative_autapse_indices[src_G + delay * G] + (n - cc_src[src_G]); 
		}
		int new_sink;

		int min_hit = -1;
		int max_hit = gc_location1 + gc_conn_shape1;
				
		// fill N_rep[n, s] for in [gc_location1, gc_location1 + gc_conn_shape1]
		for (int s = gc_location1; s < gc_location1 + gc_conn_shape1; s++)
		{
			const int write_idx = row_idx0 + s;

			while ((s == delay_col1) && (delay < D+1))
			{
				// if we reach the end of the write interval, update all variables
				if (delay >= 1){ autapse_idx = -1; }

				tdx += blockDim.x;
				delay_col0 = sh_delays[tdx];
				delay_col1 = sh_delays[tdx + blockDim.x];
				n_rep_cols = delay_col1 - delay_col0;
				if (n_rep_cols >0)
				{
					min_hit = -1;
					max_hit = gc_location1 + gc_conn_shape1;
					min = 0;
					max = n_targets[tdx] - 1;
					minf = 0.f;
					maxf0 = __int2float_rn(max);
					maxf = maxf0;
					sort_key = write_idx;				
				}
				delay++;
			}

			if (n_rep_cols > 0) {	

				if (min > max){ printf("\n Warning [min>max] (%d, %d in [%d, %d], d=%d) %d > %d, range=[%f, %f] targets %d/%d", 
				n, s, delay, delay_col0, delay_col1, min, max, 
				0.f, maxf0, n_targets[tdx], G_neuron_counts[src_G + (delay) * G]); }

				new_sink = random_uniform_int_with_exclusion(&local_state, minf, maxf, maxf0, (has_autapses) && (delay == 0), autapse_idx, n, s);

				
				if (s == delay_col0){
					min_hit = new_sink;
					max_hit = new_sink;
				} else {
					
					int k = 0;
					bool duplicated = true;
					while (duplicated && (k<=50))
					{
						duplicated = duplicated_int(min_hit, max_hit, row_idx0, n, s, maxf0, minf, maxf, delay_col0, delay_col1, autapse_idx, new_sink, &N_rep[0], k);
						if (duplicated) {
							new_sink = random_uniform_int_with_exclusion(&local_state, minf, maxf, maxf0, (has_autapses) && (delay == 0), autapse_idx, n, s);
						}
						k++;
					}
				} 

				// we can narrow the range if we hit the border
				//if (new_sink > max){ printf("\n Loop-Warning [new_sink>max] (%d, %d) range=[%f, %f] -> [*, %d], sink=%d", n, s, 0.f, maxf0, max, new_sink); }
				if (new_sink == max){ max--; maxf -= 1.f; }
				else if (new_sink == min){ min++; minf += 1.f; }
				else if (new_sink < min_hit){ min_hit = new_sink; }
				else if (new_sink > max_hit){ max_hit = new_sink; }
				
				sort_keys[write_idx] = sort_key;
				N_rep[write_idx] = new_sink;	

				// if (n == gc_location0) { printf("(%d, %d) = %d\n", n, s, N_rep[write_idx]);}
			}	
		}
		
		curand_states[n] = local_state;
	}
}

void fill_N_rep(
	const int N,
	const int S,
	const int D,
	const int G,
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
	const bool has_autapses,
	const int gc_location0,
	const int gc_location1,
	const int gc_conn_shape0,
	const int gc_conn_shape1,
	const int* cc_syn,
	int* N_delays,
	int* sort_keys,
	int* N_rep,
	const int N_flags_row_group,
	bool verbose
)
{
	printf("Connecting: ((%d, %d), (%d, %d))", gc_location0, gc_location1, gc_conn_shape0, gc_conn_shape1);
	cudaDeviceSynchronize();
	LaunchParameters launch(G, (void *)fill_relative_autapse_indices_); 
	fill_relative_autapse_indices_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		D,
		G,
		cc_src,
		cc_snk,
		G_rep,
		G_group_delay_counts,
		G_autapse_indices,
		G_relative_autapse_indices,
		false);

	const int block_size_limit = 64;
	const int shared_memory_size = block_size_limit * ((2 * D) + 1) * sizeof(int);
	
	LaunchParameters l(gc_conn_shape0, (void*)k_set_locally_indexed_connections, block_size_limit, shared_memory_size);
	cudaDeviceSynchronize();

	// l.print_info();
	
	k_set_locally_indexed_connections KERNEL_ARGS3(l.grid3, l.block3, l.block3.x * ((2 * D) + 1) * sizeof(int))(
		N,
		S,
		D,
		G,
		curand_states,
		N_flags,
		cc_src,
		G_neuron_counts,
		G_relative_autapse_indices,
		has_autapses,
		gc_location0,
		gc_location1,
		gc_conn_shape0,
		gc_conn_shape1,
		//group_conn.initial_weight,
		//weights,
		cc_syn,
		N_delays,
		sort_keys,
		N_rep,
		N_flags_row_group,
		verbose
	  );

	  checkCudaErrors(cudaDeviceSynchronize());
	  printf("\n");
}


void sort_N_rep(
	const int N,
	const int S,
	int* sort_keys,
	int* N_rep,
	const bool verbose
){

	auto sort_keys_ptr = thrust::device_pointer_cast(sort_keys);
	auto N_rep_ptr = thrust::device_pointer_cast(N_rep);

	int n_sorted = 0;
	int N_batch_size = 50000;
	int S_batch_size = N_batch_size * S;

	std::string msg;
	if (verbose) {
		msg = "sorted: 0/" + std::to_string(N);
		std::cout << msg;
	}

	// thrust::stable_sort_by_key(N_rep_ptr, N_rep_ptr + N * S, sort_keys_ptr);
	// thrust::stable_sort_by_key(sort_keys_ptr, sort_keys_ptr + N * S, N_rep_ptr);
	
	while (n_sorted < N){
			
	 	if (n_sorted + N_batch_size > N){
	 		N_batch_size = N - n_sorted;
			S_batch_size = N_batch_size * S;
	 	} 

	 	thrust::stable_sort_by_key(N_rep_ptr, N_rep_ptr + S_batch_size, sort_keys_ptr);
	 	thrust::stable_sort_by_key(sort_keys_ptr, sort_keys_ptr + S_batch_size, N_rep_ptr);
		
	 	n_sorted += N_batch_size;
	 	sort_keys_ptr += S_batch_size;
	 	N_rep_ptr += S_batch_size;

	 	if (verbose) { 
	 		std::cout << std::string(msg.length(),'\b');
	 		msg = "sorted: " + std::to_string(n_sorted) + "/" + std::to_string(N);
	 		std::cout << msg;
	 	}
	}

	if (verbose) printf("\n");

}


__global__ void reindex_N_rep_(
	const int N,
	const int S,
	const int D,
	const int G,
	const int* N_flags,
	const int* cc_src,
	const int* cc_snk,
	const int* G_rep,
	const int* G_neuron_counts,
	const int* G_group_delay_counts,
	const int gc_location0,
	const int gc_location1,
	const int gc_conn_shape0,
	const int gc_conn_shape1,
	const int* cc_syn,
	int* N_delays,
	int* sort_keys,
	int* N_rep,
	const int  N_flags_row_group,
	bool verbose
)
{

	extern __shared__ int sh_delays[];
	int* n_targets = &sh_delays[(D+1) * blockDim.x];

	const int n = gc_location0 + blockIdx.x * blockDim.x + threadIdx.x;
	
	if (n < gc_location0 + gc_conn_shape0){

		int print_N = gc_location0 + gc_conn_shape0 - 1;

		const int src_G = N_flags[n + N_flags_row_group * N];
		int tdx = threadIdx.x;

		sh_delays[tdx] = gc_location1;
		for (int d=1; d<D+1; d++)
		{
			sh_delays[tdx + d * blockDim.x] = gc_location1 + cc_syn[src_G + d * G];
			n_targets[tdx + (d-1)* blockDim.x] = G_neuron_counts[src_G + (d-1) * G];	
		}
		
		
		int N_rep_idx = n * S + gc_location1;
		int snk_local = N_rep[N_rep_idx];
		int G_group_delay_counts_idx =  src_G * (D + 1);
		int n_groups_per_delay = G_group_delay_counts[G_group_delay_counts_idx + 1] - G_group_delay_counts[G_group_delay_counts_idx];
		int G_rep_idx = src_G * G;
		int G_rep_idx0 = G_rep_idx;
		int G_rep_idx1 = G_rep_idx + n_groups_per_delay;
		int snk_G0 = G_rep[G_rep_idx0];
		int snk_G1 = G_rep[G_rep_idx1];
		int snk0 = cc_snk[snk_G0];
		int snk1 = cc_snk[snk_G0 + 1];
		int delay = 0;
		int delay_col0 = sh_delays[tdx];
		int delay_col1 = sh_delays[tdx + blockDim.x];
		int n_rep_cols = delay_col1 - delay_col0;
		int idx_offset = snk0;
		int offset_delta;

		int snk_global;

		//sort_keys[N_rep_idx] = n * S + delay;

		if (verbose && (n == print_N)) 
		{
			printf("(%d, %d, %d in [%d, %d]) N_rep[%d]=%d G_rep[%d]=%d, G_rep[%d]=%d\n",  
				src_G, n, gc_location1, delay_col0, delay_col1, N_rep_idx, snk_local,G_rep_idx0, snk_G0, G_rep_idx1, snk_G1);
		}

		for (int s = gc_location1; s < gc_location1 + gc_conn_shape1; s++){

			N_rep_idx = n * S + s; 
			snk_local = N_rep[N_rep_idx];

			while ((s == delay_col1) && (delay < D+1))
			{
				
				// if we reach the end of the write interval, update all variables
				tdx += blockDim.x;
				delay_col0 = sh_delays[tdx];
				delay_col1 = sh_delays[tdx + blockDim.x];
				n_rep_cols = delay_col1 - delay_col0;
				delay++;
				G_rep_idx += n_groups_per_delay;

				G_group_delay_counts_idx++;
				n_groups_per_delay = G_group_delay_counts[G_group_delay_counts_idx + 1] - G_group_delay_counts[G_group_delay_counts_idx];

				G_rep_idx0 = G_rep_idx;
				G_rep_idx1 = G_rep_idx + n_groups_per_delay - 1;
				snk_G0 = G_rep[G_rep_idx0];
				snk_G1 = G_rep[G_rep_idx1];

				snk0 = cc_snk[snk_G0];
				snk1 = cc_snk[snk_G0 + 1];
				idx_offset = snk0;
			}

			if (verbose && (n == print_N)) 
			{
				printf("(%d, %d, %d in [%d, %d]) N_rep[%d]=%d G_rep[%d]=%d, G_rep[%d]=%d #g=%d = (%d - %d)\n",  
					src_G, n, s, delay_col0, delay_col1, N_rep_idx, snk_local,G_rep_idx0, snk_G0, G_rep_idx1, snk_G1, n_groups_per_delay, 
					G_group_delay_counts[G_group_delay_counts_idx + 1], G_group_delay_counts[G_group_delay_counts_idx]);
			}

			if (n_rep_cols > 0){
				snk_global = snk_local + idx_offset;

				while (snk_global >= snk1){
					G_rep_idx0 += 1;
					snk_G0 = G_rep[G_rep_idx0];
					snk0 = cc_snk[snk_G0];
					offset_delta = snk0 - snk1;
	
					snk1 = cc_snk[snk_G0 + 1];
					idx_offset += offset_delta;
					snk_global += offset_delta;
				}
				
				N_rep[N_rep_idx] = snk_global;
				sort_keys[N_rep_idx] = n * S + delay;
			}

		}

	}
}



void reindex_N_rep(
	const int N,
	const int S,
	const int D,
	const int G,
	const int* N_flags,
	const int* cc_src,
	const int* cc_snk,
	const int* G_rep,
	const int* G_neuron_counts,
	const int* G_group_delay_counts,
	const int gc_location0,
	const int gc_location1,
	const int gc_conn_shape0,
	const int gc_conn_shape1,
	const int* cc_syn,
	int* N_delays,
	int* sort_keys,
	int* N_rep,
	const int N_flags_row_group,
	bool verbose
)
{
	printf("Reindexing: ((%d, %d), (%d, %d))", gc_location0, gc_location1, gc_conn_shape0, gc_conn_shape1);
	cudaDeviceSynchronize();
	LaunchParameters launch(gc_conn_shape0, (void *)reindex_N_rep_); 

	// l.print_info();
	
	reindex_N_rep_ KERNEL_ARGS3(launch.grid3, launch.block3, launch.block3.x * ((2 * D) + 1) * sizeof(int))(
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
		gc_location0,
		gc_location1,
		gc_conn_shape0,
		gc_conn_shape1,
		cc_syn,
		N_delays,
		sort_keys,
		N_rep,
		N_flags_row_group,
		verbose
	  );

	  checkCudaErrors(cudaDeviceSynchronize());
	  printf("\n");
}


__global__ void fill_N_rep_groups_(
	const int N,
	const int S,
	const int* N_flags,
	const int* N_rep,
	int* N_rep_groups,
    const int N_flags_row_group
){
	const int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < N){

		int rep_idx;

		for (int s=0; s<S; s++){
			rep_idx = n + s * N;
			N_rep_groups[rep_idx] = N_flags[N_rep[rep_idx] + N_flags_row_group * N]; 
		}
	}
}


void fill_N_rep_groups(
	const int N,
	const int S,
	const int* N_flags,
	const int* N_rep,
	int* N_rep_groups,
    const int N_flags_row_group
){
	LaunchParameters launch(N, (void *)fill_N_rep_groups_); 

	fill_N_rep_groups_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		N,
		S,
		N_flags,
		N_rep,
		N_rep_groups,
		N_flags_row_group
	);
}