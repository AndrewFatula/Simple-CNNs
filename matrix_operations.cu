#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <cmath>


#include "matrix_operations.cuh"

#define OptNofThreads 128
#define OptNofBlocks 128
#define OptNofBlocksX 32
#define OptNofBlocksY 32

template <typename T>
__global__ void get_first_layer_dropout(T* first_layer_weights, T* dropped_first_layer, unsigned int* first_dropout_indices, int n_of_neurons, int n_for_first_dropout) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	int rand_index = first_dropout_indices[bdx*n_for_first_dropout + tdx] % n_of_neurons;

	first_dropout_indices[bdx*n_for_first_dropout + tdx] = rand_index;

	dropped_first_layer[bdx*n_for_first_dropout + tdx] = first_layer_weights[bdx*(n_of_neurons + 1) + rand_index];

	first_layer_weights[bdx*(n_of_neurons + 1) + rand_index] = 0.0f;

}


template <typename T>
__global__ void get_second_layer_dropout(T* second_layer_weights, T* dropped_second_layer, unsigned int* second_dropout_indices, int n_of_second_layer_neurons, int n_for_second_dropout) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	int rand_index = second_dropout_indices[bdx*n_for_second_dropout + tdx] % n_of_second_layer_neurons;

	second_dropout_indices[bdx*n_for_second_dropout + tdx] = rand_index;
	
	dropped_second_layer[bdx*n_for_second_dropout + tdx] = second_layer_weights[bdx*(n_of_second_layer_neurons + 1) + rand_index];

	second_layer_weights[bdx*(n_of_second_layer_neurons + 1) + rand_index] = 0.0f;

}




template <typename T>
__global__ void get_third_layer_dropout(T* third_layer_weights, T* dropped_third_layer, unsigned int* third_dropout_indices, int n_of_third_layer_neurons, int n_for_third_dropout) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	int rand_index = third_dropout_indices[bdx*n_for_third_dropout + tdx] % n_of_third_layer_neurons;

	third_dropout_indices[bdx*n_for_third_dropout + tdx] = rand_index;

	dropped_third_layer[bdx*n_for_third_dropout + tdx] = third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + rand_index];

	third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + rand_index] = 0.0f;
}






template <typename T>
__global__ void take_first_layer_dropout(T* first_layer_weights, T* dropped_first_layer, unsigned int* first_dropout_indices, int n_of_neurons, int n_for_first_dropout) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	first_layer_weights[bdx*(n_of_neurons + 1) + first_dropout_indices[bdx*n_for_first_dropout + tdx]] = dropped_first_layer[bdx*n_for_first_dropout + tdx];
}



template <typename T>
__global__ void take_second_layer_dropout(T* second_layer_weights, T* dropped_second_layer, unsigned int* second_dropout_indices, int n_of_second_layer_neurons, int n_for_second_dropout) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	second_layer_weights[bdx*(n_of_second_layer_neurons + 1) + second_dropout_indices[bdx*n_for_second_dropout + tdx]] = dropped_second_layer[bdx*n_for_second_dropout + tdx];
}



template <typename T>
__global__ void take_third_layer_dropout(T* third_layer_weights, T* dropped_third_layer, unsigned int* third_dropout_indices, int n_of_third_layer_neurons, int n_for_third_dropout) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + third_dropout_indices[bdx*n_for_third_dropout + tdx]] = dropped_third_layer[bdx*n_for_third_dropout + tdx];
}










template<typename T>
__global__ void get_first_layer_output(T* images, T* first_layer_weights, T* first_layer_output, T* sigmoid_multiplier, int n_of_images, int n_of_second_layer_neurons, int n_of_neurons) {

	int tdx = threadIdx.x;
	int idx = threadIdx.x;
	int bdx = blockIdx.x;





	while (tdx < n_of_images) {

		if (bdx < n_of_second_layer_neurons) {
			first_layer_output[bdx * n_of_images + tdx] = 0.0f;
			sigmoid_multiplier[bdx*n_of_images + idx] = 0.0f;
		}
		else {
			first_layer_output[bdx * n_of_images + tdx] = 1.0f;
		}

		if (bdx < n_of_second_layer_neurons) {
			for (int i = 0; i < n_of_neurons; ++i) {
				first_layer_output[bdx * n_of_images + tdx] += images[tdx*n_of_neurons + i] *
					first_layer_weights[bdx * n_of_neurons + i];
			}

			first_layer_output[bdx * n_of_images + tdx] = (float)1.0f /
				(1 + exp(-first_layer_output[bdx * n_of_images + tdx]));

		}

		tdx += OptNofThreads;
	}

}


template <typename T>
__global__ void get_second_layer_output(T* first_layer_output, T* second_layer_output, T* second_layer_weights, T* error_multiplier, int n_of_images, int n_of_second_layer_neurons, int n_of_third_layer_neurons) {

	int tdx = threadIdx.x;
	int idx = threadIdx.x;
	int bdx = blockIdx.x;


	__shared__ T second_layer_output_shared[OptNofThreads];

	while (bdx < n_of_third_layer_neurons + 1) {
		while (idx < n_of_images) {

			if (bdx < n_of_third_layer_neurons) {
				second_layer_output_shared[tdx] = 0;
				error_multiplier[bdx*n_of_images + idx] = 0.0f;
			}
			else {
				second_layer_output_shared[tdx] = 1;
			}

			if (bdx < n_of_third_layer_neurons) {
				for (int i = 0; i < n_of_second_layer_neurons + 1; ++i) {

					second_layer_output_shared[tdx] += second_layer_weights[bdx*(n_of_second_layer_neurons + 1) + i] * first_layer_output[i*n_of_images + idx];
				}
				second_layer_output_shared[tdx] = 1.0f / (1 + exp(-second_layer_output_shared[tdx]));

			}


			second_layer_output[n_of_images*bdx + idx] = second_layer_output_shared[tdx];

			idx += OptNofThreads;
		}
		if (idx >= n_of_images) {
			idx = threadIdx.x;
		}
		bdx += OptNofBlocks;
	}

}


template <typename T>
__global__ void get_third_layer_output(T* third_layer_weights, T* second_layer_output, T* third_layer_output, T* third_layer_sums, int n_of_images, int n_of_third_layer_neurons) {

	int tdx = threadIdx.x;
	int idx = threadIdx.x;
	int bdx = blockIdx.x;

	__shared__ T output[OptNofThreads];

	while (idx < n_of_images) {

		output[tdx] = 0.0f;

		for (int i = 0; i < n_of_third_layer_neurons + 1; ++i) {
			output[tdx] += second_layer_output[i*n_of_images + idx] * third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + i];
		}


		output[tdx] = exp(output[tdx]);

		third_layer_output[bdx*n_of_images + idx] = output[tdx];



		if (bdx == 0) {
			third_layer_sums[idx] = 0.0f;
		}

		idx += OptNofThreads;
	}

}


template <typename T>
__global__ void get_posibilities(T* third_layer_output, T* third_layer_sums, T* posibilities, int n_of_images, int n_of_models) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;
	int bldx = blockDim.x;

	__shared__ T sums_array[10];

	__shared__ T second_layer_sum[1];

	while (bdx < n_of_images) {

		sums_array[tdx] = third_layer_output[tdx*n_of_images + bdx];

		if (tdx == 0) {
			second_layer_sum[0] = 0.0f;
		}

		atomicAdd(&second_layer_sum[0], sums_array[tdx]);

		posibilities[tdx*n_of_images + bdx] = (float)sums_array[tdx] / second_layer_sum[0];

		bdx += OptNofBlocks;
	}

}





template <typename T>
__global__ void get_errors(T* posibilities, T* labels, T* errors, T* square_error, int n_of_images, int n_of_models, int i) {

	int idx = threadIdx.x;
	int tdx = threadIdx.x;
	int bdx = blockIdx.x;


	if (tdx == 0) {
		square_error[bdx] = 0.0f;
	}


	__shared__ T error_sums[OptNofThreads];

	while (tdx < n_of_images) {


		float posibility = posibilities[bdx*n_of_images + tdx];
		float label = labels[bdx*n_of_images + tdx];


		errors[bdx*n_of_images + tdx] = (label*(posibility - 1) + (1 - label)*posibility);

		error_sums[idx] = (label - posibility)*(label - posibility);

		atomicAdd(&square_error[0], error_sums[idx]);

		tdx += OptNofThreads;
	}


}



template <typename T>
__global__ void get_third_layer_correction(T* errors, T* second_layer_output, T* third_layer_correction, T* third_layer_weights, T* error_multiplier, int n_of_images, int n_of_third_layer_neurons, int n_of_models) {

	int idx = threadIdx.x;
	int bdx = blockIdx.x;

	if (idx == 0) {
		for (int i = 0; i < n_of_third_layer_neurons + 1; ++i) {
			third_layer_correction[bdx*(n_of_third_layer_neurons + 1) + i] = 0.0f;
		}
	}

	while (idx < n_of_images) {

		float error = errors[bdx*n_of_images + idx];

		for (int i = 0; i < n_of_third_layer_neurons + 1; ++i) {
			atomicAdd(&third_layer_correction[bdx*(n_of_third_layer_neurons + 1) + i], second_layer_output[i*n_of_images + idx] * error);
			atomicAdd(&error_multiplier[i*n_of_images + idx], error * third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + i]);
		}

		idx += OptNofThreads;
	}

}

template <typename T>
__global__ void third_layer_weights_update(T* third_layer_weights, T* third_layer_correction, T* previous_third_layer, int n_of_images, int n_of_third_layer_neurons, float alpha) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	while (tdx < n_of_third_layer_neurons + 1) {

		previous_third_layer[bdx*(n_of_third_layer_neurons + 1) + tdx] = third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + tdx];

		third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + tdx] -= alpha * third_layer_correction[bdx*(n_of_third_layer_neurons + 1) + tdx];



		tdx += OptNofThreads;
	}
}



template <typename T>
__global__ void third_layer_weights_back(T* third_layer_weights, T* previous_third_layer, int n_of_third_layer_neurons) {

	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	while (tdx < n_of_third_layer_neurons + 1) {


		third_layer_weights[bdx*(n_of_third_layer_neurons + 1) + tdx] = previous_third_layer[bdx*(n_of_third_layer_neurons + 1) + tdx];



		tdx += OptNofThreads;
	}
}







template <typename T>
__global__ void get_second_layer_correction(T* errors, T* first_layer_output, T* second_layer_output, T* second_layer_weights, T* third_layer_weights, T* second_layer_correction, T* error_multiplier, T* sigmoid_multiplier,
	int n_of_images, int n_of_second_layer_neurons, int n_of_third_layer_neurons, int n_of_models) {

	int tdx = threadIdx.x;
	int idx = threadIdx.x;
	int bdx = blockIdx.x;
	int bdy = blockIdx.y;

	__shared__ T additions[OptNofThreads];

	if (tdx == 0) {
		while (bdy < n_of_third_layer_neurons) {
			while (bdx < n_of_second_layer_neurons + 1) {
				second_layer_correction[bdy*(n_of_second_layer_neurons + 1) + bdx] = 0.0f;
				bdx += OptNofBlocksX;
			}
			if (bdx >= n_of_second_layer_neurons + 1) {
				bdx = blockIdx.x;
			}
			bdy += OptNofBlocksY;
		}
	}

	bdx = blockIdx.x;
	bdy = blockIdx.y;


	while (bdy < n_of_third_layer_neurons) {
		while (bdx < n_of_second_layer_neurons + 1) {
			while (idx < n_of_images) {

				additions[tdx] = 0;

				additions[tdx] = error_multiplier[bdy*n_of_images + idx] * (second_layer_output[bdy*n_of_images + idx] * (1 - second_layer_output[bdy*n_of_images + idx])) * first_layer_output[bdx*n_of_images + idx];

				if (bdx < n_of_second_layer_neurons) {
					sigmoid_multiplier[bdx*n_of_images + idx] += error_multiplier[bdy*n_of_images + idx] *
						second_layer_output[bdy*n_of_images + idx] * (1 - second_layer_output[bdy*n_of_images + idx])*second_layer_weights[bdy*(n_of_second_layer_neurons + 1) + bdx];
				}

				atomicAdd(&second_layer_correction[bdy*(n_of_second_layer_neurons + 1) + bdx], additions[tdx]);

				idx += OptNofThreads;
			}
			if (idx >= n_of_images) {
				idx = tdx;
			}
			bdx += OptNofBlocksX;
		}
		if (bdx >= n_of_second_layer_neurons + 1) {
			bdx = blockIdx.x;
		}
		bdy += OptNofBlocksY;
	}
}



template <typename T>
__global__ void second_layer_weights_update(T* second_layer_weights, T* correction, T* previous_second_layer, float alpha, int n_of_second_layer_neurons) {


	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	previous_second_layer[bdx*(n_of_second_layer_neurons + 1) + tdx] = second_layer_weights[bdx*(n_of_second_layer_neurons + 1) + tdx];

	second_layer_weights[bdx*(n_of_second_layer_neurons + 1) + tdx] -= alpha * correction[bdx*(n_of_second_layer_neurons + 1) + tdx];

}



template <typename T>
__global__ void second_layer_weights_back(T* second_layer_weights, T* previous_second_layer, int n_of_second_layer_neurons) {


	int tdx = threadIdx.x;
	int bdx = blockIdx.x;

	second_layer_weights[bdx*(n_of_second_layer_neurons + 1) + tdx] = previous_second_layer[bdx*(n_of_second_layer_neurons + 1) + tdx];

}







template <typename T>
__global__ void get_first_layer_correction(T* errors, T* images, T* first_layer_output, T* second_layer_output, T* third_layer_weights, T* second_layer_weights, T* first_layer_correction, T* sigmoid_multiplier,
	int n_of_images, int n_of_neurons, int n_of_second_layer_neurons, int n_of_third_layer_neurons, int n_of_models) {

	int idx = threadIdx.x;
	int tdx = threadIdx.x;
	int bdx = blockIdx.x;
	int bdy = blockIdx.y;

	if (tdx == 0) {
		while (bdy < n_of_second_layer_neurons) {
			while (bdx < n_of_neurons) {
				first_layer_correction[bdy * n_of_neurons + bdx] = 0.0f;
				bdx += OptNofBlocksX;
			}
			if (bdx >= n_of_neurons) {
				bdx = blockIdx.x;
			}
			bdy += OptNofBlocksY;
		}
	}

	__threadfence();
	bdx = blockIdx.x;
	bdy = blockIdx.y;

	while (bdy < n_of_second_layer_neurons) {
		while (bdx < n_of_neurons) {
			while (idx < n_of_images) {

				float main_multiplier;

				main_multiplier = first_layer_output[bdy*n_of_images + idx] * (1 - first_layer_output[bdy*n_of_images + idx])* sigmoid_multiplier[bdy*n_of_images + idx];
				atomicAdd(&first_layer_correction[bdy*n_of_neurons + bdx], main_multiplier*images[idx*n_of_neurons + bdx]);
				idx += OptNofThreads;
			}
			if (idx >= n_of_images) {
				idx = threadIdx.x;
			}
			bdx += OptNofBlocksX;
		}
		if (bdx >= n_of_neurons) {
			bdx = blockIdx.x;
		}
		bdy += OptNofBlocksY;
	}




}




template <typename T>
__global__ void first_layer_weights_update(T* first_layer_weights, T* first_layer_correction, T* previous_second_layer, float alpha, int n_of_neurons, int n_of_second_layer_neurons) {

	int idx = threadIdx.x;
	int bdx = blockIdx.x;

	while (bdx < n_of_second_layer_neurons) {
		while (idx < n_of_neurons) {


			previous_second_layer[bdx*n_of_neurons + idx] = first_layer_weights[bdx*n_of_neurons + idx];

			first_layer_weights[bdx*n_of_neurons + idx] -= alpha * first_layer_correction[bdx*n_of_neurons + idx];


			idx += OptNofThreads;
		}
		if (idx >= n_of_neurons) {
			idx = threadIdx.x;
		}
		bdx += OptNofBlocks;
	}


}


template <typename T>
__global__ void first_layer_weights_back(T* first_layer_weights, T* previous_second_layer, int n_of_neurons, int n_of_second_layer_neurons) {

	int idx = threadIdx.x;
	int bdx = blockIdx.x;

	while (bdx < n_of_second_layer_neurons) {
		while (idx < n_of_neurons) {


			first_layer_weights[bdx*n_of_neurons + idx] = previous_second_layer[bdx*n_of_neurons + idx];


			idx += OptNofThreads;
		}
		if (idx >= n_of_neurons) {
			idx = threadIdx.x;
		}
		bdx += OptNofBlocks;
	}
}