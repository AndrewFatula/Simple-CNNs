#ifndef MATRIX_OPERATIONS_CUH
#define MATRIX_OPERATIONS_CUH

template <typename T>
__global__ void get_first_layer_dropout(T* first_layer_weights, T* dropped_first_layer, unsigned int* first_dropout_indices, int n_of_neurons, int n_for_first_dropout);

template <typename T>
__global__ void get_second_layer_dropout(T* second_layer_weights, T* dropped_second_layer, unsigned int* second_dropout_indices, int n_of_second_layer_neurons, int n_for_dropout);

template <typename T>
__global__ void get_third_layer_dropout(T* third_layer_weights, T* dropped_third_layer, unsigned int* third_dropout_indices, int n_of_third_layer_neurons, int n_for_third_dropout);

template <typename T>
__global__ void take_first_layer_dropout(T* first_layer_weights, T* dropped_first_layer, unsigned int* first_dropout_indices, int n_of_neurons, int n_for_first_dropout);

template <typename T>
__global__ void take_second_layer_dropout(T* second_layer_weights, T* dropped_second_layer, unsigned int* second_dropout_indices, int n_of_second_layer_neurons, int n_for_dropout);

template <typename T>
__global__ void take_third_layer_dropout(T* third_layer_weights, T* dropped_third_layer, unsigned int* third_dropout_indices, int n_of_third_layer_neurons, int n_for_third_dropout);

template<typename T>
__global__ void get_first_layer_output(T* images, T* first_layer_weights, T* first_layer_output, T* sigmoid_multiplier, int n_of_images, int n_of_second_layer_neurons, int n_of_neurons);

template <typename T>
__global__ void get_second_layer_output(T* first_layer_output, T* second_layer_output, T* second_layer_weights, T* error_multiplier, int n_of_images, int n_of_second_layer_neurons, int n_of_third_layer_neurons);

template <typename T>
__global__ void get_third_layer_output(T* third_layer_weights, T* second_layer_output, T* third_layer_output, T* third_layer_sums, int n_of_images, int n_of_third_layer_neurons);

template <typename T>
__global__ void get_posibilities(T* third_layer_output, T* third_layer_sums, T* posibilities, int n_of_images, int n_of_models);

template <typename T>
__global__ void get_errors(T* posibilities, T* labels, T* errors, T* square_error, int n_of_images, int n_of_models, int i);

template <typename T>
__global__ void get_third_layer_correction(T* errors, T* second_layer_output, T* third_layer_correction, T* third_layer_weights, T* error_multiplier, int n_of_images, int n_of_third_layer_neurons, int n_of_models);

template <typename T>
__global__ void third_layer_weights_update(T* third_layer_weights, T* third_layer_correction, int n_of_images, int n_of_third_layer_neurons, float alpha);

template <typename T>
__global__ void get_second_layer_correction(T* errors, T* first_layer_output, T* second_layer_output, T* second_layer_weights, T* third_layer_weights, T* second_layer_correction, T* error_multiplier, T* sigmoid_multiplier,
	int n_of_images, int n_of_second_layer_neurons, int n_of_third_layer_neurons, int n_of_models);

template <typename T>
__global__ void second_layer_weights_update(T* second_layer_weights, T* correction, float alpha, int n_of_second_layer_neurons);

template <typename T>
__global__ void get_first_layer_correction(T* errors, T* images, T* first_layer_output, T* second_layer_output, T* third_layer_weights, T* second_layer_weights, T* first_layer_correction, T* sigmoid_multiplier,
	int n_of_images, int n_of_neurons, int n_of_second_layer_neurons, int n_of_third_layer_neurons, int n_of_models);

template <typename T>
__global__ void first_layer_weights_update(T* first_layer_weights, T* first_layer_correction, float alpha, int n_of_neurons, int n_of_second_layer_neurons);

#endif