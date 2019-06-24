#ifndef HOST_FUNCTION_CUH
#define HOST_FUNCTION_CUH

template <typename T>
__host__ void train_executer(T* d_images, T* d_labels, T* d_posibilities, T* d_errors, T* d_third_layer_sums, T* d_third_layer_output, T* d_second_layer_output, T* d_first_layer_output,
	T* d_sigmoid_multiplier, T* d_error_multiplier, T* d_first_layer_weights,
	T* d_second_layer_weights, T* d_third_layer_weights, T* d_third_layer_correction, T* d_second_layer_correction, T* d_first_layer_correction, T* d_square_error,
	int n_of_images, int n_of_models, int n_of_third_layer_neurons, int n_of_second_layer_neurons,
	int n_of_neurons, float alpha, float* square_error, float previous_error, float current_error, float** first_layer_versions, float** second_layer_versions,
	float** third_layer_versions, int n_of_versions, float first_dropout_rate, float second_layer_dropout_rate, float third_layer_dropout_rate, float second_layer_rate, float first_layer_rate, int n_of_steps);

template <typename T>
__host__ void executer(T* d_images, T* d_first_layer_weights, T* d_second_layer_weights, T* d_third_layer_weights, T* d_posibilities, int n_of_images, int n_of_neurons,
	int n_of_second_layer_neurons, int n_of_third_layer_neurons, int n_of_models);


#endif