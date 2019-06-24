#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <time.h>
#include <algorithm>

#include <fstream>
#include <string>



#include "matrix_operations.cu"

#include "matrix_operations.cuh"


#define OptNofThreads 128
#define OptNofBlocks 128
#define OptNofBlocksX 32
#define OptNofBlocksY 32





__host__ void Error_check(cudaError_t x, std::string copy_type) {
	if (x != cudaSuccess) {
		std::cout << "There is an error in " + copy_type + ", error_type: " << x << std::endl;
	}
}



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "curand.h"

#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <time.h>
#include <algorithm>

#include <fstream>
#include <string>

#include "host_function.cuh"
#include "matrix_operations.cuh"

#define OptNofThreads 128
#define OptNofBlocks 128
#define OptNofBlocksX 32
#define OptNofBlocksY 32





template <typename T>
__host__ void train_executer(T* d_images, T* d_labels, T* d_first_layer_weights, T* d_second_layer_weights, T* d_third_layer_weights,
	int n_of_images, int n_of_models, int n_of_third_layer_neurons, int n_of_second_layer_neurons,
	int n_of_neurons, float alpha, float** first_layer_versions, float** second_layer_versions, float** third_layer_versions,
	int n_of_versions, float first_layer_dropout_rate, float second_layer_dropout_rate, float third_layer_dropout_rate, float second_layer_rate, float first_layer_rate, int n_of_steps) {

	int n_of_v = 0;



	float* d_posibilities, *d_third_layer_output, *d_second_layer_output, *d_first_layer_output, *d_sigmoid_multiplier, *d_error_multiplier, *d_third_layer_sums, *d_errors, *d_square_error;
	float* d_third_layer_correction, *d_second_layer_correction, *d_first_layer_correction, *d_previous_third_layer, *d_previous_second_layer, *d_previous_first_layer;


	Error_check(cudaMalloc((void**)&d_posibilities, sizeof(float)*n_of_images*n_of_models), " possiblitities allocating on device");
	Error_check(cudaMalloc((void**)&d_third_layer_output, sizeof(float)*n_of_images*n_of_models), " possiblitities all ocating on device");
	Error_check(cudaMalloc((void**)&d_second_layer_output, sizeof(float)*n_of_images*(n_of_third_layer_neurons + 1)), " possiblitities allocating on device");
	Error_check(cudaMalloc((void**)&d_first_layer_output, sizeof(float)*n_of_images*(n_of_second_layer_neurons + 1)), " first_output allocating on device");
	Error_check(cudaMalloc((void**)&d_sigmoid_multiplier, sizeof(float)*n_of_images*n_of_second_layer_neurons), " labels allocating on device");
	Error_check(cudaMalloc((void**)&d_error_multiplier, sizeof(float)*n_of_images*n_of_third_layer_neurons), " labels allocating on device");
	Error_check(cudaMalloc((void**)&d_third_layer_sums, sizeof(float)*n_of_images), " second_layer_sums allocating on device");
	Error_check(cudaMalloc((void**)&d_errors, sizeof(float)*n_of_images*n_of_models),  " errors allocating on device");
	Error_check(cudaMalloc((void**)&d_square_error, sizeof(float)*n_of_models), " square_error allocating on device");

	Error_check(cudaMalloc((void**)&d_previous_third_layer, sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1)), " third_layer_weights allocating on device");
	Error_check(cudaMalloc((void**)&d_previous_second_layer, sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1)), " second_layer_correction allocating on device");
	Error_check(cudaMalloc((void**)&d_previous_first_layer, sizeof(float)*n_of_second_layer_neurons*n_of_neurons), " firts_layer_correction allocating on device");

	Error_check(cudaMalloc((void**)&d_third_layer_correction, sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1)), " third_layer_weights allocating on device");
	Error_check(cudaMalloc((void**)&d_second_layer_correction, sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1)), " second_layer_correction allocating on device");
	Error_check(cudaMalloc((void**)&d_first_layer_correction, sizeof(float)*n_of_second_layer_neurons*n_of_neurons), " firts_layer_correction allocating on device");


	float previous_error = 999999.9 + 1;
	float current_error = 999999.9;

	float* square_error;
	square_error = (float*)malloc(sizeof(float)*n_of_models);

	T* dropped_first_layer, *dropped_second_layer, *dropped_third_layer;
	unsigned int* first_dropout_indices, *second_dropout_indices, *third_dropout_indices;
	int n_for_third_dropout = third_layer_dropout_rate * n_of_third_layer_neurons;
	int n_for_second_dropout = second_layer_dropout_rate * n_of_second_layer_neurons;
	int n_for_first_dropout = first_layer_dropout_rate * n_of_neurons;

	if (third_layer_dropout_rate > 0) {

		cudaMalloc((void**)&dropped_third_layer, sizeof(T)*n_for_third_dropout*n_of_models);
		cudaMalloc((void**)&third_dropout_indices, sizeof(unsigned int)*n_for_third_dropout*n_of_models);
	}

	if (second_layer_dropout_rate > 0) {

		cudaMalloc((void**)&dropped_second_layer, sizeof(T)*n_for_second_dropout*n_of_third_layer_neurons);
		cudaMalloc((void**)&second_dropout_indices, sizeof(unsigned int)*n_for_second_dropout*n_of_third_layer_neurons);
	}

	if (first_layer_dropout_rate > 0) {

		cudaMalloc((void**)&dropped_first_layer, sizeof(T)*n_for_first_dropout*n_of_second_layer_neurons);
		cudaMalloc((void**)&first_dropout_indices, sizeof(unsigned int)*n_for_first_dropout*n_of_second_layer_neurons);
	}


	curandGenerator_t gen;

	curandCreateGenerator(&gen,
			CURAND_RNG_PSEUDO_DEFAULT);

	curandSetPseudoRandomGeneratorSeed(gen,
			1234ULL);
	


	//while (previous_error > current_error) 
	for (int i = 0; i < n_of_steps; ++i) {

		if (first_layer_dropout_rate > 0) {
			curandGenerate(gen, first_dropout_indices, n_for_first_dropout*n_of_second_layer_neurons);
		}

		if (second_layer_dropout_rate > 0) {
			curandGenerate(gen, second_dropout_indices, n_for_second_dropout*n_of_third_layer_neurons);
		}

		if (third_layer_dropout_rate > 0) {
			curandGenerate(gen, third_dropout_indices, n_for_third_dropout*n_of_models);
		}




		//GENERATING OUTPUT
		//*****************
		get_first_layer_output << < n_of_second_layer_neurons + 1, OptNofThreads >> > (d_images, d_first_layer_weights, d_first_layer_output, d_sigmoid_multiplier, n_of_images, n_of_second_layer_neurons, n_of_neurons);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << " get_first_layer_output kernel launch without error " << std::endl;
			}
		}
		cudaDeviceSynchronize();



		get_second_layer_output << <  OptNofBlocks, OptNofThreads >> > (d_first_layer_output, d_second_layer_output, d_second_layer_weights, d_error_multiplier,
			n_of_images, n_of_second_layer_neurons, n_of_third_layer_neurons);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << " get_second_layer_output kernel launch without error " << std::endl;
			}
		}
		cudaDeviceSynchronize();



		get_third_layer_output << <  n_of_models, OptNofThreads >> > (d_third_layer_weights, d_second_layer_output, d_third_layer_output, d_third_layer_sums,
			n_of_images, n_of_third_layer_neurons);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << " get_second_layer_output kernel launch without error " << std::endl;
			}
		}
		cudaDeviceSynchronize();



		get_posibilities << <OptNofBlocks, n_of_models >> > (d_third_layer_output, d_third_layer_sums, d_posibilities, n_of_images, n_of_models);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << " get_posibilities kernel launch without error " << std::endl;
			}
		}
		cudaDeviceSynchronize();
		//*********************
		//END GENERATING OUTPUT


	
		previous_error = current_error;

		get_errors << < n_of_models, OptNofThreads >> > (d_posibilities, d_labels, d_errors, d_square_error, n_of_images, n_of_models, i);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << " get_errors kernel launch without error " << std::endl;
			}
		}
		cudaDeviceSynchronize();


		Error_check(cudaMemcpy(square_error, d_square_error, sizeof(float)*n_of_models, cudaMemcpyDeviceToHost), " copy square_error back to host");
		current_error = square_error[0];
		std::wcout << " \n\n\n " << current_error << "\n\n\n" << std::endl;

		if (current_error < 1 && previous_error < current_error) {

			third_layer_weights_back << <n_of_models, OptNofThreads >> > (d_third_layer_weights, d_previous_third_layer, n_of_third_layer_neurons);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << " get_third_layer_back kernel launch without error " << std::endl;
				}
			}
			cudaDeviceSynchronize();

			second_layer_weights_back<<<n_of_third_layer_neurons, n_of_second_layer_neurons + 1 >>>(d_second_layer_weights, d_previous_second_layer, n_of_second_layer_neurons);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << " get_second_layer_back kernel launch without error " << std::endl;
				}
			}
			cudaDeviceSynchronize();

			first_layer_weights_back << <OptNofBlocks, OptNofThreads >> > (d_first_layer_weights, d_previous_second_layer, n_of_neurons, n_of_second_layer_neurons);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << " get_first_layer_back kernel launch without error " << std::endl;
				}
			}
			cudaDeviceSynchronize();

			alpha /= 1.05;

		}



		//THIRD_LAYER_WEIGHTS UPDATE
		//***************************
		get_third_layer_correction << <n_of_models, OptNofThreads >> > (d_errors, d_second_layer_output, d_third_layer_correction, d_third_layer_weights, d_error_multiplier, n_of_images, n_of_third_layer_neurons, n_of_models);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << " get_third_layercorrection kernel launch without error " << std::endl;
			}
		}
		cudaDeviceSynchronize();

		if (third_layer_dropout_rate > 0) {
			//DROPOUT
			get_third_layer_dropout << <n_of_models, n_for_third_dropout >> > (d_third_layer_weights, dropped_third_layer, third_dropout_indices, n_of_third_layer_neurons, n_for_third_dropout);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << " get_third_layer_dropout kernel launch without error " << std::endl;
				}
			}
			cudaDeviceSynchronize();
			//DROPOUT BACK
		}

		third_layer_weights_update << <n_of_models, OptNofThreads >> > (d_third_layer_weights, d_third_layer_correction, d_previous_third_layer, n_of_images, n_of_third_layer_neurons, alpha);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << "third_layer_update kernel launch without error" << std::endl;
			}
		}
		cudaDeviceSynchronize();

		if (third_layer_dropout_rate > 0) {
			//DROPOUT BACK
			take_third_layer_dropout << <n_of_models, n_for_third_dropout >> > (d_third_layer_weights, dropped_third_layer, third_dropout_indices, n_of_third_layer_neurons, n_for_third_dropout);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << "third_layer_dropout_back kernel launch without error" << std::endl;
				}
			}
			cudaDeviceSynchronize();
			//END DROPOUT BACK
		}
		//*******************************
		//END THIRD_LAYER_WEIGHTS UPDATE





		//SECOND_LAYER_WEIGHTS UPDATE
		//**************************
		dim3 GridDim2(OptNofBlocksX, OptNofBlocksY);
		get_second_layer_correction << < GridDim2, OptNofThreads >> > (d_errors, d_first_layer_output, d_second_layer_output, d_second_layer_weights, d_third_layer_weights, d_second_layer_correction, d_error_multiplier, d_sigmoid_multiplier,
			n_of_images, n_of_second_layer_neurons, n_of_third_layer_neurons, n_of_models);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << "get_second_layer_correction kernel launch without error" << std::endl;
			}
		}
		cudaDeviceSynchronize();

		if (second_layer_dropout_rate > 0) {
			//DROPOUT
			get_second_layer_dropout << <n_of_third_layer_neurons, n_for_second_dropout >> > (d_second_layer_weights, dropped_second_layer, second_dropout_indices, n_of_second_layer_neurons, n_for_second_dropout);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << "get_second_layer_dropout kernel launch without error" << std::endl;
				}
			}
			cudaDeviceSynchronize();
			//DROPOUT BACK
		}

		second_layer_weights_update << < n_of_third_layer_neurons, n_of_second_layer_neurons + 1 >> > (d_second_layer_weights, d_second_layer_correction, d_previous_second_layer, alpha * second_layer_rate, n_of_second_layer_neurons);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << "second_layer_weights_update kernel launch without error" << std::endl;
			}
		}
		cudaDeviceSynchronize();

		if (second_layer_dropout_rate > 0) {
			//DROPOUT BACK
			take_second_layer_dropout << <n_of_third_layer_neurons, n_for_second_dropout >> > (d_second_layer_weights, dropped_second_layer, second_dropout_indices, n_of_second_layer_neurons, n_for_second_dropout);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << "second_layer_dropout_back kernel launch without error" << std::endl;
				}
			}
			cudaDeviceSynchronize();
			//END DROPOUT BACK
		}

		//**********************************
		//END OF SECOND_LAYER_WEIGHTS UPDATE



		//FIRST_LAYER_WEIGHTS UPDATE
		//**************************
		dim3 GridDim1(OptNofBlocksX, OptNofBlocksY);
		get_first_layer_correction << < GridDim1, OptNofThreads >> > (d_errors, d_images, d_first_layer_output, d_second_layer_output, d_third_layer_weights, d_second_layer_weights, d_first_layer_correction,
				d_sigmoid_multiplier, n_of_images, n_of_neurons, n_of_second_layer_neurons, n_of_third_layer_neurons, n_of_models);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << "get_first_layer_correction kernel launch without error" << std::endl;
			}
		}
		cudaDeviceSynchronize();

		if (first_layer_dropout_rate > 0) {
			//DROPOUT
			get_first_layer_dropout << <n_of_second_layer_neurons, n_for_first_dropout >> > (d_first_layer_weights, dropped_first_layer, first_dropout_indices, n_of_neurons, n_for_first_dropout);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << "get_first_layer_dropout kernel launch without error" << std::endl;
				}
			}
			cudaDeviceSynchronize();
			//DROPOUT BACK
		}


		first_layer_weights_update << <OptNofBlocks, OptNofThreads >> > (d_first_layer_weights, d_first_layer_correction, d_previous_first_layer, alpha * first_layer_rate, n_of_neurons, n_of_second_layer_neurons);
		{
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
			}
			else {
				std::cout << "first_layer_weights_update kernel launch without error" << std::endl;
			}
		}
		cudaDeviceSynchronize();

		if (first_layer_dropout_rate > 0) {
			//DROPOUT BACK
			take_first_layer_dropout << <n_of_second_layer_neurons, n_for_first_dropout >> > (d_first_layer_weights, dropped_first_layer, first_dropout_indices, n_of_neurons, n_for_first_dropout);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << "first_layer_dropout_back kernel launch without error" << std::endl;
				}
			}
			cudaDeviceSynchronize();
			//END DROPOUT BACK
		}

		
		//alpha *= 1.002;
		

		//******************************
		//END FIRST_LAYER_WEIGHTS UPDATE

		if ((i + 1) % (n_of_steps / n_of_versions) == 0) {
			std::cout << "\n\n\n copying weights \n\n\n" << std::endl;
			cudaMemcpy(first_layer_versions[n_of_v], d_first_layer_weights, sizeof(float)*n_of_second_layer_neurons*n_of_neurons, cudaMemcpyDeviceToHost);
			cudaMemcpy(second_layer_versions[n_of_v], d_second_layer_weights, sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1), cudaMemcpyDeviceToHost);
			cudaMemcpy(third_layer_versions[n_of_v], d_third_layer_weights, sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1), cudaMemcpyDeviceToHost);
			n_of_v++;
		}


	
	}

	cudaFree(d_first_layer_output);
	cudaFree(d_second_layer_output);
	cudaFree(d_third_layer_output);
	cudaFree(d_third_layer_sums);
	cudaFree(d_errors);
	cudaFree(d_third_layer_correction);
	cudaFree(d_second_layer_correction);
	cudaFree(d_first_layer_correction);



}



template <typename T>
__host__ void executer(T* d_images, T* d_first_layer_weights, T* d_second_layer_weights, T* d_third_layer_weights, T* d_posibilities, int n_of_images, int n_of_neurons,
	int n_of_second_layer_neurons, int n_of_third_layer_neurons, int n_of_models) {


	float* d_first_layer_output, *d_second_layer_output, *d_third_layer_output, *d_third_layer_sums, *d_error_multiplier, *d_sigmoid_multiplier;

	Error_check(cudaMalloc((void**)&d_first_layer_output, sizeof(float)*(n_of_second_layer_neurons + 1)*n_of_images), "first_layer_output for executing allocating on device");
	Error_check(cudaMalloc((void**)&d_second_layer_output, sizeof(float)*(n_of_third_layer_neurons + 1)*n_of_images), "second_layer_output for executing allocating on device");
	Error_check(cudaMalloc((void**)&d_third_layer_output, sizeof(float)*n_of_images*n_of_models),"third_layer_output for executing allocating on device" );
	Error_check(cudaMalloc((void**)&d_third_layer_sums, sizeof(float)*n_of_images), "third_layer_sums for executing allocating on device");
	Error_check(cudaMalloc((void**)&d_sigmoid_multiplier, sizeof(float)*n_of_images*n_of_second_layer_neurons), "sigmoid_multiplier for executing allocating on device");
	Error_check(cudaMalloc((void**)&d_error_multiplier, sizeof(float)*n_of_images*n_of_third_layer_neurons), "error_multiplier for executing allocating on device");


	//GENERATING OUTPUT
	//*****************

	get_first_layer_output << < OptNofBlocks, OptNofThreads >> > (d_images, d_first_layer_weights, d_first_layer_output, d_sigmoid_multiplier, n_of_images, n_of_second_layer_neurons, n_of_neurons);
	{
		cudaError err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cout << cudaGetErrorString(err) << std::endl;
		}
		else {
			std::cout << "get_first_layer_output kernel launch without error" << std::endl;
		}
	}
	cudaDeviceSynchronize();



	get_second_layer_output << <  OptNofBlocks, OptNofThreads >> > (d_first_layer_output, d_second_layer_output, d_second_layer_weights, d_error_multiplier,
		n_of_images, n_of_second_layer_neurons, n_of_third_layer_neurons);
	{
		cudaError err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cout << cudaGetErrorString(err) << std::endl;
		}
		else {
			std::cout << "get_second_layer_output kernel launch without error" << std::endl;
		}
	}
	cudaDeviceSynchronize();



	get_third_layer_output << <  n_of_models, OptNofThreads >> > (d_third_layer_weights, d_second_layer_output, d_third_layer_output, d_third_layer_sums,
		n_of_images, n_of_third_layer_neurons);
	{
		cudaError err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cout << cudaGetErrorString(err) << std::endl;
		}
		else {
			std::cout << "get_second_layer_output kernel launch without error" << std::endl;
		}
	}
	cudaDeviceSynchronize();



	get_posibilities << <OptNofBlocks, n_of_models >> > (d_third_layer_output, d_third_layer_sums, d_posibilities, n_of_images, n_of_models);
	{
		cudaError err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cout << cudaGetErrorString(err) << std::endl;
		}
		else {
			std::cout << "get_posibilities kernel launch without error" << std::endl;
		}
	}
	cudaDeviceSynchronize();
	//*********************
	//END GENERATING OUTPUT

	cudaFree(d_first_layer_output);
	cudaFree(d_second_layer_output);
	cudaFree(d_third_layer_output);
	cudaFree(d_third_layer_sums);
	cudaFree(d_sigmoid_multiplier);
	cudaFree(d_error_multiplier);

}










































int main()
{



	//DATA EXTRACTION
	//opening data file

	std::ifstream input;
	input.open("16x16_digits.txt");

	//an array for numbers
	int* all_numbers;

	int n_of_numbers = 0;

	//file reading and filling umbers array
	//count number of numbers in file
	if (input.is_open()) {
		std::cout << "Reading from file...." << std::endl;
		float number;
		while (input >> number) {
			++n_of_numbers;
		}
	}



	//reading all th numbers from file to an array
	all_numbers = (int*)malloc(n_of_numbers * sizeof(int));
	if (input.is_open()) {
		input.close();

		input.open("16x16_digits.txt");
		int i = 0;
		float number;
		while (input >> number) {
			all_numbers[i] = (int)number;
			++i;
		}
	}

	std::cout << "File readed" << std::endl;




	//DAPA PREPARATION
	//define metrices
	const int n_of_images = 1593;

	const int n_of_neurons = 257;

	const int n_of_models = 10;

	const int n_of_second_layer_neurons = 100;

	const int n_of_third_layer_neurons = 100;

	const int n_of_versions = 100;

	const int n_of_steps = 300;


	float alpha = 0.0005f;

	const float second_rate = 0.2;
	const float first_rate = 4.0 / n_of_third_layer_neurons;


	//split ratio
	float split_ratio = 0.2;
	float dropout1 = 0.2f;
	float dropout2 = 0.2f;
	float dropout3 = 0.15f;
	int n_test_images = 0, n_train_images = 0;



	float indices[n_of_images];

	for (int i = 0; i < n_of_images; ++i) {
		indices[i] = (float)(rand() % n_of_images) / n_of_images;
		if (indices[i] > split_ratio) {
			++n_train_images;
		}
		else {
			++n_test_images;
		}
	}



	//create and allocate memory for images and labels pointers
	float *train_images, *test_images, *train_labels, *test_labels;
	train_images = (float*)malloc(sizeof(float)*n_train_images*n_of_neurons);
	test_images = (float*)malloc(sizeof(float)*n_test_images*n_of_neurons);
	train_labels = (float*)malloc(sizeof(float)*n_of_models*n_train_images);
	test_labels = (float*)malloc(sizeof(float)*n_of_models*n_test_images);

	int train_count = 0, test_count = 0;


	//fill images and labels arrays
	for (int i = 0; i < n_of_images; ++i) {
		int index_ = i * 266;

		if (indices[i] > split_ratio) {
			//filling images
			for (int j = 0; j < 257; ++j) {
				if (j == 256) {
					train_images[train_count*n_of_neurons + j] = (float)1;
				}
				else {
					train_images[train_count*n_of_neurons + j] = (float)all_numbers[index_ + j];
				}
			}
			//filling labels
			for (int k = 0; k < 10; ++k) {
				train_labels[k*n_train_images + train_count] = (float)all_numbers[index_ + 256 + k];
			}
			++train_count;
		}
		else {
			for (int j = 0; j < 257; ++j) {
				if (j == 256) {
					test_images[test_count*n_of_neurons + j] = (float)1;
				}
				else {
					test_images[test_count*n_of_neurons + j] = (float)all_numbers[index_ + j];
				}
			}
			//filling labels
			for (int k = 0; k < 10; ++k) {
				test_labels[k*n_test_images + test_count] = (float)all_numbers[index_ + 256 + k];
			}
			++test_count;
		}
	}




	//Creating pointers for weights on the host
	float* first_layer_weights, *second_layer_weights, *third_layer_weights;
	first_layer_weights = (float*)malloc(sizeof(float)*n_of_second_layer_neurons*n_of_neurons);
	second_layer_weights = (float*)malloc(sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1));
	third_layer_weights = (float*)malloc(sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1));





	srand(2);
	//*****************FILLING FIRST AND SECOND LAYER_WEIGHTS**********************
	for (int i = 0; i < n_of_second_layer_neurons; ++i) {

		for (int j = 0; j < n_of_neurons; ++j) {
			first_layer_weights[i*n_of_neurons + j] = ((float)(rand() % 100) / 100 - 0.5);
		}

		for (int k = 0; k < n_of_third_layer_neurons; ++k) {
			second_layer_weights[k*(n_of_second_layer_neurons + 1) + i] = ((float)(rand() % 100) / 100 - 0.5);
		}
	}
	for (int i = 0; i < n_of_third_layer_neurons + 1; ++i) {
		if (i < n_of_third_layer_neurons) {
			second_layer_weights[i*(n_of_second_layer_neurons + 1) + n_of_second_layer_neurons] = ((float)(rand() % 100) / 100 - 0.5);
		}
		for (int j = 0; j < n_of_models; ++j) {
			third_layer_weights[j*(n_of_third_layer_neurons + 1) + i] = ((float)(rand() % 100) / 100 - 0.5);
		}
	}
	//*****************************************************************************










	//Creating pointers for data on the device
	float *d_train_images, *d_test_images, *d_train_labels, *d_train_posibilities, *d_train_errors, *d_train_third_layer_output, *d_train_second_layer_output, *d_train_third_layer_sums, *d_train_first_layer_output, *d_train_sigmoid_multiplier, *d_train_error_multiplier;
	float *d_first_layer_weights, *d_second_layer_weights, *d_third_layer_weights, *d_third_layer_correction, *d_second_layer_correction, *d_first_layer_correction, *d_square_error;






	//Allocating memory for all data on the device and copy that data to device
	Error_check(cudaMalloc((void**)&d_train_images, sizeof(float)*n_train_images*n_of_neurons), "images allocating on device");
	Error_check(cudaMalloc((void**)&d_train_labels, sizeof(float)*n_train_images*n_of_models), "images allocating on device");
	Error_check(cudaMalloc((void**)&d_test_images, sizeof(float)*n_test_images*n_of_neurons), "images allocating on device");

	//Copy images and labels on the device
	Error_check(cudaMemcpy(d_train_images, train_images, sizeof(float)*n_train_images*n_of_neurons, cudaMemcpyHostToDevice), "copy images on device");
	Error_check(cudaMemcpy(d_train_labels, train_labels, sizeof(float)*n_train_images*n_of_models, cudaMemcpyHostToDevice), "copy images on device");
	Error_check(cudaMemcpy(d_test_images, test_images, sizeof(float)*n_test_images*n_of_neurons, cudaMemcpyHostToDevice), "copy images on device");

	//Allocating  weights of neurons and corrections for those wights 
	Error_check(cudaMalloc((void**)&d_first_layer_weights, sizeof(float)*n_of_second_layer_neurons*n_of_neurons), "first_layer_weights allocating on device");
	Error_check(cudaMalloc((void**)&d_second_layer_weights, sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1)), "second_layer_weights allocating on device");
	Error_check(cudaMalloc((void**)&d_third_layer_weights, sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1)), "third_layer_weights allocating on device");


	//Copy first_layer_weights and second_layer_weights on device allocated memory
	Error_check(cudaMemcpy(d_first_layer_weights, first_layer_weights, sizeof(float)*n_of_second_layer_neurons*n_of_neurons, cudaMemcpyHostToDevice), "copy first_layer_weights on device");
	Error_check(cudaMemcpy(d_second_layer_weights, second_layer_weights, sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1), cudaMemcpyHostToDevice), "copy second_layer_weights on device");
	Error_check(cudaMemcpy(d_third_layer_weights, third_layer_weights, sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1), cudaMemcpyHostToDevice), "copy second_layer_weights on device");


	


	float* second_layer_versions[n_of_versions + 1];
	float* first_layer_versions[n_of_versions + 1];
	float* third_layer_versions[n_of_versions + 1];

	for (int i = 0; i < n_of_versions + 1; ++i) {
		third_layer_versions[i] = (float*)malloc(sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1));
		second_layer_versions[i] = (float*)malloc(sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1));
		first_layer_versions[i] = (float*)malloc(sizeof(float)*n_of_second_layer_neurons*n_of_neurons);
	}


	//FREE HOST MEMORY
	free(all_numbers);




	clock_t  start_clock = clock();

	train_executer(d_train_images, d_train_labels, d_first_layer_weights, d_second_layer_weights, d_third_layer_weights, n_train_images, n_of_models,
		n_of_third_layer_neurons, n_of_second_layer_neurons, n_of_neurons, alpha, first_layer_versions, second_layer_versions, third_layer_versions, n_of_versions, dropout1, dropout2, dropout3, second_rate, first_rate, n_of_steps);


	clock_t  end_clock = clock();


	//FREE GPU MEMORY
	cudaFree(d_train_images);
	cudaFree(d_train_labels);


	float time = (float)(end_clock - start_clock) / CLOCKS_PER_SEC;

	



	float **all_posibilities;
	all_posibilities = (float**)malloc(sizeof(float*)*n_of_versions);
	for (int i = 0; i < n_of_versions; ++i) {
		cudaMalloc((void**)&all_posibilities[i], sizeof(float)*n_test_images*n_of_models);
	}


	float test_results[n_of_versions];

	std::cout << "\n\n\n\n\n and the results are:" << std::endl;




	for (int i = 1; i < n_of_versions; ++i) {

		float* d_first_layer_weights1, *d_second_layer_weights1, *d_third_layer_weights1;

		cudaMalloc((void**)&d_first_layer_weights1, sizeof(float)*n_of_second_layer_neurons*n_of_neurons);
		cudaMemcpy(d_first_layer_weights1, first_layer_versions[i], sizeof(float)*n_of_second_layer_neurons*n_of_neurons, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_second_layer_weights1, sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1));
		cudaMemcpy(d_second_layer_weights1, second_layer_versions[i], sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_third_layer_weights1, sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1));
		cudaMemcpy(d_third_layer_weights1, third_layer_versions[i], sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1), cudaMemcpyHostToDevice);




		executer(d_test_images, d_first_layer_weights1, d_second_layer_weights1, d_third_layer_weights1, all_posibilities[i - 1], n_test_images, n_of_neurons,
			n_of_second_layer_neurons, n_of_third_layer_neurons, n_of_models);

		{

			float* my_labels, *my_test_labels;
			my_labels = (float*)malloc(sizeof(float)*n_test_images);
			my_test_labels = (float*)malloc(sizeof(float)*n_test_images);

			float *test_posibilities;
			test_posibilities = (float*)malloc(sizeof(float)*n_of_models*n_test_images);
			cudaMemcpy(test_posibilities, all_posibilities[i - 1], sizeof(float)*n_of_models*n_test_images, cudaMemcpyDeviceToHost);

			for (int j = 0; j < n_test_images; ++j) {
				float max = 0;
				float max_index;

				for (int k = 0; k < n_of_models; ++k) {

					if (test_posibilities[k*n_test_images + j] > max) {
						max = test_posibilities[k*n_test_images + j];

						max_index = k;
					}
					if (test_labels[k*n_test_images + j] == 1) {
						my_test_labels[j] = k;
					}

				}
				my_labels[j] = max_index;

			}

			float n_of_true = 0;



			for (int j = 0; j < n_test_images; ++j) {
				if (my_labels[j] == my_test_labels[j]) {
					++n_of_true;
				}

			}

			float result;
			result = (float)n_of_true / n_test_images;

			test_results[i] = result;


			free(my_labels);
			free(test_posibilities);
		}

		cudaFree(d_first_layer_weights1);
		cudaFree(d_second_layer_weights1);
		cudaFree(d_third_layer_weights1);

	}

	std::cout << " \n\n\n" << std::endl;

	

	for (int i = 1; i < n_of_versions; ++i) {
		std::cout << test_results[i] << std::endl;
	}

	std::cout << " \n" << std::endl;

	std::cout << "GPU time of execution: " << time - 3 << std::endl;

}






