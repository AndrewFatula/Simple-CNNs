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
__host__ void train_executer(T* d_images, T* d_labels, T* d_posibilities, T* d_errors, T* d_third_layer_sums, T* d_third_layer_output, T* d_second_layer_output, T* d_first_layer_output,
	T* d_sigmoid_multiplier, T* d_error_multiplier, T* d_first_layer_weights,
	T* d_second_layer_weights, T* d_third_layer_weights, T* d_third_layer_correction, T* d_second_layer_correction, T* d_first_layer_correction, T* d_square_error, int n_of_images,
	int n_of_models, int n_of_third_layer_neurons, int n_of_second_layer_neurons,
	int n_of_neurons, float alpha, float* square_error, float previous_error, float current_error, float** first_layer_versions, float** second_layer_versions, float** third_layer_versions,
	int n_of_versions, float dropout_rate, bool execute) {


	int n_of_v = 0;

	int n_for_dropout = dropout_rate * n_of_second_layer_neurons;

	//T* dropped_second_layer;
	//cudaMalloc((void**)&dropped_second_layer, sizeof(T)*n_for_dropout*n_of_models);

	//unsigned int* second_layer_indices;
	//cudaMalloc((void**)&second_layer_indices, sizeof(unsigned int)*n_for_dropout*n_of_models);


	//curandGenerator_t gen;


	//curandCreateGenerator(&gen,
	//	CURAND_RNG_PSEUDO_DEFAULT);

	//curandSetPseudoRandomGeneratorSeed(gen,
	//	1234ULL);



	//while (previous_error > current_error) 
	for (int i = 0; i < 500; ++i) {

		//curandGenerate(gen, second_layer_indices, n_for_dropout*n_of_models);


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


		if (execute) { break; }
		else {

			//THIRD_LAYER_WEIGHTS UPDATE
			//***************************
			previous_error = current_error;

			get_errors << < n_of_models, OptNofThreads >> > (d_posibilities, d_labels, d_errors, d_square_error, n_of_images, n_of_models);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << "get_errors kernel launch without error" << std::endl;
				}
			}
			cudaDeviceSynchronize();


			Error_check(cudaMemcpy(square_error, d_square_error, sizeof(float)*n_of_models, cudaMemcpyDeviceToHost), " copy square_error back to host");
			current_error = square_error[0];
			std::wcout << " \n\n\n " << current_error << std::endl;


			get_third_layer_correction << <n_of_models, OptNofThreads >> > (d_errors, d_second_layer_output, d_third_layer_correction, d_third_layer_weights, d_error_multiplier, n_of_images, n_of_third_layer_neurons, n_of_models);
			{
				cudaError err = cudaGetLastError();
				if (err != cudaSuccess) {
					std::cout << cudaGetErrorString(err) << std::endl;
				}
				else {
					std::cout << "get_third_layercorrection kernel launch without error" << std::endl;
				}
			}
			cudaDeviceSynchronize();



			third_layer_weights_update << <n_of_models, OptNofThreads >> > (d_third_layer_weights, d_third_layer_correction, n_of_images, n_of_third_layer_neurons, 0.05* alpha);
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


			second_layer_weights_update << < n_of_third_layer_neurons, n_of_second_layer_neurons + 1 >> > (d_second_layer_weights, d_second_layer_correction, alpha * 0.2, n_of_second_layer_neurons);
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


			first_layer_weights_update << <OptNofBlocks, OptNofThreads >> > (d_first_layer_weights, d_first_layer_correction, alpha, n_of_neurons, n_of_second_layer_neurons);
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




			//******************************
			//END FIRST_LAYER_WEIGHTS UPDATE

			if ((i + 1) % (500 / n_of_versions) == 0) {
				cudaMemcpy(first_layer_versions[n_of_v], d_first_layer_weights, sizeof(float)*n_of_second_layer_neurons*n_of_neurons, cudaMemcpyDeviceToHost);
				cudaMemcpy(second_layer_versions[n_of_v], d_second_layer_weights, sizeof(float)*n_of_third_layer_neurons*(n_of_second_layer_neurons + 1), cudaMemcpyDeviceToHost);
				cudaMemcpy(third_layer_versions[n_of_v], d_third_layer_weights, sizeof(float)*n_of_models*(n_of_third_layer_neurons + 1), cudaMemcpyDeviceToHost);
				n_of_v++;
			}


		}
	}



}



template <typename T>
__host__ void executer(T* d_images, T* d_first_layer_weights, T* d_second_layer_weights, T* d_third_layer_weights, T* d_posibilities, int n_of_images, int n_of_neurons,
	int n_of_second_layer_neurons, int n_of_third_layer_neurons, int n_of_models) {


	float* d_first_layer_output, *d_second_layer_output, *d_third_layer_output, *d_third_layer_sums, *d_error_multiplier, *d_sigmoid_multiplier;

	cudaMalloc((void**)&d_first_layer_output, sizeof(float)*(n_of_second_layer_neurons + 1)*n_of_images);
	cudaMalloc((void**)&d_second_layer_output, sizeof(float)*(n_of_third_layer_neurons + 1)*n_of_models);
	cudaMalloc((void**)&d_third_layer_output, sizeof(float)*n_of_images*n_of_models);
	cudaMalloc((void**)&d_third_layer_sums, sizeof(float)*n_of_images);
	Error_check(cudaMalloc((void**)&d_sigmoid_multiplier, sizeof(float)*n_of_images*n_of_second_layer_neurons), "labels allocating on device");
	Error_check(cudaMalloc((void**)&d_error_multiplier, sizeof(float)*n_of_images*n_of_third_layer_neurons), "labels allocating on device");


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
	cudaFree(d_third_layer_sums);


}











