#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include "cuda_runtime.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <time.h>
#include <math.h>


#define BlockSize 32

//N é o tamanho da MATRIZ
#define N 8

__global__ void deslocaColuna(float* M, float* T) {

	//Coluna
	int C = blockDim.x * blockIdx.x + threadIdx.x;
	//Linha
	int L = blockDim.y * blockIdx.y + threadIdx.y;

	if (C < N && L < N) {

		if (C >= 0 ) { //Desloca as colunas da matriz
			T[C + L * N + 1] = M[C + L * N];
		}
		if (C == N - 1) { //Bordas da Matriz
			printf("\n BORDAS ## Row/Col = (%d,%d)", L, C);
			//transforma o primeiro elemento da linha para ser o mesmo elementa dessa linha porém na ultima coluna
			T[L * N] = M[C + L * N];
		}
		
		
	}
}



int main(void) {

	float* M_h;
	float* T_h;
	float* M_d;
	float* T_d;

	float time;
	cudaEvent_t start, stop;

	size_t size = N * N * sizeof(float);

	cudaMallocHost((float**)&M_h, size);
	T_h = (float*)malloc(size);
	cudaMalloc((float**)&M_d, size);

	// init matrix
	for (int i = 0; i < N * N; ++i) {
		M_h[i] = rand() % 10;
	}

	cudaMemcpyAsync(M_d, M_h, size, cudaMemcpyHostToDevice);
	cudaMalloc((float**)&T_d, size);
	cudaMemset(T_d, 0, size);

	printf("\nMATRIZ GERADA\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			int num = M_h[i * N + j];
			printf(" %d ", num);
		}
		printf("\n");
	}


	//two dimension threads
	dim3 dimBlock(BlockSize, BlockSize);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	deslocaColuna << <dimGrid, dimBlock >> > (M_d, T_d);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaMemcpy(T_h, T_d, size, cudaMemcpyDeviceToHost);
	printf("\n\ndeslocaColuna\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			int num = T_h[i * N + j];
			printf(" %d ", num);
		}
		printf("\n");
	}



	printf("\n Tempo para gerar a matriz:  %3.5f ms \n", time);


	cudaFree(T_d);
	cudaFree(M_d);
	return 0;
}