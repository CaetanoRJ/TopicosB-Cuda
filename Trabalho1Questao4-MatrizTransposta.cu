#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>


#include <random>
#include <time.h>

#define N 5

__global__ void transposta(float* M, float* T) {


	int C = blockDim.x * blockIdx.x + threadIdx.x;
	int L = blockDim.y * blockIdx.y + threadIdx.y;

	if (C < N && L < N) {
		T[C * N + L] = M[C + L * N];
	}
}



int main(void) {



	float* M_h;
	float* T_h;
	float* M_d;
	float* T_d;

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





	int blocksizeX = N;
	int blocksizeY = N;
	int A = (N + blocksizeX - 1) / blocksizeX;
	int B = (N + blocksizeY - 1) / blocksizeY;

	dim3 block(blocksizeX, blocksizeY);
	dim3 grid(A, B);



	transposta << <grid, block >> > (M_d, T_d);

	cudaMemcpy(T_h, T_d, size, cudaMemcpyDeviceToHost);

	printf("\nTransposta\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			int num = T_h[i * N + j];
			printf(" %d ", num);
		}
		printf("\n");
	}




	cudaFree(T_d);
	cudaFree(M_d);
	return 0;
}