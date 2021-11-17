#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//bibliotecas para gerar numeros aleatorios
#include <random>
#include <time.h>

#define arraySize 10

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	
	int a[arraySize] = {};
	int b[arraySize] = {};
	int c[arraySize] = {};

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	srand(time(0));
	//populate the arrays A and B
	for (int i = 0; i < arraySize; i++) {
		
			a[i] = rand() % 10;
			b[i] = rand() % 10;
		}
	
	/* VETOR a */
	printf("VETOR A\n");
	for (int i = 0; i < arraySize; i++) {
		printf("%d\t", a[i]);
	}

	/* VETOR B */
	printf("\nVETOR B\n");
	for (int i = 0; i < arraySize; i++) {
		printf("%d\t", b[i]);
	}

	// Alocar espaço na memória do device
	cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copia os vetores do host para a device
	cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



	// Executar o kernel
	addKernel << <arraySize, 1 >> > (dev_c, dev_a, dev_b);

	// Verificar se o kernel foi executado corretamente
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Espera o kernel terminar e retorna quaisquer erros encontrados durante a execução
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copia o resultado do device para a memória do host.
	cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	/* VETOR C */
	printf("\nVETOR C\n");
	for (int i = 0; i < arraySize; i++) {
		printf("%d\t", c[i]);
	}

	// Limpa a memória
Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}