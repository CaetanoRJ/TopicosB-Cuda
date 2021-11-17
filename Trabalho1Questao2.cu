#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernelA() {
    // TEORIA : https://anuradha-15.medium.com/cuda-thread-indexing-fb9910cba084
    //blockDim.x — number of threads in the x dimension if the grid (eg:4)
    //blockIdx.x — block’s index in x dimension
    //ThreadIdx.x — thread’s index in x dimension

        //1D grid, 1d BLOCK
        //threadId = (blockIdx.x * blockDim.x) + threadIdx.x
        //Let’s check the equation for Thread(2, 0) in Block(1, 0).
        //Thread ID = (1 * 3) + 2 = 3 + 2 = 5

    int globalThreadId = (blockIdx.x * blockDim.x) + threadIdx.x;

    printf("My threadIdx.x is %d, blockIdx.x is %d, blockDim.x is %d, Global thread id is %d\n",
        threadIdx.x, blockIdx.x, blockDim.x, globalThreadId);

    //1D grid of 2D blocks
    //threadId = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x

    //2D grid of 1D blocks
    //threadId = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x
}

int main()
{
 
    cudaSetDevice(0);
    kernelA << <4, 3 >> > ();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}