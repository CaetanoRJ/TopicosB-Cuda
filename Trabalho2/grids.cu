#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdio.h>
#include <random>
#include <time.h>
#include <math.h>
__global__ void grids()
{
    int  threadRowID, threadColId;

    threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
    threadColId = blockIdx.y * blockDim.y + threadIdx.y;

    /* ------------------------------------
       Print the thread's 2 dim grid ID
       ------------------------------------ */
    printf("Block: (%d,%d) Thread: (%d,%d) -> Row/Col = (%d,%d)\n",
        blockIdx.x, blockIdx.y,
        threadIdx.x, threadIdx.y,
        threadRowID, threadColId);
}

int main()
{
    dim3 blockShape = dim3(2, 2);
    dim3 gridShape = dim3(2, 2);

    grids << < gridShape, blockShape >> > ();  // Launch a 2 dim grid of threads  

  

    cudaDeviceSynchronize();

    return 0;
}