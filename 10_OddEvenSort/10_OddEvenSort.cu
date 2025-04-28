/* This program will sort a vector of size 2048 in 1 ThreadBlock.
*
*  +---------+    +---------+
*  |876543210| -> |012345678|
*  +---------+    +---------+
*
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <pac_helper.h>
#include <algorithm> 
using namespace std;

#define BLOCKSIZE   2048

    
// Make a OddEvenSort using GlobalMemory
__global__ void cudaOddEvenSortGlobalMemory(int* input, const int size)
{

    const int idEven = threadIdx.x * 2;
    const int idOdd = threadIdx.x * 2 + 1;
    int tmp;

    for (int i = 0; i < BLOCKSIZE / 2; i++) // Manual unrolling by odd and even phase -> size / 2
    {
        if (input[idEven] > input[idEven + 1])
        {
            tmp = input[idEven];
            input[idEven] = input[idEven + 1];
            input[idEven + 1] = tmp;
        }
        __syncthreads();
        if (idOdd + 1 < BLOCKSIZE && input[idOdd] > input[idOdd + 1])
        {
            tmp = input[idOdd];
            input[idOdd] = input[idOdd + 1];
            input[idOdd + 1] = tmp;
        }
        __syncthreads();
    }
}


// Make a OddEvenSort using SharedMemory
__global__ void cudaOddEvenSortSharedMemory(int* input, const int size)
{
    //ToDo
}


// Make a OddEvenSort using SharedMemory no bank conflicts
__global__ void cudaOddEvenSortSharedMemoryNoConflict(int* input, const int size)
{
    //ToDo
}


int main(void)
{
    // Define the size of the vector: 1048576 elements
    int const N = BLOCKSIZE;

    // Allocate and prepare inputs
    int* hostVector = new int[N];
    int* managedVector1;
    int* managedVector2;
    int* managedVector3;
    gpuErrCheck(cudaMallocManaged(&managedVector1, N * sizeof(int)));
    gpuErrCheck(cudaMallocManaged(&managedVector2, N * sizeof(int)));
    gpuErrCheck(cudaMallocManaged(&managedVector3, N * sizeof(int)));
    for (int i = 0; i < N; i++) { 
        hostVector[i] = N - i;
        managedVector1[i] = N - i;
        managedVector2[i] = N - i;
        managedVector3[i] = N - i;
    }

    // Run the vector kernel on the CPU
    std::sort(hostVector, hostVector + N);


    cudaOddEvenSortGlobalMemory <<<1, 1024>>> (managedVector1, N);
    gpuErrCheck(cudaPeekAtLastError());
    cudaOddEvenSortSharedMemory <<<1, 1024>>> (managedVector2, N);
    gpuErrCheck(cudaPeekAtLastError());
    cudaOddEvenSortSharedMemoryNoConflict <<<1, 1024>>> (managedVector3, N);
    gpuErrCheck(cudaPeekAtLastError());

    cudaDeviceSynchronize();

    // Compare CPU vs GPU result to see if we get the same result
    auto isValid = compareResultVec(hostVector, managedVector1, N);
    isValid = compareResultVec(hostVector, managedVector2, N);
    isValid = compareResultVec(hostVector, managedVector3, N);

    // Free memory on device
    gpuErrCheck(cudaFree(managedVector1));
    gpuErrCheck(cudaFree(managedVector2));
    gpuErrCheck(cudaFree(managedVector3));

    // Free memory on host
    delete[] hostVector;
}
