/* This program will do a vector addition on two vecotrs.
*  They have the same size N (defined in main).
*
*  +---------+   +---------+   +---------+
*  |111111111| + |222222222| = |333333333|
*  +---------+   +---------+   +---------+
*
*  vectorA   = all Ones
*  vectorB   = all Twos
*  vectorC   = all Three
*
*  NOTE: vectorX is an array of int and not std::vector
*
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <pac_helper.h>
using namespace std;


// CPU kernel function to add the elements of two arrays (called vectors)
void add(int* vectorA, int* vectorB, int* vectorC, int size)
{
    //PS: You know how to do this in AVX2, don't you?
    for (int i = 0; i < size; i++)
        vectorC[i] = vectorA[i] + vectorB[i];
}


// Kernel function to add the elements of two arrays using a Grid-Stride loop
__global__ void cudaAddRegular(int* vectorA, int* vectorB, int* vectorC, int size)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x)
        {
            vectorC[idx] = vectorA[idx] + vectorB[idx]; 
        }
}


// Kernel function to add the elements of two arrays using a Grid-Stride loop
// Use reinterpret_cast<int2*> to access 2 int values per thread
__global__ void cudaAddInt2(int* vectorA, int* vectorB, int* vectorC, int size)
{
    //ToDo: Implement the kernel
}

// Kernel function to add the elements of two arrays using a Grid-Stride loop
// The vectors are already casted to int2, so each thread accesses 2 int values
__global__ void cudaAddInt2casted(int2* vectorA, int2* vectorB, int2* vectorC, int size)
{
    //ToDo: Implement the kernel
}


// Kernel function to add the elements of two arrays using a Grid-Stride loop
// Use reinterpret_cast<int4*> to access 4 int values per thread
__global__ void cudaAddInt4(int* vectorA, int* vectorB, int* vectorC, int size)
{
    //ToDo: Implement the kernel
}


// Kernel function to add the elements of two arrays using a Grid-Stride loop
// The vectors are already casted to int4, so each thread accesses 4 int values
__global__ void cudaAddInt4casted(int4* vectorA, int4* vectorB, int4* vectorC, int size)
{
    //ToDo: Implement the kernel
}


// Templated function to support different kernels and vector types
// This function launches the kernel and measures the time taken for execution
template <typename KernelFunc, typename VectorType>
void launchKernel(KernelFunc kernel, string description, int blocks, int threads, VectorType* vectorA, VectorType* vectorB, VectorType* vectorC, int* hostVectorC, int size) {

    cudaEvent_t startEvent, stopEvent;
    gpuErrCheck(cudaEventCreate(&startEvent));
    gpuErrCheck(cudaEventCreate(&stopEvent));

    const int processedMB = size * sizeof(int) / 1024 / 1024 * 2;  // 2x as 1 read and 1 write
    float ms;

    gpuErrCheck(cudaEventRecord(startEvent, 0));

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        kernel <<<blocks, threads>>> (vectorA, vectorB, vectorC, size);
    }

    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaEventRecord(stopEvent, 0));
    gpuErrCheck(cudaEventSynchronize(stopEvent));
    int correct = compareResultVec(hostVectorC, reinterpret_cast<int*>(vectorC), size);

    if (correct == 0) {
        gpuErrCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));
        cout << description << " : " << processedMB * 100 / ms  << "GB/s bandwidth" << endl;
    }
    gpuErrCheck(cudaMemset(vectorC, 0, size * 4));
    gpuErrCheck(cudaEventDestroy(startEvent));
    gpuErrCheck(cudaEventDestroy(stopEvent));
}


int main(void)
{
    // Define the size of the vector: 67108864 elements
    int const N = 1 << 26;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);  //hard coded device 0

    // Alloc N times size of int Unified Memory
    int* managedVectorA;
    int* managedVectorB;
    int* managedVectorC;
    gpuErrCheck(cudaMallocManaged(&managedVectorA, N * sizeof(int)));
    gpuErrCheck(cudaMallocManaged(&managedVectorB, N * sizeof(int)));
    gpuErrCheck(cudaMallocManaged(&managedVectorC, N * sizeof(int)));

    // Allocate and prepare input/output arrays on host memory
    int* hostVectorCCPU = new int[N];
    for (int i = 0; i < N; i++) {
        managedVectorA[i] = 1;
        managedVectorB[i] = 2;
    }

    // Run the vector kernel on the CPU
    add(managedVectorA, managedVectorB, hostVectorCCPU, N);

    // ToDo: Play with block and thread sizes (T4 GPU supports 16 Blocks per SM and 1024 Threads per SM)
    launchKernel(cudaAddRegular,    "Warmup                     ", 1024, 512, managedVectorA, managedVectorB, managedVectorC, hostVectorCCPU, N);
    launchKernel(cudaAddRegular,    "cudaAddRegular[1024,512]   ", 1024, 512, managedVectorA, managedVectorB, managedVectorC, hostVectorCCPU, N);
    launchKernel(cudaAddInt2,       "cudaAddInt2[1024,512]      ", 1024, 512, managedVectorA, managedVectorB, managedVectorC, hostVectorCCPU, N);
    launchKernel(cudaAddInt2casted, "cudaAddInt2casted[1024,512]", 1024, 512, reinterpret_cast<int2*>(managedVectorA), reinterpret_cast<int2*>(managedVectorB), reinterpret_cast<int2*>(managedVectorC), hostVectorCCPU, N);
    launchKernel(cudaAddInt4,       "cudaAddInt4[1024,512]      ", 1024, 512, managedVectorA, managedVectorB, managedVectorC, hostVectorCCPU, N);
    launchKernel(cudaAddInt4casted, "cudaAddInt4casted[1024,512]", 1024, 512, reinterpret_cast<int4*>(managedVectorA), reinterpret_cast<int4*>(managedVectorB), reinterpret_cast<int4*>(managedVectorC), hostVectorCCPU, N);
    launchKernel(cudaAddInt4,       "cudaAddInt4[numSMs,512]    ", numSMs, 512, managedVectorA, managedVectorB, managedVectorC, hostVectorCCPU, N);
    launchKernel(cudaAddInt4,       "cudaAddInt4[8*numSMs,512]  ", 8*numSMs, 512, managedVectorA, managedVectorB, managedVectorC, hostVectorCCPU, N);
    launchKernel(cudaAddInt4,       "cudaAddInt4[16*numSMs,512] ", 16*numSMs, 512, managedVectorA, managedVectorB, managedVectorC, hostVectorCCPU, N);

    // Free memory
    gpuErrCheck(cudaFree(managedVectorA));
    gpuErrCheck(cudaFree(managedVectorB));
    gpuErrCheck(cudaFree(managedVectorC));
    delete[] hostVectorCCPU;
}
