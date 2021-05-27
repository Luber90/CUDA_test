/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /**
  * Matrix multiplication: C = A * B.
  * Host code.
  *
  * This sample implements matrix multiplication as described in Chapter 3
  * of the programming guide.
  * It has been written for clarity of exposition to illustrate various CUDA
  * programming principles, not with the goal of providing the most
  * performant generic kernel for matrix multiplication.
  *
  * See also:
  * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
  * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
  * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
  */

  // System includes
#define WIN32
#include <stdio.h>
#include <assert.h>


// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */


template <int BLOCK_SIZE> __global__ void
matrixMulCUDAk2l4(float* C, float* A, float* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;


    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    float Csub1 = 0;
    float Csub2 = 0;
    float Csub3 = 0;
    float Csub4 = 0;
    float Csub5 = 0;
    float Csub6 = 0;
    float Csub7 = 0;
    float Csub8 = 0;


    __shared__ float As1[2 * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE][4 * BLOCK_SIZE];
    __shared__ float As2[2 * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs2[BLOCK_SIZE][4 * BLOCK_SIZE];

    float(*currentAs)[BLOCK_SIZE];
    float(*currentBs)[4 * BLOCK_SIZE];
    float(*nextAs)[BLOCK_SIZE];
    float(*nextBs)[4 * BLOCK_SIZE];
    float(*tempA)[BLOCK_SIZE];
    float(*tempB)[4 * BLOCK_SIZE];

    As1[ty][tx] = A[aBegin + wA * ty + tx];
    As1[ty + BLOCK_SIZE][tx] = A[aBegin + wA * (ty + wB / 2) + tx];
    Bs1[ty][tx] = B[bBegin + wB * ty + tx];
    Bs1[ty][tx + BLOCK_SIZE] = B[bBegin + wA * (ty)+tx + wB / 4];
    Bs1[ty][tx + 2*BLOCK_SIZE] = B[bBegin + wA * (ty)+tx + 2*wB / 4];
    Bs1[ty][tx + 3*BLOCK_SIZE] = B[bBegin + wA * (ty)+tx + 3*wB / 4];

    currentAs = As2;
    currentBs = Bs2;
    nextAs = As1;
    nextBs = Bs1;

    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {

        tempA = currentAs;
        currentAs = nextAs;
        nextAs = tempA;

        tempB = currentBs;
        currentBs = nextBs;
        nextBs = tempB;

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub1 += currentAs[ty][k] * currentBs[k][tx];
            Csub2 += currentAs[ty][k] * currentBs[k][tx+BLOCK_SIZE];
            Csub3 += currentAs[ty][k] * currentBs[k][tx + 2* BLOCK_SIZE];
            Csub4 += currentAs[ty][k] * currentBs[k][tx + 3* BLOCK_SIZE];
            Csub5 += currentAs[ty + BLOCK_SIZE][k] * currentBs[k][tx];
            Csub6 += currentAs[ty + BLOCK_SIZE][k] * currentBs[k][tx+ BLOCK_SIZE];
            Csub7 += currentAs[ty + BLOCK_SIZE][k] * currentBs[k][tx + 2* BLOCK_SIZE];
            Csub8 += currentAs[ty + BLOCK_SIZE][k] * currentBs[k][tx + 3* BLOCK_SIZE];
        }
        __syncthreads();
        nextAs[ty][tx] = A[a + aStep + wA * ty + tx];
        nextAs[ty + BLOCK_SIZE][tx] = A[a + aStep + wA * (ty + wB / 2) + tx];
        nextBs[ty][tx] = B[b + bStep + wB * ty + tx];
        nextBs[ty][tx + BLOCK_SIZE] = B[b + bStep + wB * ty + tx + wB / 4];
        nextBs[ty][tx + 2*BLOCK_SIZE] = B[b + bStep + wB * ty + tx + 2*wB / 4];
        nextBs[ty][tx + 3*BLOCK_SIZE] = B[b + bStep + wB * ty + tx + 3*wB / 4];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub1;
    C[c + wB * ty + tx + wB/4] = Csub2;
    C[c + wB * ty + tx + 2*wB / 4] = Csub3;
    C[c + wB * ty + tx + 3*wB / 4] = Csub4;
    C[c + wB * (ty + wB / 2) + tx] = Csub5;
    C[c + wB * (ty + wB / 2) + tx + wB / 4] = Csub6;
    C[c + wB * (ty + wB / 2) + tx + 2*wB / 4] = Csub7;
    C[c + wB * (ty + wB / 2) + tx + 3*wB / 4] = Csub8;
}


template <int BLOCK_SIZE> __global__ void
matrixMulCUDAk2l2(float* C, float* A, float* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;


    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    float Csub1 = 0;
    float Csub2 = 0;
    float Csub3 = 0;
    float Csub4 = 0;


    __shared__ float As1[2 * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE][2*BLOCK_SIZE];
    __shared__ float As2[2 * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs2[BLOCK_SIZE][2*BLOCK_SIZE];

    float(*currentAs)[BLOCK_SIZE];
    float(*currentBs)[2*BLOCK_SIZE];
    float(*nextAs)[BLOCK_SIZE];
    float(*nextBs)[2*BLOCK_SIZE];
    float(*tempA)[BLOCK_SIZE];
    float(*tempB)[2*BLOCK_SIZE];

    As1[ty][tx] = A[aBegin + wA * ty + tx];
    As1[ty + BLOCK_SIZE][tx] = A[aBegin + wA * (ty + wB / 2) + tx];
    Bs1[ty][tx] = B[bBegin + wB * ty + tx];
    Bs1[ty][tx + BLOCK_SIZE] = B[bBegin + wA * (ty) + tx + wB / 2];

    currentAs = As2;
    currentBs = Bs2;
    nextAs = As1;
    nextBs = Bs1;

    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {

        tempA = currentAs;
        currentAs = nextAs;
        nextAs = tempA;

        tempB = currentBs;
        currentBs = nextBs;
        nextBs = tempB;

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub1 += currentAs[ty][k] * currentBs[k][tx];
            Csub2 += currentAs[ty + BLOCK_SIZE][k] * currentBs[k][tx];
            Csub3 += currentAs[ty][k] * currentBs[k][tx+BLOCK_SIZE];
            Csub4 += currentAs[ty + BLOCK_SIZE][k] * currentBs[k][tx + BLOCK_SIZE];
        }
        __syncthreads();
        nextAs[ty][tx] = A[a + aStep + wA * ty + tx];
        nextAs[ty + BLOCK_SIZE][tx] = A[a + aStep + wA * (ty + wB / 2) + tx];
        nextBs[ty][tx] = B[b + bStep + wB * ty + tx];
        nextBs[ty][tx + BLOCK_SIZE] = B[b + bStep + wB * ty + tx + wB / 2];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub1;
    C[c + wB * (ty + wB / 2) + tx] = Csub2;
    C[c + wB * ty + tx + wB/2] = Csub3;
    C[c + wB * (ty + wB / 2) + tx + wB / 2] = Csub4;
}

template <int BLOCK_SIZE> __global__ void
matrixMulCUDAk2l1(float* C, float* A, float* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;


    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    float Csub1 = 0;
    float Csub2 = 0;


    __shared__ float As1[2*BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float As2[2 * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs2[BLOCK_SIZE][BLOCK_SIZE];

    float(*currentAs)[BLOCK_SIZE];
    float(*currentBs)[BLOCK_SIZE];
    float(*nextAs)[BLOCK_SIZE];
    float(*nextBs)[BLOCK_SIZE];
    float(*temp)[BLOCK_SIZE];

    As1[ty][tx] = A[aBegin + wA * ty + tx];
    As1[ty+ BLOCK_SIZE][tx] = A[aBegin + wA * (ty+ wB / 2) + tx];
    Bs1[ty][tx] = B[bBegin + wB * ty + tx];

    currentAs = As2;
    currentBs = Bs2;
    nextAs = As1;
    nextBs = Bs1;

    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {

        temp = currentAs;
        currentAs = nextAs;
        nextAs = temp;

        temp = currentBs;
        currentBs = nextBs;
        nextBs = temp;

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub1 += currentAs[ty][k] * currentBs[k][tx];
            Csub2 += currentAs[ty + BLOCK_SIZE][k] * currentBs[k][tx];
        }
        __syncthreads();
        nextAs[ty][tx] = A[a + aStep + wA * ty + tx];
        nextAs[ty + BLOCK_SIZE][tx] = A[a + aStep + wA * (ty + wB / 2) + tx];
        nextBs[ty][tx] = B[b + bStep + wB * ty + tx];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub1;
    C[c + wB * (ty + wB/2) + tx] = Csub2;
}

template <int BLOCK_SIZE> __global__ void
matrixMulCUDAk1l1(float* C, float* A, float* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;


    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    float Csub = 0;


    __shared__ float As1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float As2[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs2[BLOCK_SIZE][BLOCK_SIZE];

    float(*currentAs)[BLOCK_SIZE];
    float(*currentBs)[BLOCK_SIZE];
    float(*nextAs)[BLOCK_SIZE];
    float(*nextBs)[BLOCK_SIZE];
    float(*temp)[BLOCK_SIZE];

    As1[ty][tx] = A[aBegin + wA * ty + tx];
    Bs1[ty][tx] = B[bBegin + wB * ty + tx];

    currentAs = As2;
    currentBs = Bs2;
    nextAs = As1;
    nextBs = Bs1;

    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {

        temp = currentAs;
        currentAs = nextAs;
        nextAs = temp;

        temp = currentBs;
        currentBs = nextBs;
        nextBs = temp;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += currentAs[ty][k] * currentBs[k][tx];
        }
        __syncthreads();
        nextAs[ty][tx] = A[a + aStep + wA * ty + tx];
        nextBs[ty][tx] = B[b + bStep + wB * ty + tx];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


void constantInit(float* data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int block_size, dim3& dimsA, dim3& dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float* d_A, * d_B, * d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C = (float*)malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void**)&d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y );


    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute the kernel
    int nIter = 10;

    for (int j = 0; j < nIter; j++)
    {
        if (block_size == 8) {
            matrixMulCUDAk1l1<8> << <grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else if (block_size == 16)
        {
            matrixMulCUDAk1l1<16> <<<grid, threads>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y/2);
            //matrixMulCUDAk2l1<16> << <grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //dim3 grid(dimsB.x / threads.x/2, dimsA.y / threads.y/2);
            //matrixMulCUDAk2l2<16> << <grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //dim3 grid(dimsB.x / threads.x / 4, dimsA.y / threads.y / 2);
            //matrixMulCUDAk2l4<16> << <grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            matrixMulCUDAk1l1<32> <<<grid, threads>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y/2);
            //matrixMulCUDAk2l1<32> << <grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //dim3 grid(dimsB.x / threads.x/2, dimsA.y / threads.y/2);
            //matrixMulCUDAk2l2<32> << <grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //dim3 grid(dimsB.x / threads.x / 4, dimsA.y / threads.y / 2);
            //matrixMulCUDAk2l4<32> << <grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }
    cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Checking computed result for correctness: ");
    bool correct = true;

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        if (fabs(h_C[i] - (dimsA.x * valB)) > 1e-3)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-5\n", i, h_C[i], dimsA.x * valB);
            correct = false;
        }
    }

    printf("%s\n", correct ? "OK" : "FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

int cpuMatrixMul() {


    return 0;
}


/**
 * Program main
 */
int main(int argc, char** argv)
{

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    cudaSetDevice(devID);
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    int block_size = 16;
    dim3 dimsA(1024, 1024, 1);
    dim3 dimsB(1024, 1024, 1);


    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
            dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(block_size, dimsA, dimsB);

    exit(matrix_result);
}
