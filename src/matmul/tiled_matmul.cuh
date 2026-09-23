#ifndef TILED_MATMUL_CUH
#define TILED_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

/*
    ==================================================
        TILED MATRIX MULTIPLICATION KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix

    *Description here
*/

template <int TILE_WIDTH>
__global__ void tiledMatmulKernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    // if((row >= M) || (col >= N)) - handle this later
    //     return;

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    float sum{};

    for(int tile_idx{}; tile_idx < ((K + TILE_WIDTH - 1) / TILE_WIDTH); ++tile_idx) {
        //Loading into shared memory
        Ads[ty][tx] = A[row * K + (tile_idx * TILE_WIDTH + tx)];
        Bds[ty][tx] = B[(N * (tile_idx * TILE_WIDTH + ty)) + col];  // Need to implement boundary conditions

        __syncthreads();

        for(int i{}; i < TILE_WIDTH; ++i) {
            sum += Ads[ty][i] * Bds[i][tx];
        }

        __syncthreads();

    }

    C[row * N + col] = sum;
}

template <int TILE_WIDTH>
void tiledMatmulGPU(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K,
) {
    dim3 dimBlock(
        TILE_WIDTH,
        TILE_WIDTH,
        1
    );

    dim3 dimGrid(
        (N + dimBlock.x - 1) / dimBlock.x,
        (M + dimBlock.y - 1) / dimBlock.y,
        1
    ); 

    tiledMatmulKernel<TILE_WIDTH><<<dimGrid, dimBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess)
        std::cout << "Tiled Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';

    err = cudaDeviceSynchronize();

    if(err != cudaSuccess)
        std::cout << "Tiled Matmul Kernel Execution Error: " << cudaGetErrorString(err) << '\n';

}

#endif