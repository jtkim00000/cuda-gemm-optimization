#ifndef MATMUL_H
#define MATMUL_H

void naiveMatmulGPU(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K
);

#endif