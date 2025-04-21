#include <stdio.h>
#include <cuda_runtime.h>
int main(){
    int n;
    cudaGetDeviceCount(&n);
    for(int i=0; i<n; ++i){
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("GPU %d: %s – SMs: %d\n", i, p.name, p.multiProcessorCount);
    }
    return 0;
}

