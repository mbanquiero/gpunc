
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h> 
#define RANDOM_DATA_SIZE      10000


static __device__ __inline__ int tea16(int val0, int val1)
{
    int v0 = val0;
    int v1 = val1;
    int s0 = 0;

    for (int n = 0; n < 16; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// random int entre [0, 2^24)
static __device__ __inline__ unsigned int lcg(unsigned int& prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}




// blockDim.x, y, z gives the number of threads in a block, in the particular direction
// gridDim.x, y, z gives the number of blocks in a grid, in the particular direction


#define MAX_DIMW            256 // maxima dimension de cada palabra
extern int DIMW;                // dimension de cada palabra 
extern int NUMPROC;            // numero de procesadores
extern int NUMBLOCKS;
extern int NUMTHREADS;
extern int BLOCKSIZE;
extern int DIMP;  
extern int PSIZE;
extern int BSIZE;




// globales
// en device (GPU)
extern char* dev_data0;                 // datos internos del procesador
extern char* dev_data1;                 
extern char* dev_data_bus;             // datos que ya pasaron el filtro y estan listos para comunicarse

// en el sistema (CPU)
extern char* data0;
extern char* data_bus;             // datos que ya pasaron el filtro y estan listos para comunicarse
extern int step;

cudaError_t device_alloc();
cudaError_t to_device_random();
cudaError_t to_device();
cudaError_t to_cpu();
cudaError_t device_reset();
cudaError_t evolution_step();
cudaError_t comm_out_step();
cudaError_t comm_in_step();
void dibujar();
int hay_solucion();
int RUN(bool pantalla=true);


__global__ void gopKernel(char* data_out, char* data, const int step, const int DIMW, const int NUMPROC);
__global__ void filterOutKernel(char* data_out, char* data, const int step, const int DIMW, const int NUMPROC);


// las tiene que proveer el ejemplo
__device__ char srule(char c, int N, const int NUMPROC);
__device__ bool passOutFilter(char* W, int N, const int DIMW, const int NUMPROC);
__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC,const int Nfrom);
__global__ void filterInKernel(char* data_out, char* data, const int step, const int DIMW, const int NUMPROC);
void domain_init();
void grid_layout();








