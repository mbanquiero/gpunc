
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h> 

// random int entre [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}


#define DIMW                64              // dimension de cada palabra 



/*
#define NUMBLOCKS        4
#define NUMTHREADS       32
#define DIMW             32              // dimension de cada palabra
*/

#define NUMBLOCKS           2
#define NUMTHREADS          32 

#define NUMPROC             11              // numero de procesadores
#define BLOCKSIZE           NUMTHREADS*NUMTHREADS
#define DIMP                NUMBLOCKS*BLOCKSIZE // cantidad de palabras por procesador
#define PSIZE               DIMW*DIMP       // memoria en bytes del procesador
#define BSIZE               NUMPROC*PSIZE   // memoria total del buffer



// globales
// en device (GPU)
extern char* dev_data0;                 // datos internos del procesador
extern char* dev_data1;                 
extern char* dev_data_bus;             // datos que ya pasaron el filtro y estan listos para comunicarse

// en el sistema (CPU)
extern char* data0;
extern char* data1;
extern char* data_bus;             // datos que ya pasaron el filtro y estan listos para comunicarse
extern int step;

cudaError_t device_alloc();
cudaError_t to_device();
cudaError_t to_cpu();
cudaError_t device_reset();
cudaError_t evolution_step();
cudaError_t comm_out_step();
cudaError_t comm_in_step();
void dibujar();
int hay_solucion();



// las tiene que proveer el ejemplo
__global__ void gopKernel(char* data_out, char* data, int step);
__global__ void filterOutKernel(char* data_out, char* data, int step);
__global__ void filterInKernel(char* data_out, char* data, int step);








