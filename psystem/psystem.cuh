
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h> 

extern int NUMBLOCKS;
extern int NUMTHREADS;

#define RANDOM_DATA_SIZE      1024


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


struct pobject
{
    char c;			// caracter o simbolo
    int n;			// numero de repeticiones
    int h;			// nro de membrana
};


#define MAX_RULE_LEN	10
#define MAX_NODOS_HIJOS	10
#define MAX_MEMBRANAS	16
#define MAX_P_OBJECTS	1024
#define MAX_RULES		256


// a -> a (b , in) (c , in) (c, in)
// aa -> (a , out)(a , out)

struct prule
{
    // parte izquierda
    int h;						// nro de membrana
    pobject u[MAX_RULE_LEN];	// parte izquierda

    // parte derecha
    // v.h = h si es una regla de sustitucion
    // v.h = w que pertenece a la membra h, si es de comm in 
    // v.h = -1 indica que es de salida = padre de h
    pobject v[MAX_RULE_LEN];
    bool disuelve;

};

struct p_membrana
{
    int parent;							// membrana padre , -1 si es la piel
    int H[MAX_NODOS_HIJOS];			// membranas hijos
    int cant_H;


};


// A > B
struct p_order
{
    int A, B;
};

struct CELL
{
    // estructura de membranas
    int cant_M;
    p_membrana M[MAX_MEMBRANAS];
    // reglas de computacion
    int cant_R;
    prule R[MAX_RULES];
    // prioridad de las reglas
    int cant_Pr;
    p_order Pr[100];
};



#define MAX_OBJECTS           32        // maxima cantidad de objetos en una celula
#define MAX_CELLS             1024        // maxima cantidad de celulas
#define BSIZE                 MAX_CELLS * MAX_OBJECTS*sizeof(pobject)
// globales
// en device (GPU)
// membrane objects
extern pobject* dev_m_objects;
extern char* dev_dbg_buffer;                    // para debug
extern CELL* dev_cell;                      // una sola celula

extern int step;

// en el sistema (CPU)
extern pobject* m_objects;
extern CELL* cell;                      // una sola celula
extern char *dbg_buffer;                    // para debug

cudaError_t device_alloc();
cudaError_t to_device_random();
cudaError_t to_device();
cudaError_t to_cpu();
cudaError_t device_reset();
cudaError_t compute_step();
void p_print(int n,int h);
void print_report();


